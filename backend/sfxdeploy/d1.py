"""The D1 system (deployed 2026-10-15; the thesis hero): an ensemble of two fine-tuned twins, the MATPAC++ twin and the BEATs twin (seed 123 each), plus the shared soft-DTW head and CSLS.
ZERO-SHOT on SFX: no weight of this system was trained on or adapted to any SFX or qvim-dev recording (towers and head: VimSketch only, epoch chosen on held-out VimSketch concepts).

  T(q, g)     = z( z(cos(MATPAC_q q, MATPAC_r g)) + z(cos(BEATs_q q, BEATs_r g)) )          the ensemble tower, z per query row over the gallery
  H(q, g)     = softDTW(F_q q, F_r g), gamma 0.1                                            frozen HuBERT-base layer 9 -> two FrameNets ('clean' head, seed 42)
  score(q, g) = z(T) + z(H) - 0.5 r_T(g)                                                    CSLS: r_T(g) = mean of g's top-10 fused scores over the reference queries
                                                                                              (SFX: the 84 benchmark queries, each query's own targets left out; qvim-dev: the 979 queries)

Every numeric step of the scoring is the research code's (SFX-fusion/pair_matrix/stageA_cells.py, tower_slot_swap.py, sfx_adapt/common.py); the soft-DTW utilities are softdtw_head.py's
(verbatim from the research code). Regime: every clip encoded ALONE; the towers see the clip cut to 10 s (no padding), the head frames see it at natural length; trim 15 dB + RMS 0.1.
scripts/build_d1_index.py, scripts/build_qdev_d1_index.py and scripts/verify_d1_search.py check the served numbers against the research export (models/d1/reference_tables.npz)."""
import json, hashlib, threading
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from . import config as C
from .d1_towers import D1_DIR, load_tower
from .softdtw_head import (FrameNet, FrameBank, HUBERT, LAYER, K_CSLS, SR, L10, sdtw_matrix, zrow, cos_np, arrays_to_frames, frames_to_arrays)
from .precision import cudnn_tf32

INDEX_PATH = C.INDEX_DIR / 'sfx_gallery_d1.npz'
OVERLAY_PATH = C.INDEX_DIR / 'sfx_gallery_d1_submitted.npz'
# cuDNN TF32 convolutions ON = PyTorch's default, the condition the research matrices were computed under. Measured 2026-10-15 on 84 queries x 400 gallery clips against models/d1/reference_tables.npz:
# TF32 on: max |diff| MATPAC++ 1.9e-7, BEATs 2.1e-6, head 1.1e-6; TF32 off: 1.9e-7, 2.1e-5, 7.2e-4 (exact fp32 is the LESS faithful setting here, unlike for the Phase 40 EfficientAT tower).
TF32_TOWERS = True
TF32_HEAD = True


def sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()[:12]


def fuse(s_a, s_b, s_h):
    """[Q, N] cosines of the two towers and the soft-DTW matrix -> (ensemble tower z, fused score before CSLS), both [Q, N]"""
    t = zrow(zrow(s_a) + zrow(s_b))
    return t, zrow(t) + zrow(s_h)


def csls_hub(fz, excl=None, k=K_CSLS):
    """r_T(g): mean of the top-k fused scores of column g over the reference queries (rows of fz); `excl` [Q, N] bool = entries left out (each query's own targets)"""
    X = np.where(np.isfinite(fz), fz, -np.inf)
    if excl is not None:
        X = np.where(excl, -np.inf, X)
    top = -np.sort(-X, axis=0)[:k]
    return np.nan_to_num(np.nanmean(np.where(np.isfinite(top), top, np.nan), 0), nan=0.0).astype('float32')


class D1Model:
    """Four fine-tuned towers + the head. Frozen, fp32, eval mode; every clip encoded alone."""
    def __init__(self, device=None):
        from transformers import AutoModel, AutoFeatureExtractor
        self.device = torch.device(device or 'cuda')
        self.cfg = json.load(open(D1_DIR / 'config.json'))
        for name, want in self.cfg['sha12'].items():          # refuse to serve weights other than the documented ones
            assert sha12(D1_DIR / name) == want, f'{name}: checksum mismatch'
        self.mat_q, self.mat_r = (load_tower('matpac', D1_DIR / f'matpac_{s}.pt', self.device) for s in ('mimic', 'ref'))
        self.bea_q, self.bea_r = (load_tower('beats', D1_DIR / f'beats_{s}.pt', self.device) for s in ('mimic', 'ref'))
        # head: ONE frozen HuBERT-base (truncated to layer 9) + two FrameNets (the research head held two identical frozen copies)
        st = torch.load(D1_DIR / 'head_clean_s42.pt', map_location='cpu', weights_only=True)
        self.proc = AutoFeatureExtractor.from_pretrained(HUBERT)
        hub = AutoModel.from_pretrained(HUBERT, layerdrop=0.0, hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0)
        hub.encoder.layers = hub.encoder.layers[:LAYER]
        hub.config.apply_spec_augment = False
        hub.config.layerdrop = 0.0
        for mod in hub.modules():
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
        self.hub = hub
        self.net_q, self.net_r = FrameNet(), FrameNet()
        self.net_q.load_state_dict({k[2:]: v for k, v in st.items() if k.startswith('q.')}, strict=True)
        self.net_r.load_state_dict({k[2:]: v for k, v in st.items() if k.startswith('r.')}, strict=True)
        for mod in (self.hub, self.net_q, self.net_r):
            mod.to(self.device).eval()
            for p in mod.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _raw(self, w):                          # common.HeadSide.raw verbatim
        iv = self.proc([w], sampling_rate=SR, return_tensors='pt', padding=False)['input_values'].to(self.device)
        h = self.hub(iv, output_hidden_states=True).hidden_states[-1][0]
        T = h.shape[0] // 2 * 2
        h = (h[:T].reshape(T // 2, 2, -1).mean(1) if T >= 2 else h).half().float()
        if h.shape[0] % 2:
            h = torch.cat([h, h[-1:]])
        return h.reshape(-1, 2, h.shape[-1]).mean(1).half().float()

    @staticmethod
    def _cut10(w):                              # the towers' regime ('cut10' in the study): cut to 10 s, never padded
        return np.ascontiguousarray(w[:L10]).astype('float32')

    @torch.no_grad()
    def query(self, w):
        """front-end-processed query wave -> (MATPAC++ embedding [3840], BEATs embedding [768], head frames fp16 [T, 512] on the GPU)"""
        wt = self._cut10(w)
        with cudnn_tf32(TF32_TOWERS):
            za = self.mat_q.encode_audio([wt]).float()[0].cpu().numpy(); zb = self.bea_q.encode_audio([wt]).float()[0].cpu().numpy()
        with cudnn_tf32(TF32_HEAD):
            return za, zb, self.net_q(self._raw(w)).half()

    @torch.no_grad()
    def item(self, w, w_head=None):
        """front-end-processed gallery wave -> (MATPAC++ embedding [3840], BEATs embedding [768], head frames fp16 [T, 512] on the CPU).
        w_head: a different wave for the head's frames (only the qvim-dev index builder uses it: the research frame cache of the head did not cap clips at 30 s, the tower input did)"""
        wt = self._cut10(w)
        with cudnn_tf32(TF32_TOWERS):
            ga = self.mat_r.encode_audio([wt]).float()[0].cpu().numpy(); gb = self.bea_r.encode_audio([wt]).float()[0].cpu().numpy()
        with cudnn_tf32(TF32_HEAD):
            return ga, gb, self.net_r(self._raw(w if w_head is None else w_head)).half().cpu()


def tower_cos(q_a, q_b, G_A, G_B):
    """-> (cosine rows of the two towers [1, N] each)"""
    return cos_np(q_a[None], G_A), cos_np(q_b[None], G_B)


class D1Gallery:
    """The live SFX gallery as D1 sees it: the two reference-tower tables G_A / G_B, the head frame bank, CSLS r_T, plus the 84 reference queries' tables (needed to give a newly ingested clip its own r_T).
    The base tables come from scripts/build_d1_index.py and are never mutated; ingested clips go to an overlay file. `registry` = the ordered file names of the ingested clips (the live gallery is
    base files + registry); a clip whose tables are missing from the overlay is re-encoded from its audio at start-up (this is also how clips ingested before D1 existed were migrated)."""
    def __init__(self, model, registry, index_path=INDEX_PATH, overlay_path=OVERLAY_PATH, audio_root=C.SFX_GALLERY):
        from .audio import load_audio
        self.model, self.overlay_path, self.audio_root, self._load_audio = model, Path(overlay_path), Path(audio_root), load_audio
        z = np.load(index_path, allow_pickle=True)
        self.meta = json.load(open(Path(index_path).with_suffix('.meta.json')))
        assert self.meta['model_sha12'] == model.cfg['sha12'], 'D1 index was built with different weights -- rebuild it'
        base_files = [str(f) for f in z['files']]
        self.ref_A, self.ref_B = z['ref_A'], z['ref_B']
        self.ref_frames = [f.to(model.device) for f in arrays_to_frames(z['ref_frames'], z['ref_lengths'])]
        self.ref_stats = {k: z[k].astype('float64') for k in ('mu_a', 'sd_a', 'mu_b', 'sd_b', 'mu_u', 'sd_u', 'mu_t', 'sd_t', 'mu_h', 'sd_h')}
        bank = FrameBank.build(arrays_to_frames(z['frames'], z['lengths']), model.device)
        self._state_lock = threading.Lock()
        self.n_base = len(base_files)
        ov_files, ov_A, ov_B, ov_r, ov_frames = [], np.zeros((0, 3840), 'float32'), np.zeros((0, 768), 'float32'), np.zeros(0, 'float32'), []
        if self.overlay_path.exists():
            o = np.load(self.overlay_path, allow_pickle=True)
            ov_files, ov_A, ov_B, ov_r, ov_frames = [str(f) for f in o['files']], o['G_A'], o['G_B'], o['r_T'], arrays_to_frames(o['frames'], o['lengths'])
        registry = list(registry); k = 0
        while k < min(len(ov_files), len(registry)) and ov_files[k] == registry[k]:
            k += 1
        if k < len(ov_files):
            print(f'[d1] WARNING: dropping {len(ov_files) - k} overlay entr(y/ies) that do not match the registry: {ov_files[k:]}')
        self.files = base_files + ov_files[:k]; self.G_A = np.concatenate([z['G_A'], ov_A[:k]]); self.G_B = np.concatenate([z['G_B'], ov_B[:k]])
        self.r_T = np.concatenate([z['r_T'], ov_r[:k]]); self.bank = FrameBank(model.device, bank.base, bank.n_base, ov_frames[:k])
        for f in registry[k:]:
            print(f'[d1] backfilling ingested clip {f} (no D1 entry yet)')
            self.add(f, self.audio_root / f, persist=False)
        if len(registry) > k or k < len(ov_files):
            self._persist()
        assert self.files == base_files + registry

    def item_tables(self, wave):
        """-> (G_A row [3840], G_B row [768], head frames fp16 [T, 512] CPU, r_T value) for one new clip: its score against the 84 reference queries, standardised with the reference queries'
        base-gallery row statistics, fused like a live query and reduced to the CSLS hub term"""
        g_a, g_b, g_f = self.model.item(wave)
        s_a = cos_np(self.ref_A, g_a[None])[:, 0].astype('float64'); s_b = cos_np(self.ref_B, g_b[None])[:, 0].astype('float64')
        s_h = sdtw_matrix(self.ref_frames, [g_f], self.model.device)[:, 0].astype('float64'); R, e = self.ref_stats, 1e-12
        za = (s_a - R['mu_a']) / (R['sd_a'] + e); zb = (s_b - R['mu_b']) / (R['sd_b'] + e)
        t = ((za + zb) - R['mu_u']) / (R['sd_u'] + e)
        fz = (t - R['mu_t']) / (R['sd_t'] + e) + (s_h - R['mu_h']) / (R['sd_h'] + e)
        return g_a, g_b, g_f, float(np.sort(fz)[-K_CSLS:].mean())

    def commit(self, file, g_a, g_b, g_f, r, persist=True):
        """append one clip's tables (from item_tables) to the live state and persist the overlay"""
        with self._state_lock:
            self.files = self.files + [file]
            self.G_A = np.concatenate([self.G_A, g_a[None].astype('float32')]); self.G_B = np.concatenate([self.G_B, g_b[None].astype('float32')])
            self.r_T = np.concatenate([self.r_T, np.array([r], 'float32')])
            self.bank = self.bank.with_appended(g_f)
        if persist:
            self._persist()

    def add(self, file, path, persist=True):
        wave = self._load_audio(str(path))
        if wave is None:
            raise ValueError(f'audio front-end rejected {path}')
        self.commit(file, *self.item_tables(wave), persist=persist)

    def _persist(self):
        with self._state_lock:
            nb = len(self.bank.extra_frames); n = len(self.files)
            files, A, B, r = self.files[n - nb:], self.G_A[n - nb:], self.G_B[n - nb:], self.r_T[n - nb:]
            cat, lens = frames_to_arrays(list(self.bank.extra_frames))
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.overlay_path.with_name(self.overlay_path.stem + '.tmp.npz')
        np.savez(tmp, files=np.array(files, dtype=object), G_A=A, G_B=B, r_T=r, frames=cat, lengths=lens); tmp.replace(self.overlay_path)

    def snapshot(self):
        with self._state_lock:
            return self.files, self.G_A, self.G_B, self.r_T, self.bank

    def score(self, q, snap):
        """q = model.query(wave); snap = snapshot() -> (system score float32 [N] = z(T) + z(H) - 0.5 r_T, mean tower cosine float32 [N])"""
        q_a, q_b, q_f = q
        files, G_A, G_B, r_T, bank = snap
        s_a, s_b = tower_cos(q_a, q_b, G_A, G_B)
        _, fz = fuse(s_a, s_b, bank.sims(q_f)[None])
        return (fz - 0.5 * r_T[None, :])[0], (0.5 * (s_a + s_b))[0]
