"""Shared soft-DTW head utilities of the served systems (D1 since 2026-10-15; the retired Phase 40 system used the same functions): the FrameNet of the head, the per-row z-score, cosine with the
NaN rule, the padded soft-DTW matrix, and the immutable GPU frame bank of the gallery. Every function is copied verbatim from the research workspace (SFX-fusion/sfx_adapt/common.py, adapt.py).
The head itself = ONE frozen HuBERT-base (layer 9, 50 -> 25 -> 12.5 Hz, fp16 frame rounding) + two FrameNets, soft-DTW gamma 0.1 (see d1.py D1Model). Soft-DTW of one pair does not depend on how pairs are
batched (cells beyond a pair's own lengths never reach its end cell), so the gallery is kept on the GPU in length-sorted buckets for speed."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HUBERT, LAYER, GAMMA, K_CSLS, SR = 'facebook/hubert-base-ls960', 9, 0.1, 10, 16000
L10 = 10 * SR                                    # qvim-dev: gallery clips for the tower are cut/padded to exactly 10 s


class FrameNet(nn.Module):                      # common.py verbatim
    def __init__(s, din=768, hid=1024, dout=512, p=0.1):
        super().__init__(); s.ln = nn.LayerNorm(din)
        s.mlp = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, dout))

    def forward(s, x):
        return F.normalize(s.mlp(s.ln(x)), dim=-1)


def zrow(S):                                    # common.py verbatim
    mu = np.nanmean(S, 1, keepdims=True); sd = np.nanstd(S, 1, keepdims=True)
    return (S - mu) / (sd + 1e-12)


def cos_np(Zq, Zg):                             # adapt.py verbatim
    Zq = Zq / np.linalg.norm(Zq, axis=1, keepdims=True); Zg = Zg / (np.linalg.norm(Zg, axis=1, keepdims=True) + 1e-12)
    S = Zq @ Zg.T
    S[:, ~np.isfinite(Zg).all(1)] = np.nan
    return S


def pack(outs, device):                         # common.py verbatim (device explicit)
    lens = torch.tensor([o.shape[0] for o in outs], device=device)
    T = int(lens.max()); D = outs[0].shape[-1]
    x = torch.zeros(len(outs), T, D, device=device, dtype=outs[0].dtype)
    for i, o in enumerate(outs):
        x[i, :o.shape[0]] = o
    return x, lens


def sdtw_pairs(zq, lq, zr, lr):                 # common.py verbatim
    from softdtw import soft_dtw_similarity
    cos = torch.einsum('bqd,crd->bcqr', zq, zr)
    B, Cn, Tq, Tr = cos.shape
    return soft_dtw_similarity(cos.reshape(B * Cn, Tq, Tr), lq[:, None].expand(B, Cn).reshape(-1),
                               lr[None, :].expand(B, Cn).reshape(-1), GAMMA).reshape(B, Cn)


@torch.no_grad()
def sdtw_matrix(q_list, r_list, device, budget=1.2e8):   # common.py verbatim: lists of [T, D] frame tensors -> [Q, R]
    Q, R = len(q_list), len(r_list)
    out = np.full((Q, R), np.nan, 'float32')
    zq, lq = pack([x.float() for x in q_list], device); n = zq.shape[1]
    order = sorted(range(R), key=lambda i: r_list[i].shape[0])
    i = 0
    while i < R:
        m = r_list[order[min(R - 1, i)]].shape[0]
        step = max(1, int(budget // (Q * n * max(m, 1))))
        idx = order[i:i + step]
        m = max(r_list[j].shape[0] for j in idx)
        step = max(1, int(budget // (Q * n * m)))
        idx = order[i:i + step]
        zr, lr = pack([r_list[j].to(device).float() for j in idx], device)
        out[:, idx] = sdtw_pairs(zq, lq, zr, lr).float().cpu().numpy()
        i += len(idx)
    return out


class FrameBank:
    """Gallery head frames, fp16, on the GPU, in length-sorted buckets (each bucket padded to its own longest clip).
    Immutable: with_appended() returns a new bank sharing the base buckets, so a reader holding an old bank is never
    affected by a concurrent ingest."""
    BUCKET_CAP = 131072          # clips x frames per bucket (its fp32 copy during scoring is <= 256 MB)
    BUDGET = 5e7                 # query frames x clips x frames per soft-DTW call

    def __init__(self, device, base_buckets, n_base, extra_frames=()):
        self.device, self.base, self.n_base, self.extra_frames = device, base_buckets, n_base, tuple(extra_frames)
        self.extra = self._buckets(self.extra_frames, n_base, device) if self.extra_frames else []
        self.n = n_base + len(self.extra_frames)

    @classmethod
    def build(cls, frames, device):
        return cls(device, cls._buckets(frames, 0, device), len(frames))

    @classmethod
    def _buckets(cls, frames, offset, device):
        order = sorted(range(len(frames)), key=lambda i: frames[i].shape[0]); out = []; s = 0
        while s < len(order):
            e = s + 1
            while e < len(order) and (e + 1 - s) * frames[order[e]].shape[0] <= cls.BUCKET_CAP:
                e += 1
            idx = order[s:e]
            z, l = pack([frames[i].to(device) for i in idx], device)
            out.append((torch.tensor([offset + i for i in idx], device=device), z, l))
            s = e
        return out

    def with_appended(self, fr):
        return FrameBank(self.device, self.base, self.n_base, self.extra_frames + (fr,))

    @torch.no_grad()
    def sims(self, qf):
        """qf: fp16 [n, D] query frames -> float32 numpy [self.n] soft-DTW similarities (common.sdtw_pairs per bucket slice)"""
        zq = qf.float()[None]; n = zq.shape[1]; lq = torch.tensor([n], device=zq.device)
        out = torch.full((self.n,), float('nan'), device=zq.device)
        for idx, Z, lens in self.base + self.extra:
            step = max(1, int(self.BUDGET // (n * Z.shape[1])))
            for s in range(0, Z.shape[0], step):
                out[idx[s:s + step]] = sdtw_pairs(zq, lq, Z[s:s + step].float(), lens[s:s + step])[0].float()
        return out.cpu().numpy()


def frames_to_arrays(frames):
    return (np.concatenate([f.cpu().numpy() for f in frames]).astype('float16') if frames else np.zeros((0, 512), 'float16'),
            np.array([f.shape[0] for f in frames], 'int64'))


def arrays_to_frames(cat, lens):
    off = np.concatenate([[0], np.cumsum(lens)])
    return [torch.from_numpy(np.ascontiguousarray(cat[off[i]:off[i + 1]])) for i in range(len(lens))]
