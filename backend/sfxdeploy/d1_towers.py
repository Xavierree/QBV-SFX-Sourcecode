"""The two tower families of the D1 system, as the research workspace trained and evaluated them (SFX-fusion/pair_matrix/mm_towers.py; PROTOCOL.md of that study):

  'matpac'  MATPAC++ general-audio encoder (self-supervised AudioSet ViT-B; own port of its inference code, no remote code): 16 kHz, 80-bin log-mel, 6 s chunks, mean over the
            time tokens of the last layer with the 5 frequency patches concatenated -> 3,840-d. A clip is used at its own length, grouped by number of mel frames, so its embedding
            is the same alone or in a batch.
  'beats'   BEATs iter3+ AS2M (Chen et al. 2022, MIT; vendor/unilm/beats): 16 kHz, kaldi fbank inside the model, masked mean of the final features -> 768-d; layer drop 0; a clip shorter
            than 3,200 samples is zero-padded to it WITH the padding mask.

Each side of a twin is one such tower with an identity head (the embedding is the pooled backbone feature, L2-normalised: JKU's design). The code below is verbatim from
mm_towers.py except that nothing is read from the research workspace: the pretrained weights are not needed (the fine-tuned towers in models/d1/ hold every weight, loaded with
strict=True), and the BEATs architecture hyper-parameters come from models/d1/beats_cfg.json."""
import json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from . import config as C

D1_DIR = C.ROOT / 'models' / 'd1'
MINLEN = 3200                       # BEATs: shorter clips are zero-padded to this WITH the padding mask (the research evaluation's rule)


class IdentityHead(nn.Module):
    def __init__(self, in_dim=None, **kw):
        super().__init__(); self.dim = in_dim

    def forward(self, x):
        return F.normalize(x, dim=-1)


class BEATsTower(nn.Module):
    """Research BEATsTowerMM: the project's BEATsTower (BEATs -> masked mean-pool -> head) with encoder layer drop 0 and the 3,200-sample minimum length."""
    def __init__(self):
        super().__init__()
        beats_dir = str(C.VENDOR / 'unilm' / 'beats')
        if beats_dir not in sys.path:
            sys.path.insert(0, beats_dir)                  # BEATs.py imports 'backbone' and 'modules' by bare name
        from BEATs import BEATs, BEATsConfig
        self.backbone = BEATs(BEATsConfig(json.load(open(D1_DIR / 'beats_cfg.json'))))
        self.backbone.encoder.layerdrop = 0.0
        self.is_peft = False
        self.head = IdentityHead(in_dim=768)

    def _base(self):
        return self.backbone

    def _embed(self, audio_list):
        dev = next(self._base().parameters()).device
        outs = []
        for s in range(0, len(audio_list), 8):
            wav = [torch.from_numpy(np.asarray(a, dtype='float32')) for a in audio_list[s:s + 8]]
            T = max(max(w.shape[0] for w in wav), MINLEN)
            x = torch.stack([F.pad(w, (0, T - w.shape[0])) for w in wav]).to(dev)
            pad = torch.zeros(x.shape, dtype=torch.bool, device=dev)
            for i, w in enumerate(wav):
                pad[i, w.shape[0]:] = True
            feats, out_pad = self._base().extract_features(x, padding_mask=pad)[:2]
            keep = (~out_pad).unsqueeze(-1).float()
            outs.append((feats * keep).sum(1) / keep.sum(1).clamp(min=1.0))
        return torch.cat(outs, dim=0)

    def encode_audio(self, audio_list):
        return self.head(self._embed(audio_list))


class _MBlock(nn.Module):                                           # a timm 0.4.12 ViT Block (qkv bias, LayerNorm eps 1e-6, exact GELU, no dropout / drop-path) with the same parameter names
    def __init__(self, d=768, heads=12, ratio=4):
        super().__init__(); self.h = heads
        self.norm1 = nn.LayerNorm(d, eps=1e-6); self.attn = nn.Module(); self.attn.qkv = nn.Linear(d, 3 * d); self.attn.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, eps=1e-6); self.mlp = nn.Module(); self.mlp.fc1 = nn.Linear(d, ratio * d); self.mlp.fc2 = nn.Linear(ratio * d, d)

    def forward(self, x):
        B, N, C_ = x.shape
        q, k, v = self.attn.qkv(self.norm1(x)).reshape(B, N, 3, self.h, C_ // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.attn.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C_))
        return x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))


class MatpacEncoder(nn.Module):
    """Port of the MATPAC / MATPAC++ inference code (aurianworld/matpac, modeling_matpac.py 'precise' path, concat_freq=True, pull_time_dimension=True). pos_embed is a (fixed sin-cos) buffer."""
    D, NF, NT, PATCH, MEAN, STD = 768, 80, 608, 16, -7.056, 4.193

    def __init__(self):
        super().__init__()
        import torchaudio
        self.patch_embed = nn.Module(); self.patch_embed.proj = nn.Conv2d(1, self.D, self.PATCH, self.PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.D))
        self.register_buffer('pos_embed', torch.zeros(1, 1 + (self.NF // self.PATCH) * (self.NT // self.PATCH), self.D))
        self.student_encoder = nn.Module(); self.student_encoder.blocks = nn.ModuleList([_MBlock() for _ in range(12)]); self.student_encoder.norm = nn.LayerNorm(self.D, eps=1e-6)
        self.log_mel = nn.Module()
        with torch.device('cpu'):
            self.log_mel.MelSpectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, f_min=50.0, f_max=8000.0, n_mels=self.NF,
                                                                               center=False, pad_mode='reflect', norm='slaney', mel_scale='slaney')

    def _tokens(self, x):                                           # x: (B, 80, T) with T a multiple of 16 and <= 608 -> (B, T/16, 5*768) tokens of the last layer, cls token dropped
        fb, tf = self.NF // self.PATCH, x.shape[-1] // self.PATCH
        t = self.patch_embed.proj(x.unsqueeze(1)).flatten(2).transpose(1, 2)                      # (B, 5*tf, 768), frequency-major
        pos = self.pos_embed[:, 1:].reshape(1, fb, self.NT // self.PATCH, self.D)[:, :, :tf].reshape(1, fb * tf, self.D)
        t = torch.cat([(self.cls_token + self.pos_embed[:, :1]).expand(t.shape[0], -1, -1), t + pos], dim=1)
        for blk in self.student_encoder.blocks:
            t = blk(t)
        t = self.student_encoder.norm(t)[:, 1:]
        return t.reshape(t.shape[0], fb, tf, self.D).permute(0, 2, 1, 3).reshape(t.shape[0], tf, fb * self.D)

    def forward(self, wav):                                         # wav: (B, L) 16 kHz, one common length -> (B, 3840)
        with torch.autocast(device_type='cuda', enabled=False):    # the front end stays in fp32
            x = (torch.log(self.log_mel.MelSpectrogram(wav.float()) + torch.finfo(torch.float32).eps) - self.MEAN) / self.STD
        x = F.pad(x, (0, (-x.shape[-1]) % self.PATCH))
        toks = torch.cat([self._tokens(x[..., i:i + self.NT]) for i in range(0, x.shape[-1], self.NT)], dim=1)
        return toks.mean(dim=1)


class MatpacTower(nn.Module):
    """MATPAC++ encoder -> identity head. Clips are grouped by their number of mel frames, so a clip's embedding is the same alone or in a batch."""
    CHUNK = 16

    def __init__(self):
        super().__init__()
        self.backbone = MatpacEncoder(); self.is_peft = False
        self.head = IdentityHead(in_dim=3840)

    def _base(self):
        return self.backbone

    def _embed(self, audio_list):
        dev = next(self.backbone.parameters()).device
        wav = [torch.from_numpy(np.asarray(a, dtype='float32')) for a in audio_list]; wav = [F.pad(w, (0, max(0, 400 - w.shape[0]))) for w in wav]
        groups = {}
        for i, w in enumerate(wav):
            groups.setdefault((w.shape[0] - 400) // 160 + 1, []).append(i)                          # frames of a centre=False STFT; samples past the last frame never enter
        out = [None] * len(wav)
        for nf, idx in groups.items():
            L = 400 + (nf - 1) * 160
            for s in range(0, len(idx), self.CHUNK):
                ii = idx[s:s + self.CHUNK]; z = self.backbone(torch.stack([wav[i][:L] for i in ii]).to(dev))
                for j, i in enumerate(ii):
                    out[i] = z[j]
        return torch.stack(out)

    def encode_audio(self, audio_list):
        return self.head(self._embed(audio_list))


BUILDERS = {'matpac': MatpacTower, 'beats': BEATsTower}


def load_tower(kind, weights_path, device):
    """a fine-tuned tower of the D1 study: built from the architecture alone, every weight from `weights_path` (strict), frozen, eval mode, on `device`"""
    t = BUILDERS[kind]()
    t.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
    t.to(device).eval()
    for p in t.parameters():
        p.requires_grad = False
    return t
