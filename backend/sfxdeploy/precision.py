"""Per-model cuDNN TF32 setting. torch.backends.cudnn.allow_tf32 is process-wide; every served model runs inside cudnn_tf32(...), which holds one re-entrant lock while the flag is set, so a concurrent
request can never run one model under another's setting:
  - D1 (sfxdeploy/d1.py): TF32 ON, PyTorch's default and the condition under which the research matrices were computed. Measured 2026-10-15: exact fp32 is the LESS faithful setting for D1
    (max |diff| to the research matrices: BEATs 2e-6 with TF32 on vs 2e-5 off, head 1e-6 vs 7e-4).
  - AF3 speech gate (sfxdeploy/speech_gate.py): TF32 ON, the condition of the pre-registered gate study. AF3's audio tower is fp32, so TF32 changes its logits: with it on, 0/84 gate decisions differ from
    the study; with it off, 1/84 does.
  - (The retired Phase 40 EfficientAT tower, kept in the production copy for rollback, needed exact fp32: with TF32 on, cuDNN's run-time kernel choice moved its embeddings by ~1e-4 between runs.)
Models without fp32 convolutions (Whisper fp16, AF3's 4-bit language model) are unaffected by the flag and are not locked."""
import contextlib, threading
import torch

_LOCK = threading.RLock()


@contextlib.contextmanager
def cudnn_tf32(enabled):
    with _LOCK:
        old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = enabled, False
        try:
            yield
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old
