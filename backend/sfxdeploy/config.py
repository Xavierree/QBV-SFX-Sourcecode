"""Paths and constants. Everything lives under the repository root; nothing is read from ~/.cache or from the research workspace (SFX/)."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'vendor'                       # softdtw.py, unilm/beats (BEATs, MIT)
HF_HOME = ROOT / 'hf-cache'                    # HuBERT-base (the head's frame model) lives here (HF cache layout)

# ---- data ----
SFX_GALLERY = ROOT / 'data' / 'sfx' / 'gallery'                # 9,723 files, 9,708 usable by the front-end
SFX_QUERIES = ROOT / 'data' / 'sfx' / 'queries'                # 86 imitations, 84 pair with the gallery
QVIM_DEV = ROOT / 'data' / 'qvim-dev'
INDEX_DIR = ROOT / 'data' / 'indexes'
ALM_DIR = ROOT / 'models' / 'alm'

# ---- front-end constants (identical to the training and evaluation code of the research) ----
SPEECH_SR, MAX_DUR = 16000, 30.0

def setup_env(offline=True):
    """Call before importing transformers/huggingface_hub. Keeps every HF artefact inside the repo; offline by default (set SFXDEPLOY_ONLINE=1 to allow downloads)."""
    os.environ.setdefault('HF_HOME', str(HF_HOME))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if offline and not os.environ.get('SFXDEPLOY_ONLINE'):
        os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    for p in (str(VENDOR),):                                   # softdtw.py is imported by bare name
        if p not in sys.path: sys.path.insert(0, p)
