"""Start-up check that every model / data file the server needs is present. Standard library only, so it also runs
before the virtual environment exists:  python3 scripts/check_assets.py"""
import sys
from . import config as C

D1 = C.ROOT / 'models' / 'd1'
HUB = C.HF_HOME / 'hub'
REQUIRED = [  # (path, what it is)
    (D1 / 'config.json', 'models/d1: D1 configuration (checksums of the weights)'),
    (D1 / 'matpac_mimic.pt', 'models/d1: MATPAC++ twin, imitation tower'),
    (D1 / 'matpac_ref.pt', 'models/d1: MATPAC++ twin, reference tower'),
    (D1 / 'beats_mimic.pt', 'models/d1: BEATs twin, imitation tower'),
    (D1 / 'beats_ref.pt', 'models/d1: BEATs twin, reference tower'),
    (D1 / 'beats_cfg.json', 'models/d1: BEATs architecture hyper-parameters'),
    (D1 / 'head_clean_s42.pt', 'models/d1: soft-DTW head (two FrameNets)'),
    (C.ROOT / 'models' / 'asr' / 'whisper-large-v3-turbo', 'Whisper large-v3-turbo folder'),
    (C.ALM_DIR / 'af3', 'Audio Flamingo 3 folder (scripts/download_alm.py)'),
    (HUB / 'models--facebook--hubert-base-ls960', 'hf-cache: facebook/hubert-base-ls960 (the head\'s frame model)'),
    (C.SFX_GALLERY, 'data/sfx/gallery: SFX clips'),
    (C.QVIM_DEV / 'Items', 'data/qvim-dev/Items: qvim-dev clips'),
    (C.INDEX_DIR / 'sfx_gallery_d1.npz', 'data/indexes: SFX index of D1 (scripts/build_d1_index.py)'),
    (C.INDEX_DIR / 'sfx_gallery_d1.meta.json', 'data/indexes: SFX index metadata'),
    (C.INDEX_DIR / 'qdev_d1.npz', 'data/indexes: qvim-dev index of D1 (scripts/build_qdev_d1_index.py)'),
    (C.INDEX_DIR / 'qdev_d1.meta.json', 'data/indexes: qvim-dev index metadata'),
    (C.ROOT / 'data' / 'sfx' / 'metadata' / 'gallery_transcripts.json', 'data/sfx/metadata: gallery transcripts'),
]


def missing():
    return [(p, what) for p, what in REQUIRED if not p.exists()]


def check_or_exit():
    miss = missing()
    if not miss:
        return
    print(f'[assets] {len(miss)} of {len(REQUIRED)} required files/folders are missing under {C.ROOT}:', file=sys.stderr)
    for p, what in miss:
        print(f'  - {p.relative_to(C.ROOT)}   ({what})', file=sys.stderr)
    print('[assets] The models, data and indexes are not part of the source code. See README.md section 4.', file=sys.stderr)
    sys.exit(1)
