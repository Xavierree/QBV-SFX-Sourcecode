"""Per-request qvim-dev search with the D1 system (the same weights as the SFX search; sfxdeploy/d1.py):
score(q, g) = z(T) + z(softDTW head) - 0.5 r_T(g), over the 121 benchmark items, T = z( z(cos MATPAC++) + z(cos BEATs) ).

qvim-dev protocol (scripts/build_qdev_d1_index.py): every clip is encoded ALONE; the towers see the items cut to 10 s (no padding) while the head frames stay at natural length; clips longer
than 30 s are capped, not rejected; CSLS r_T(g) = mean of g's top-10 fused scores over the 979 benchmark queries (no exclusion). All of that is fixed at index-build time -- the index is
read-only, qvim-dev takes no submissions. Research numbers of this system on qvim-dev (zero-shot): tower 0.2829, tower + head + CSLS 0.2884 MRR.

Load once per process (QdevSearch(model)), then call .search(path, k) per request."""
import json
from types import SimpleNamespace
import numpy as np
from . import config as C
from .audio import load_audio
from .d1 import fuse, tower_cos
from .softdtw_head import FrameBank, arrays_to_frames

QDEV_INDEX_PATH = C.INDEX_DIR / 'qdev_d1.npz'


class QdevSearch:
    def __init__(self, model, index_path=QDEV_INDEX_PATH):
        self.model = model
        z = np.load(index_path, allow_pickle=True)
        self.meta = json.load(open(index_path.with_suffix('.meta.json')))
        assert self.meta['model_sha12'] == model.cfg['sha12'], 'qvim-dev D1 index was built with different weights -- rebuild it'
        self.G_A, self.G_B, self.r_T = z['G_A'], z['G_B'], z['r_T']
        self.bank = FrameBank.build(arrays_to_frames(z['frames'], z['lengths']), model.device)
        self.index = SimpleNamespace(files=[str(f) for f in z['files']])         # relative to data/qvim-dev/Items/

    def score(self, q):
        """q = model.query(wave) -> (system score float32 [121], mean tower cosine float32 [121])"""
        q_a, q_b, q_f = q
        s_a, s_b = tower_cos(q_a, q_b, self.G_A, self.G_B)
        _, fz = fuse(s_a, s_b, self.bank.sims(q_f)[None])
        return (fz - 0.5 * self.r_T[None, :])[0], (0.5 * (s_a + s_b))[0]

    def search(self, path, k=10):
        """-> {'results': [{'rank', 'file', 'score', 'tower_score'}, ...]} or None if the audio is unreadable."""
        wave = load_audio(path)
        if wave is None:
            return None
        row, s_tower = self.score(self.model.query(wave))      # a live query is always encoded ALONE
        order = np.argsort(-row, kind='stable')[:k]
        return {'results': [{'rank': r, 'file': self.index.files[i], 'score': float(row[i]), 'tower_score': float(s_tower[i])} for r, i in enumerate(order, 1)]}
