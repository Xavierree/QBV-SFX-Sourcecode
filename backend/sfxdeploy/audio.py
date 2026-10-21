"""Tower audio front-end - verbatim from the SFX zero-shot evaluation notebook (identical to the VimSketch pipeline's `load_audio`)."""
import os, warnings, concurrent.futures as cf
import numpy as np, librosa
warnings.filterwarnings('ignore', module='librosa')      # .m4a queries fall back to audioread, which librosa flags as deprecated on every call
from .config import SPEECH_SR, MAX_DUR

def load_audio(path, cap=False, unlimited=False):
    """mono 16 kHz, silence-trim (15 dB), RMS-normalise to 0.1. None if unreadable, empty, > MAX_DUR seconds, or < 1024 samples after trimming.
    cap=True keeps the first MAX_DUR seconds of a longer clip instead of rejecting it (the qvim-dev benchmark protocol of SFX-fusion/sfx_adapt/common.py).
    unlimited=True accepts any length (the research frame cache of the qvim-dev head similarities: scripts/build_qdev_d1_index.py)."""
    try:
        y, _ = librosa.load(path, sr=SPEECH_SR, mono=True)
    except Exception:
        return None
    if cap: y = y[:int(MAX_DUR * SPEECH_SR)]
    if y.size == 0 or (not unlimited and y.size / SPEECH_SR > MAX_DUR): return None
    y, _ = librosa.effects.trim(y, top_db=15)
    if y.size < 1024: return None
    rms = np.sqrt(np.mean(y ** 2))
    return (y * (0.1 / (rms + 1e-6))).astype('float32')

def load_many(paths, desc='decode', workers=min(8, os.cpu_count() or 4), progress=True, cap=False, unlimited=False):
    with cf.ThreadPoolExecutor(workers) as ex:
        it = ex.map(lambda p: load_audio(p, cap=cap, unlimited=unlimited), paths)
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(paths), desc=desc, leave=False)
        return list(it)
