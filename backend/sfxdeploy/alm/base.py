"""Shared bits of the audio-language-model wrappers: audio input, 4-bit loading, describe() contract, unloading.
Both ALMs are ~8B parameters: they are loaded 4-bit nf4 (bitsandbytes) to fit the 12 GB RTX 3080 Ti, ONE AT A TIME (call .unload() before loading the other)."""
import gc, time
from pathlib import Path
import numpy as np, torch
ALM_SR = 16000

def load_wave(audio, sr=ALM_SR):
    """path -> mono float32 at 16 kHz (raw: no trimming / RMS-normalisation - the ALMs see the imitation as recorded, like the earlier Qwen2-Audio probe); ndarray -> assumed already 16 kHz mono"""
    if isinstance(audio, (str, Path)):
        import librosa
        y, _ = librosa.load(str(audio), sr=sr, mono=True); return y.astype('float32')
    return np.asarray(audio, dtype='float32')

def bnb_4bit(skip_modules=None):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=skip_modules)

class ALM:
    """contract: describe(audio, prompt, max_new_tokens) -> str.  `audio` = path or 16 kHz ndarray."""
    name = 'alm'
    model = None; processor = None
    def describe(self, audio, prompt, max_new_tokens=128): raise NotImplementedError
    def vram_gb(self): return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    def unload(self):
        self.model = None; self.processor = None; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    @staticmethod
    def _timed(f):
        t = time.time(); out = f(); return out, time.time() - t
