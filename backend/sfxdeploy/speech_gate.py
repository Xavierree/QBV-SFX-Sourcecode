"""Speech gate for SFX search (Phase 43, pre-registered): does the query contain spoken words?

Audio Flamingo 3 is asked one Yes/No question on the raw query audio; speech iff logit(Yes) - logit(No) > 0 (the
model's own answer, no tuned threshold). Prompt, input and scoring are verbatim from
SFX-deployment/experiments/asr/speech_gate.py (method C). Why it exists: the Whisper transcript route lifts queries
that contain words (SFX phrase MRR 0.28 -> 0.53) but drags down non-speech queries (sound 0.18 -> 0.12); routing
by this gate gave 0.361 vs 0.308 for Whisper-always (3 tower seeds, LOOCV, +0.054 [+0.026, +0.088], 29 queries
better / 0 worse) -- SFX-fusion/speech_gate/gate_results.json.

The AF3 instance is a process-wide singleton, shared with the /ingest captioner (sfxdeploy/submit.py)."""
import threading
import torch
from .alm.base import load_wave
from .precision import cudnn_tf32

PROMPT = 'Does this audio contain a person speaking actual words (not just a vocal imitation or a sound effect)? Answer Yes or No.'
THRESHOLD = 0.0

_af3, _af3_lock = None, threading.Lock()


def get_af3():
    global _af3
    with _af3_lock:
        if _af3 is None:
            from . import alm
            _af3 = alm.load('af3')
    return _af3


class SpeechGate:
    def __init__(self):
        self.m = get_af3()
        tok = self.m.processor.tokenizer
        self.yes, self.no = tok.encode('Yes', add_special_tokens=False)[0], tok.encode('No', add_special_tokens=False)[0]

    @torch.no_grad()
    def logit(self, path):
        inp = self.m._inputs(load_wave(path), PROMPT)
        # logits_to_keep=1: only the last position's logits (the study computed all positions and took the last). Without
        # it the 4-bit lm_head is dequantised for the whole sequence (+3 GB peak, OOM next to the rest of the pipeline);
        # scripts/verify_d1_search.py checks the logits still equal the study's for all 84 SFX queries.
        # TF32 on: the study's condition (AF3's audio tower is fp32, so the flag changes the logits; see precision.py).
        with cudnn_tf32(True):
            lg = self.m.model(**inp, logits_to_keep=1).logits[0, -1].float()
        return float(lg[self.yes] - lg[self.no])

    def __call__(self, path):
        """-> (is_speech, logit)"""
        x = self.logit(path)
        return x > THRESHOLD, x
