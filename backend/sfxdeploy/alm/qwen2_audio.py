"""Qwen2-Audio-7B-Instruct (Apache-2.0), 4-bit nf4. Prompt construction is the earlier probe's (experiments/alm/probe_qwen.py): a chat turn with one audio placeholder + the
text prompt; the waveform is handed to the processor as `audio=` (SINGULAR - the plural `audios=` silently ran the model text-only in the first probe; see docs/06 finding 3),
and describe() asserts that `input_features` really reached the model."""
import torch
from .. import config as C
from .base import ALM, ALM_SR, load_wave, bnb_4bit

class Qwen2Audio(ALM):
    name = 'qwen2-audio-7b-instruct'
    def __init__(self, path=C.ALM_DIR / 'qwen2-audio', device_map=None):
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(path))
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(str(path), quantization_config=bnb_4bit(), device_map=device_map or {'': 0}, low_cpu_mem_usage=True).eval()

    def _inputs(self, wave, prompt):
        conv = [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': 'x'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = self.processor(text=text, audio=[wave], sampling_rate=ALM_SR, return_tensors='pt')
        assert 'input_features' in inp and inp['input_features'].abs().sum() > 0, 'the audio did not reach the model (text-only run)'
        return inp.to(self.model.device)

    @torch.no_grad()
    def describe(self, audio, prompt, max_new_tokens=128):
        inp = self._inputs(load_wave(audio), prompt)
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(gen[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
