"""NVIDIA Audio Flamingo 3 (transformers-native `audioflamingo3`, NVIDIA OneWay Noncommercial licence: research use), Qwen2-7B language model + Whisper-style audio encoder.
LLM weights 4-bit nf4; the audio encoder and projector stay bf16 (they are small and quantising them would degrade the audio front-end, not the language model).
Audio goes through the chat template as an in-memory array ({"type": "audio", "audio": ndarray}, 16 kHz)."""
import torch
from .. import config as C
from .base import ALM, load_wave, bnb_4bit

class AudioFlamingo3(ALM):
    name = 'audio-flamingo-3'
    def __init__(self, path=C.ALM_DIR / 'af3', device_map=None, quantize=True):
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(path))
        kw = dict(quantization_config=bnb_4bit(skip_modules=['audio_tower', 'multi_modal_projector'])) if quantize else dict(dtype=torch.bfloat16)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(str(path), device_map=device_map or {'': 0}, low_cpu_mem_usage=True, **kw).eval()

    def _inputs(self, wave, prompt):
        conv = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'audio', 'audio': wave}]}]
        inp = self.processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True)
        assert 'input_features' in inp, f'the audio did not reach the model; processor returned {list(inp.keys())}'
        return inp.to(self.model.device)

    @torch.no_grad()
    def describe(self, audio, prompt, max_new_tokens=128):
        inp = self._inputs(load_wave(audio), prompt)
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(gen[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
