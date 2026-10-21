"""ALM wrappers. `load('qwen2-audio' | 'af3')` -> object with .describe(audio, prompt, max_new_tokens). Load one at a time; .unload() frees the GPU."""
def load(name, **kw):
    if name == 'qwen2-audio':
        from .qwen2_audio import Qwen2Audio; return Qwen2Audio(**kw)
    if name in ('af3', 'audio-flamingo-3'):
        from .audio_flamingo3 import AudioFlamingo3; return AudioFlamingo3(**kw)
    raise ValueError(f'unknown ALM {name!r}: choose qwen2-audio | af3')
