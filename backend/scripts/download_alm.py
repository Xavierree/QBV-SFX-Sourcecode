#!/usr/bin/env python3
"""Download the two audio-language models used for the ALM-reranker experiments into models/alm/<name>/ (plain folders, no symlink cache).

  qwen2-audio   Qwen/Qwen2-Audio-7B-Instruct   Apache-2.0, 16.8 GB bf16 (5 safetensors shards)     -> transformers `qwen2_audio`
  af3           nvidia/audio-flamingo-3-hf     NVIDIA OneWay Noncommercial licence (research use only), 16.5 GB bf16 shards
                                               + the optional "think" LoRA adapter (1.4 GB). The repo also carries a second, single-file copy of the
                                               weights (`model.safetensors`, 16.5 GB, identical content) - skipped on purpose.   -> transformers `audioflamingo3`

Both are ~8B-parameter models: fp16/bf16 does NOT fit the 12 GB RTX 3080 Ti, load them 4-bit nf4 (see sfxdeploy/alm/).
The resolved git commit of every download is written to models/alm/<name>/DOWNLOAD_INFO.json so the exact weights are auditable.
usage: python scripts/download_alm.py [qwen2-audio] [af3]     (no args = both)"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME', str(ROOT / 'hf-cache'))                       # nothing may land in ~/.cache
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
from huggingface_hub import snapshot_download, HfApi
SPECS = {
    'qwen2-audio': dict(repo='Qwen/Qwen2-Audio-7B-Instruct', allow=None, ignore=['*.md', '.gitattributes']),
    'af3': dict(repo='nvidia/audio-flamingo-3-hf', allow=None, ignore=['model.safetensors', 'static/*', '.gitattributes']),
}
def main(names):
    tok = os.environ.get('HF_TOKEN') or None
    for n in names:
        s = SPECS[n]; dst = ROOT / 'models' / 'alm' / n; dst.mkdir(parents=True, exist_ok=True); t0 = time.time()
        sha = HfApi().model_info(s['repo'], token=tok).sha
        print(f'[{n}] {s["repo"]} @ {sha} -> {dst}', flush=True)
        snapshot_download(s['repo'], revision=sha, local_dir=str(dst), allow_patterns=s['allow'], ignore_patterns=s['ignore'], token=tok, max_workers=8)
        files = sorted(p for p in dst.rglob('*') if p.is_file() and '.cache' not in p.parts)
        info = {'repo': s['repo'], 'commit': sha, 'downloaded_seconds': round(time.time() - t0), 'total_bytes': sum(p.stat().st_size for p in files), 'files': {str(p.relative_to(dst)): p.stat().st_size for p in files}}
        (dst / 'DOWNLOAD_INFO.json').write_text(json.dumps(info, indent=1)); print(f'[{n}] done: {info["total_bytes"] / 1e9:.1f} GB in {info["downloaded_seconds"]}s', flush=True)
if __name__ == '__main__':
    args = sys.argv[1:] or list(SPECS); bad = [a for a in args if a not in SPECS]
    if bad: sys.exit(f'unknown model(s) {bad}; choose from {list(SPECS)}')
    main(args)
