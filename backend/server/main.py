"""
main.py — internal-only GPU compute API for the Q-SFX Next.js frontend.

Everything runs on ONE computer. The server binds to 127.0.0.1 (port 8500) so it is reachable only from this
machine; the Next.js frontend (port 3000) calls it over localhost. This process holds the models and does ONLY the
work that needs a GPU: tower embedding, the Audio Flamingo 3 speech gate, Whisper transcription and AF3 captioning.
The submissions database, the OmniRoute metadata call and the admin dashboard live in the frontend (TypeScript),
see sfxdeploy/submit.py.

Transport: search/ingest request bodies ARE the raw WAV bytes (no multipart parsing) — the browser records
audio, encodes it to WAV client-side, and the Next.js API route streams that body through unchanged.

Endpoints:
    POST /search?gallery=sfx|qdev&k=10   body: raw WAV bytes
    POST /ingest?gallery=sfx             body: raw WAV bytes, header X-Filename: original name (optional)
                                          -> {file, transcript, caption, caption_error} (synchronous; slow)
    GET  /audio/{gallery}/{path}         gallery clip playback (Range-supported), path-traversal-guarded
    GET  /health
"""
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sfxdeploy import assetcheck  # noqa: E402  (stdlib only)
assetcheck.check_or_exit()  # a clear list of missing files instead of a stack trace deep inside model loading
import uvicorn  # noqa: E402
from fastapi import FastAPI, Header, HTTPException, Query, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

import sfxdeploy  # noqa: E402,F401  (env setup before transformers)
from sfxdeploy import config as C  # noqa: E402
from sfxdeploy import submit as submit_pipeline  # noqa: E402
from sfxdeploy.d1 import D1Model  # noqa: E402
from sfxdeploy.sfx_search import SfxSearch  # noqa: E402
from sfxdeploy.qdev_search import QdevSearch  # noqa: E402

HOST = os.environ.get('SFXDEPLOY_HOST', '127.0.0.1')
PORT = int(os.environ.get('SFXDEPLOY_PORT', '8500'))
DEFAULT_K = 10
MAX_K = 200
AUDIO_ROOTS = {'sfx': C.SFX_GALLERY.resolve(), 'qdev': (C.QVIM_DEV / 'Items').resolve()}

app = FastAPI()
STATE = {}  # populated at startup: searchers


@app.on_event('startup')
def startup():
    t0 = time.time()
    print('[server] loading D1 (MATPAC++ twin + BEATs twin + soft-DTW head) ...', flush=True)
    model = D1Model()                                # shared by both pipelines — one set of weights, not two
    print(f'[server] D1 loaded on {model.device} in {time.time() - t0:.1f}s, loading SFX (D1 + AF3 speech gate + Whisper) + qvim-dev (D1) pipelines ...', flush=True)

    sfx = SfxSearch(model=model)
    qdev = QdevSearch(model)                         # qvim-dev: the same D1 system as SFX (ensemble tower + soft-DTW head + CSLS), no gate, no Whisper
    STATE['searchers'] = {'sfx': sfx, 'qdev': qdev}
    print(f'[server] both pipelines loaded in {time.time() - t0:.1f}s | sfx={len(sfx.index.files)} qdev={len(qdev.index.files)}', flush=True)

    # Warm up CUDA kernels (D1 + AF3 gate, + Whisper if the warm-up clip routes to speech; qvim-dev) so the first real request isn't the slow cold call.
    warm_path = str(ROOT / 'data' / 'sfx' / 'gallery' / sfx.index.files[0])
    t0 = time.time()
    sfx.search(warm_path, k=1)
    qdev.search(str(ROOT / 'data' / 'qvim-dev' / 'Items' / qdev.index.files[0]), k=1)
    print(f'[server] warmup done in {(time.time() - t0) * 1000:.0f}ms -- ready', flush=True)


@app.get('/health')
def health():
    ready = 'searchers' in STATE
    return {
        'status': 'ok' if ready else 'loading',
        'root': str(ROOT),   # lets scripts/run_server.sh tell THIS backend from another instance answering on the same port
        'models_loaded': ready,
        'galleries': {k: len(v.index.files) for k, v in STATE['searchers'].items()} if ready else None,
        'sfx_system': 'D1 (MATPAC++ twin + BEATs twin, zero-shot) + soft-DTW head + CSLS + AF3 speech gate -> Whisper RRF on speech' if ready else None,
        'qdev_system': 'D1 (MATPAC++ twin + BEATs twin) + soft-DTW head + CSLS' if ready else None,
    }


@app.post('/search')
async def search(request: Request,
                  gallery: str = Query(...),
                  k: int = Query(DEFAULT_K)):
    if 'searchers' not in STATE:
        raise HTTPException(503, 'models still loading')
    if gallery not in STATE['searchers']:
        raise HTTPException(400, f'unknown gallery {gallery!r}, expected sfx|qdev')
    if not (1 <= k <= MAX_K):
        raise HTTPException(400, f'k must be in [1,{MAX_K}], got {k}')

    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(400, 'empty request body -- expected raw WAV bytes')

    t0 = time.time()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        out = await run_in_threadpool(STATE['searchers'][gallery].search, tmp_path, k=k)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if out is None:
        raise HTTPException(422, 'could not read query audio (too short, silent, or unsupported)')

    return JSONResponse({
        'gallery': gallery,
        'gallery_size': len(STATE['searchers'][gallery].index.files),   # live count -- grows as items are ingested
        'results': out['results'],
        'transcript': out.get('transcript'),     # gallery=sfx, speech route only (None when the gate says no speech)
        'route': out.get('route'),               # gallery=sfx: 'speech' (D1 + Whisper RRF) | 'non-speech' (D1 alone)
        'speech_logit': out.get('speech_logit'),  # gallery=sfx: AF3 logit(Yes) - logit(No); speech iff > 0
        'query_ms': round((time.time() - t0) * 1000, 1),
    })


@app.post('/ingest')
async def ingest(request: Request, gallery: str = Query('sfx'), x_filename: str | None = Header(default=None)):
    """Embed + transcribe + caption ONE new item and return synchronously. Slow (embedding is fast, but a
    cold Flamingo load + captioning can take tens of seconds) -- SFX-web-app is expected to call this from a
    background job on its own side and poll its own submission record, not block a user-facing request on it.
    Only gallery=sfx: qvim-dev is a fixed 20-class academic benchmark, not a general gallery."""
    if gallery != 'sfx':
        raise HTTPException(400, "only gallery=sfx accepts ingestion (qvim-dev is a fixed benchmark set)")
    if 'searchers' not in STATE:
        raise HTTPException(503, 'models still loading')

    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(400, 'empty request body -- expected raw WAV bytes')

    submit_pipeline.SUBMITTED_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = submit_pipeline.SUBMITTED_DIR / f'{uuid.uuid4().hex}.wav'
    stored_path.write_bytes(wav_bytes)

    try:
        result = await run_in_threadpool(submit_pipeline.ingest_item, STATE['searchers']['sfx'], stored_path)
    except Exception as e:
        raise HTTPException(422, f'ingestion failed: {e}') from e

    return JSONResponse(result)


@app.get('/audio/{gallery}/{rel_path:path}')
def audio(gallery: str, rel_path: str):
    # Serves gallery clips for playback to the frontend. FileResponse handles Range requests (seeking) for free.
    base = AUDIO_ROOTS.get(gallery)
    if base is None:
        raise HTTPException(404, f'unknown gallery {gallery!r}, expected sfx|qdev')
    target = (base / rel_path).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(400, 'invalid path')
    if not target.is_file():
        raise HTTPException(404, 'not found')
    return FileResponse(target, media_type='audio/wav')


if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT, log_level='info')
