"""Gallery ingestion: the GPU-bound half of adding a new sound to the live SFX index. Embeds it (reference
tower, appended to the LIVE index), transcribes it (Whisper, appended to the transcript overlay), and
captions it (local ALM, Audio Flamingo 3). Returns the results synchronously -- no database, no submission
tracking, no hosted-LLM call here. That's owned by the frontend instead: the OmniRoute metadata step is a plain
API call with no GPU involvement -- only actual GPU work belongs here. This module is called from server/main.py's POST /ingest.
"""
from pathlib import Path
from . import config as C

SUBMITTED_DIR = C.SFX_GALLERY / 'gallery_submitted'  # nested INSIDE the gallery dir (so the audio-serving
# endpoint resolves it with no changes) but benchmark.gallery_listing() only os.listdir()s the top level, so
# it never appears in the frozen benchmark gallery — build_index.py/verify_backend.py keep reproducing forever.

CAPTION_PROMPT = (
    'Describe this sound in one factual sentence: what is making the sound and what it sounds like. '
    'No speculation, no flourish.'
)

def _get_flamingo():
    """The process-wide Audio Flamingo 3 singleton (~6.2 GB VRAM, 4-bit). Since 2026-10-05 it is loaded at start-up by
    the SFX speech gate (sfxdeploy/speech_gate.py) and shared here, instead of a second lazily-loaded copy."""
    from .speech_gate import get_af3
    return get_af3()


def ingest_item(sfx_searcher, stored_path: Path) -> dict:
    """Embed + transcribe + caption one new item. Raises on embedding/transcription failure (nothing was
    persisted yet in that case — see SfxSearch.register_new_item's ordering). A captioning failure is NOT
    raised: the item is already searchable by that point, so this returns normally with caption=None and
    caption_error set, rather than losing an otherwise-successful ingest over a captioning hiccup."""
    file_rel = f'gallery_submitted/{stored_path.name}'
    transcript = sfx_searcher.register_new_item(file_rel, stored_path)

    caption, caption_error = None, None
    try:
        flamingo = _get_flamingo()
        caption = flamingo.describe(str(stored_path), CAPTION_PROMPT, max_new_tokens=96)
    except Exception as e:
        caption_error = str(e)

    return {'file': file_rel, 'transcript': transcript, 'caption': caption, 'caption_error': caption_error}
