"""Per-request SFX search (2026-10-15): the D1 system + a speech gate in front of the Whisper route. (Until 2026-10-15 the system under the gate was the Phase 40 dual tower; Phase 43 introduced the gate.)

  1. D1 score over the WHOLE SFX gallery: z(T) + z(soft-DTW head) - 0.5 r_T, T = z( z(cos MATPAC++ twin) + z(cos BEATs twin) ) -- sfxdeploy/d1.py (zero-shot: no weight was trained on SFX).
  2. Speech gate (sfxdeploy/speech_gate.py): Audio Flamingo 3 Yes/No "does this audio contain spoken words?",
     speech iff logit(Yes) - logit(No) > 0.
  3a. Speech -> Whisper-transcribe the query (large-v3-turbo, first 30 s, greedy, fp16), character 3-5-gram TF-IDF
      cosine against the gallery's cached transcripts, and RRF: score = 1/(60+rank_system) + 1/(60+rank_text), the
      text term only for clips with a positive text score (the Phase 42 fusion, verbatim).
  3b. No speech -> the D1 ranking alone (Whisper never runs).
Why the gate: Whisper lifts queries with words but hurts non-speech ones when it is always on; behind the gate it no longer hurts. Whisper is a lexical shortcut for the speech half of the library
(it matches words only); on queries that are sounds the tower decides. Research numbers of this system (84 SFX queries x 9,708 clips, zero-shot, per-fold CSLS pool; thesis Bab 4):
tower 0.3302, head + CSLS 0.5007, Whisper always on 0.4123, complete pipeline (gate) 0.5221 MRR -- development-set numbers, SFX is the set D1 was chosen on.
scripts/verify_d1_search.py checks fidelity to the research export; the accuracy estimate is the research one (nothing here was trained on the 84 queries).

Clips ingested through /ingest keep their D1 tables in an overlay file (data/indexes/sfx_gallery_d1_submitted.npz) and their transcripts in the transcript overlay; the overlay's file list is the registry of
ingested clips. Clips that were ingested before D1 existed are taken from the old index overlay and encoded from their audio at the first start. Load once per process (SfxSearch()), then .search(path, k)."""
import json, pickle, re, threading
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config as C
from .audio import load_audio
from .d1 import D1Model, D1Gallery, INDEX_PATH as D1_INDEX_PATH, OVERLAY_PATH as D1_OVERLAY_PATH
from .speech_gate import SpeechGate

RRF_K = 60
_norm = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', (s or '').lower())).strip()
_ok = lambda s: len(s) > 2

TRANSCRIPTS_PATH = C.ROOT / 'data' / 'sfx' / 'metadata' / 'gallery_transcripts.json'
TRANSCRIPTS_OVERLAY_PATH = C.ROOT / 'data' / 'sfx' / 'metadata' / 'gallery_transcripts_submitted.json'
LEGACY_OVERLAY_PATH = C.ROOT / 'data' / 'indexes' / 'sfx_gallery_submitted.npz'      # before D1: the registry of ingested clips (read once, to migrate them)
VECTORIZER_CACHE = C.ROOT / 'models' / 'text' / 'sfx_gallery_tfidf.pkl'
WHISPER_DIR = C.ROOT / 'models' / 'asr' / 'whisper-large-v3-turbo'


class SfxSearch:
    def __init__(self, model=None, index_path=D1_INDEX_PATH, d1_overlay_path=D1_OVERLAY_PATH, legacy_overlay_path=LEGACY_OVERLAY_PATH, transcripts_path=TRANSCRIPTS_PATH,
                 transcripts_overlay_path=TRANSCRIPTS_OVERLAY_PATH, vectorizer_cache=VECTORIZER_CACHE, whisper_dir=WHISPER_DIR):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.model = model or D1Model()
        # registry = the ordered file names of the clips ingested so far: the D1 overlay once it exists, else the pre-D1 index overlay (migration: D1Gallery encodes those clips from their audio)
        base_files = [str(f) for f in np.load(index_path, allow_pickle=True)['files']]
        d1_overlay_path, legacy_overlay_path = Path(d1_overlay_path), Path(legacy_overlay_path)
        src = d1_overlay_path if d1_overlay_path.exists() else legacy_overlay_path
        registry = [str(f) for f in np.load(src, allow_pickle=True)['files']] if src.exists() else []
        self.d1 = D1Gallery(self.model, registry, index_path=index_path, overlay_path=d1_overlay_path)

        base_gt = json.load(open(transcripts_path))
        assert len(base_gt) == len(base_files), f'gallery_transcripts.json has {len(base_gt)} entries but the index has {len(base_files)} base files.'
        self.transcripts_overlay_path = Path(transcripts_overlay_path)
        overlay_gt = json.load(open(self.transcripts_overlay_path)) if self.transcripts_overlay_path.exists() else []
        # The D1 overlay and transcript overlay are two separate files, written in sequence by
        # register_new_item() -- NOT one atomic transaction. If a submission dies between the two writes
        # (Whisper OOM, the process getting killed, anything), they end up out of lockstep. Rather than a
        # hard assert that would refuse to start the whole server over one bad submission, self-heal:
        # an index item with no transcript yet just falls back to tower-only ranking for that one clip
        # (the same fallback every non-speech clip already gets), which is always safe and never loses
        # an already-embedded item.
        n_index_overlay = len(registry)
        if len(overlay_gt) < n_index_overlay:
            missing = n_index_overlay - len(overlay_gt)
            print(f'[sfx_search] WARNING: index overlay has {n_index_overlay} submitted item(s) but the transcript overlay '
                  f'only has {len(overlay_gt)} -- backfilling {missing} empty transcript(s) (a prior submission likely died '
                  'between the embedding and transcription steps; the affected item(s) stay searchable by tower alone).')
            overlay_gt = overlay_gt + [''] * missing
            self.transcripts_overlay_path.write_text(json.dumps(overlay_gt, ensure_ascii=False))
        elif len(overlay_gt) > n_index_overlay:
            extra = overlay_gt[n_index_overlay:]
            overlay_gt = overlay_gt[:n_index_overlay]
            print(f'[sfx_search] WARNING: transcript overlay has {len(extra)} more entries than the index overlay -- '
                  f'dropping the extras (should not happen; investigate if this recurs): {extra}')
            self.transcripts_overlay_path.write_text(json.dumps(overlay_gt, ensure_ascii=False))
        gt = base_gt + overlay_gt
        assert len(gt) == len(self.d1.files)  # guaranteed true now
        self.gn = [_norm(x) for x in gt]

        vectorizer_cache = Path(vectorizer_cache)
        vectorizer_cache.parent.mkdir(parents=True, exist_ok=True)
        if vectorizer_cache.exists():
            self.vec, Xg_base = pickle.loads(vectorizer_cache.read_bytes())
        else:
            self.vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True).fit([g for g in self.gn[:len(base_gt)] if _ok(g)])
            Xg_base = self.vec.transform([g if _ok(g) else '' for g in self.gn[:len(base_gt)]])
            vectorizer_cache.write_bytes(pickle.dumps((self.vec, Xg_base)))
        # Extend with any already-submitted items' transcripts (fixed vocabulary — .transform() works fine on
        # text outside the fit vocabulary, it just contributes no n-gram weight; no need to ever refit).
        overlay_n = [_norm(x) for x in overlay_gt]
        Xg_overlay = self.vec.transform([g if _ok(g) else '' for g in overlay_n]) if overlay_n else None
        self.Xg = Xg_base if Xg_overlay is None else sparse_vstack([Xg_base, Xg_overlay])

        self.whisper_proc = WhisperProcessor.from_pretrained(str(whisper_dir))
        self.whisper = WhisperForConditionalGeneration.from_pretrained(str(whisper_dir), dtype=torch.float16).to(self.model.device).eval()
        # FastAPI runs /search directly on the event-loop thread but BackgroundTasks (submissions) on a
        # separate thread-pool thread -- they CAN run concurrently. self.d1's tables/files and self.Xg/gn are
        # two separately-mutated structures that must stay the same length; a search reading them mid-commit
        # (tables already grown, Xg not yet, or vice versa) would shape-mismatch or silently misalign a row.
        # One lock, held only around the actual state mutation/read (never around the slow GPU embed/transcribe
        # calls), makes every commit atomic with respect to every reader.
        self._state_lock = threading.Lock()
        # the speech gate (AF3, shared with the /ingest captioner); the D1 tables (self.d1) were built above, aligned 1:1 with the transcripts
        self.gate = SpeechGate()

    @property
    def index(self):
        """the live gallery's file list (base + ingested), read fresh each time (what /health, /search and the warm-up use)"""
        return SimpleNamespace(files=self.d1.files)

    def register_new_item(self, file, source_path):
        """Add one new gallery item to the LIVE gallery: D1 tables (both reference-tower embeddings, head frames, CSLS r_T)
        and the Whisper transcript, committed under one lock and persisted to their overlay files so search
        sees it immediately and a restart doesn't lose it.
        `file` is the relative path this item will be served/played at (e.g. "submitted/<id>.wav");
        `source_path` is where the audio actually sits on disk right now (may differ from `file`'s eventual
        gallery-relative location — the caller decides that mapping)."""
        wave = load_audio(source_path)
        if wave is None:
            raise ValueError('audio front-end rejected this file (unreadable, silent, too long, or too short)')

        # Transcribe BEFORE persisting anything: if this throws (Whisper OOM, etc.), NOTHING has been
        # written yet, so there's no partial state for __init__'s self-heal above to ever need to fix.
        transcript = self.transcribe(source_path)
        d1_item = self.d1.item_tables(wave)                  # both tower embeddings + head frames + CSLS r_T (the clip alone)

        with self._state_lock:
            overlay_gt = json.load(open(self.transcripts_overlay_path)) if self.transcripts_overlay_path.exists() else []
            overlay_gt.append(transcript)
            self.transcripts_overlay_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcripts_overlay_path.write_text(json.dumps(overlay_gt, ensure_ascii=False))

            qn = _norm(transcript)
            self.gn.append(qn)
            row = self.vec.transform([qn if _ok(qn) else ''])
            self.Xg = sparse_vstack([self.Xg, row])
            self.d1.commit(file, *d1_item)
        return transcript

    def transcribe(self, path):
        import torch, librosa
        y, _ = librosa.load(path, sr=16000, mono=True)
        y = y[:30 * 16000].astype('float32')
        feats = self.whisper_proc([y], sampling_rate=16000, return_tensors='pt').input_features.to(self.model.device, torch.float16)
        with torch.no_grad():
            gen = self.whisper.generate(input_features=feats, task='transcribe', max_new_tokens=96)
        return self.whisper_proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def search(self, path, k=10):
        """-> {'results': [{'rank', 'file', 'score', 'system_score', 'tower_score'}, ...], 'transcript': str | None,
        'route': 'speech' | 'non-speech', 'speech_logit': float} or None if unreadable. 'score' is what the ranking
        used (RRF score on the speech route, the D1 fused+CSLS score otherwise); 'tower_score' is the mean cosine of the two towers."""
        wave = load_audio(path)
        if wave is None:
            return None
        q = self.model.query(wave)                         # a live query is always encoded ALONE
        is_speech, logit = self.gate(path)
        transcript = self.transcribe(path) if is_speech else None

        # Snapshot everything a concurrent register_new_item() could be mutating, under the same lock it commits
        # under, so the D1 tables, the file list and the transcript matrix are always the same length.
        with self._state_lock:
            snap = self.d1.snapshot()
            vec, Xg = self.vec, self.Xg
        files = snap[0]
        assert Xg.shape[0] == len(files), (Xg.shape, len(files))
        row, s_tower = self.d1.score(q, snap)

        if is_speech:
            score = 1.0 / (RRF_K + rankdata(-row, method='average'))
            qn = _norm(transcript)
            if _ok(qn):
                sx = np.asarray((vec.transform([qn]) @ Xg.T).todense()).ravel()
                m = sx > 0
                if m.any():
                    score[m] += 1.0 / (RRF_K + rankdata(-sx[m], method='average'))
        else:
            score = row.astype('float64')

        order = np.argsort(-score, kind='stable')[:k]
        results = [{'rank': r, 'file': files[i], 'score': float(score[i]), 'system_score': float(row[i]), 'tower_score': float(s_tower[i])}
                   for r, i in enumerate(order, 1)]
        return {'results': results, 'transcript': transcript, 'route': 'speech' if is_speech else 'non-speech', 'speech_logit': logit}
