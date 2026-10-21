# The served system since 2026-10-15: D1

**`/search?gallery=sfx`:** D1 → **Audio Flamingo 3 speech gate** → **Whisper route only for queries with spoken words**.
**`/search?gallery=qdev`:** D1 alone (no gate, no Whisper).
(Until 2026-10-15 the system under the gate was the Phase 40 dual tower of the earlier thesis draft; the gate and the Whisper route did not change.)

## What D1 is
Two fully fine-tuned **twins**, each with an imitation tower and a reference tower of the same pretrained model, trained on VimSketch with the JKU recipe (seed 123, epoch chosen on held-out VimSketch *concepts*; never qvim-dev, never SFX):

| twin | pretrained model | embedding | parameters (both towers) |
|---|---|---|---|
| M1 | MATPAC++ general-audio encoder (self-supervised AudioSet ViT-B) | 3,840-d | 170.8 M |
| S3 | BEATs iter3+ AS2M | 768-d | 180.7 M |

plus the shared soft-DTW **head** (one frozen HuBERT-base, layer 9, and two FrameNets; the "clean" head seed 42, trained on VimSketch only) and **CSLS**.

```
T(q, g)     = z( z(cos_MATPAC(q, g)) + z(cos_BEATs(q, g)) )           the ensemble tower, z per query row over the gallery
score(q, g) = z(T) + z(softDTW(q, g)) - 0.5 * r_T(g)                  r_T(g) = mean of g's top-10 fused scores over the reference queries
```
**Zero-shot on SFX:** nothing in D1 was trained on or adapted to any SFX recording. (Adapting D1 to the 84 SFX queries was tested and does not help it: gated 0.5310 vs 0.5221, not resolved. It is therefore not applied.)
Reference queries for r_T: the 84 SFX benchmark queries (each query's own targets left out) for the SFX gallery, the 979 qvim-dev queries for the qvim-dev gallery.

## Research numbers (thesis, Bab 4: 84 SFX queries × 9,708 clips; qvim-dev 979 × 121; zero-shot; per-fold CSLS pool)
| stage | SFX MRR | qvim-dev MRR |
|---|---|---|
| tower alone (ensemble) | 0.3302 | 0.2829 |
| tower + head | 0.4336 | 0.2777 |
| tower + CSLS | 0.3666 | 0.2950 |
| **tower + head + CSLS** (what the system returns before the gate) | **0.5007** | **0.2884** |
| + Whisper always on | 0.4123 | — |
| **+ AF3 speech gate (complete pipeline)** | **0.5221** | — |

SFX is the *development* set (D1 was chosen on it), n = 84, 95% CI about ±0.04: these are development-set estimates, not a test result. A sealed set of new recordings is the open item of the thesis.
**The demo's own numbers on the 84 benchmark queries are higher** (the served r_T leaves those queries' own targets out: `build_d1_index.py` prints 0.5464 head + CSLS, `verify_d1_search.py` 0.5610 gated) — that is plumbing, not accuracy; the research numbers above are the accuracy estimate.
Whisper is a lexical shortcut for the speech half of the gallery (it matches words only); on queries that are sounds the tower decides.

## Code and data
| What | Where |
|---|---|
| system (towers + head, gallery tables, ingest, scoring) | `sfxdeploy/d1.py` |
| the two tower families (MATPAC++ port, BEATs wrapper), strict loaders | `sfxdeploy/d1_towers.py`, `vendor/unilm/beats/` (BEATs, MIT) |
| soft-DTW head utilities (frame bank, soft-DTW matrix, z-score) | `sfxdeploy/softdtw_head.py` |
| request flow (gate, Whisper RRF, live ingest) | `sfxdeploy/sfx_search.py`, `sfxdeploy/qdev_search.py`, `sfxdeploy/speech_gate.py` |
| weights + provenance + checksums (enforced at load) | `models/d1/` (`config.json`, `MANIFEST.json`) |
| indexes | `data/indexes/sfx_gallery_d1.npz`, `qdev_d1.npz` (+ `.meta.json`); ingested clips: `sfx_gallery_d1_submitted.npz` |
| weights, provenance, checksums | `models/d1/` are copies of the research runs `M1_matpac_twin` and `S3_beats_twin` (seed 123, best held-out epoch) and of the 'clean' head seed 42 (see the thesis, Bab 3 and 4) |
| rebuild the indexes (each refuses to write unless it reproduces the research export) | `scripts/build_d1_index.py` (≈ 3 min), `scripts/build_qdev_d1_index.py` (≈ 3 min) |
| **fidelity gate after any model or index change** | `.venv/bin/python scripts/verify_d1_search.py <scratch dir>` → must print `GATE: PASS` |

## What the gate checked (2026-10-15)
(a) served system rows reproduce the research matrices put through the same rule: target ranks identical 84/84; (b) the live AF3 gate: decisions identical 84/84 (52 speech); (c) `search()` end to end vs the offline gated ranks: identical 84/84; (d) ingest: a base clip re-ingested under a new name gets the same tower rows (|diff| 0) and r_T (2e-7), and survives a reload; (e) qvim-dev served path over 979 queries: MRR 0.2884, HardNeg 0.5196 = research.
Two known, documented deviations from the research matrices: **one gallery clip** (`liam-h-54272.wav`, 4.4 s) has a BEATs embedding that is numerically sensitive (its cosines move by 1e-3 between the research run, this GPU and a CPU; ranks and MRR unaffected); on **qvim-dev** the head frames of one item (a 64 s rain recording) are computed uncapped, as in the research frame cache, while the towers see it capped at 30 s.
**TF32:** unlike the Phase 40 EfficientAT tower (which needed exact fp32), D1 reproduces the research matrices best with cuDNN TF32 on (PyTorch's default; `sfxdeploy/d1.py` documents the measurement). The AF3 gate also runs with TF32 on.

## Resources (RTX 3080 Ti 12 GB, measured 2026-10-15)
- Start-up 13.5 s with warm disk caches (+ 2.3 s warm-up query); clips ingested earlier without D1 tables are encoded from their audio at the next start (a few seconds).
- GPU memory: 11,053 MiB of 12,288 resident after the measurement run (nvidia-smi; 9.7 GB with Phase 40), peak allocated 9.9 GiB over searches and 10.35 GiB while captioning (9.3 GiB with Phase 40).
- Latency over HTTP, server side (84 benchmark imitations, 98 qvim-dev queries): SFX median 0.59 s (0.48 s on the non-speech route, 0.61 s on the speech route, p90 0.69 s), qvim-dev median 0.16 s; one user, not concurrent load.
- D1 = four networks, 351.5 M parameters counting the fixed buffers (JKU: 9.8 M).

## Ingest (`POST /ingest`)
A new clip gets both reference-tower embeddings, its head frames, its own CSLS hub term (scored against the 84 reference queries with their base-gallery row statistics), a Whisper transcript and an AF3 caption; all of it is appended to the live gallery and persisted to the overlay files. The overlay's file list is the registry of ingested clips; a clip without D1 tables is re-encoded from its audio at start-up.

Research record: the thesis (Bab 3 and 4) and its experiment folder (`FEATURE_MATRIX.md`, `THESIS_CLAIMS_LEDGER.md`).
