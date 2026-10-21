#!/usr/bin/env python3
"""Verification gate for the D1 SFX pipeline (D1 system + AF3 speech gate + Whisper RRF), run through the SERVABLE SfxSearch path one query at a time, on the BASE gallery (overlays
pointed at empty scratch files, so the live data is never touched and the numbers are comparable to the research export).

D1 is ZERO-SHOT (nothing in it was trained on the 84 SFX queries), so this checks FIDELITY to the research export. The accuracy estimate is the thesis's per-fold research number: the served CSLS pool leaves
all 84 queries' own targets out (right for new queries), so the numbers printed here for these 84 queries are higher than the per-fold ones and are plumbing, not accuracy:
  (a) system rows reproduce the research matrices (models/d1/reference_tables.npz) put through the same rule: target ranks identical
  (b) the live AF3 gate reproduces the logged gate logits of the pre-registered study (decisions identical)
  (c) search() end to end reproduces the offline gated ranks computed from the export + the offline transcripts
      (queries whose live Whisper transcript differs from the offline one are listed and excluded from (c))
  (d) ingest path: a base clip re-ingested under a new name gets the same tower rows and CSLS r_T as the original (only if it is no query's target), is ranked level with it,
      and survives a reload from the overlay file; the captioner (shared AF3) still fits in memory
  (e) qvim-dev: the served path over the 979 queries reproduces the research MRR of the system
usage: .venv/bin/python scripts/verify_d1_search.py SCRATCH_DIR"""
import sys, json, time, gc
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from scipy.stats import rankdata
import sfxdeploy  # noqa: F401
from sfxdeploy import config as C
from sfxdeploy.audio import load_audio
from sfxdeploy.benchmark import load_pairs
from sfxdeploy.d1 import D1_DIR, fuse, csls_hub
from sfxdeploy.sfx_search import SfxSearch, _norm, _ok
from sfxdeploy.qdev_search import QdevSearch
from sfxdeploy.qdev_bench import load_qdev, metrics

SCR = Path(sys.argv[1]); SCR.mkdir(parents=True, exist_ok=True)
for f in ('d1_overlay.npz', 'tr_overlay.json', 'no_legacy_overlay.npz'):
    (SCR / f).unlink(missing_ok=True)
OV = dict(d1_overlay_path=SCR / 'd1_overlay.npz', legacy_overlay_path=SCR / 'no_legacy_overlay.npz', transcripts_overlay_path=SCR / 'tr_overlay.json')
ref = np.load(D1_DIR / 'reference_tables.npz', allow_pickle=True); RG = np.load(D1_DIR / 'reference_speech_gate.npz'); cfg = json.load(open(D1_DIR / 'config.json'))
qt_off = [str(x) for x in ref['query_transcripts']]          # offline Whisper transcripts of the 84 queries (models/d1/reference_tables.npz)
pairs, _, _ = load_pairs(); assert len(pairs) == 84 and [p['mimic_file'] for p in pairs] == [str(f) for f in ref['query_files']]

t0 = time.time(); S = SfxSearch(**OV); print(f'SfxSearch (base gallery) loaded in {time.time() - t0:.0f}s', flush=True)
files = S.index.files; N = len(files); assert files == [str(f) for f in ref['gallery_files']], 'base gallery differs from the export'
pos = {f: i for i, f in enumerate(files)}; targets = [np.array([pos[t] for t in p['target_files'] if t in pos]) for p in pairs]
excl = np.zeros((84, N), bool)
for q, t in enumerate(targets):
    excl[q, t] = True
_, Fzr = fuse(ref['S_a'], ref['S_b'], ref['S_h']); Fcr = Fzr - 0.5 * csls_hub(Fzr, excl)[None, :]        # the research matrices through the served rule
Sx_off = np.asarray((S.vec.transform([_norm(x) if _ok(_norm(x)) else '' for x in qt_off]) @ S.Xg.T).todense())


def rank_avg(score, t):
    m = score[t].max(); return 1 + (score > m).sum() + 0.5 * ((score == m).sum() - 1)


def rrf(row, sx):
    s = 1.0 / (60 + rankdata(-row, method='average')); m = sx > 0
    if m.any(): s[m] += 1.0 / (60 + rankdata(-sx[m], method='average'))
    return s


rows, logits, live_ranks, exp_ranks, tr_diff = [], [], [], [], []
t0 = time.time()
for q, d in enumerate(pairs):
    wave = load_audio(d['mimic_path'])
    row, _ = S.d1.score(S.model.query(wave), S.d1.snapshot()); rows.append(row)
    out = S.search(d['mimic_path'], k=N); logits.append(out['speech_logit'])
    sc = np.full(N, -np.inf); sc[[pos[r['file']] for r in out['results']]] = [r['score'] for r in out['results']]
    live_ranks.append(rank_avg(sc, targets[q]))
    if out['route'] == 'speech':
        if (out['transcript'] or '').strip() != (qt_off[q] or '').strip(): tr_diff.append(q)
        exp_ranks.append(rank_avg(rrf(Fcr[q].astype('float64'), Sx_off[q]), targets[q]))
    else:
        exp_ranks.append(rank_avg(Fcr[q].astype('float64'), targets[q]))
    if (q + 1) % 21 == 0: print(f'  {q + 1}/84 ({time.time() - t0:.0f}s, {(time.time() - t0) / (q + 1):.2f}s/query incl. the duplicate scoring)', flush=True)
rows, logits, live_ranks, exp_ranks = np.stack(rows), np.array(logits), np.array(live_ranks), np.array(exp_ranks)
ok = True

r_live = np.array([rank_avg(rows[q].astype('float64'), targets[q]) for q in range(84)]); r_ref = np.array([rank_avg(Fcr[q].astype('float64'), targets[q]) for q in range(84)])
a_ok = bool((r_live == r_ref).all()); ok &= a_ok
print(f'(a) system rows vs the research matrices through the served rule: max |diff| {np.abs(rows - Fcr).max():.2e}, target ranks identical {int((r_live == r_ref).sum())}/84 -> {"OK" if a_ok else "FAIL"}')
dec_live, dec_ref = logits > 0, RG['c_sfx'] > 0
b_ok = bool((dec_live == dec_ref).all()); ok &= b_ok
print(f'(b) AF3 gate vs logged study: max |logit diff| {np.abs(logits - RG["c_sfx"]).max():.3f}, decisions identical {int((dec_live == dec_ref).sum())}/84 '
      f'(speech {int(dec_live.sum())}) -> {"OK" if b_ok else "FAIL"}' + ('' if b_ok else f' differing: {np.where(dec_live != dec_ref)[0].tolist()}'))
keep = np.array([q not in tr_diff for q in range(84)])
c_ok = bool((live_ranks[keep] == exp_ranks[keep]).all()); ok &= c_ok
print(f'(c) search() vs offline gated ranks: identical {int((live_ranks[keep] == exp_ranks[keep]).sum())}/{int(keep.sum())} '
      f'({len(tr_diff)} speech-routed queries with a different live transcript excluded: {tr_diff}) -> {"OK" if c_ok else "FAIL"}')
peak_search = torch.cuda.max_memory_allocated() / 2**30
print(f'    peak GPU memory allocated during the 84 searches: {peak_search:.2f} GiB')
rn = cfg['zero_shot_research_numbers']['sfx_MRR']
print(f'    plumbing only (NOT accuracy: the served r_T leaves each of these queries own targets out): complete pipeline MRR {np.mean(1 / live_ranks):.4f}, system alone {np.mean(1 / r_live):.4f}; '
      f'the accuracy estimates are the research per-fold numbers {rn["+AF3 gate (complete pipeline)"]:.4f} / {rn["head+CSLS (full)"]:.4f} (config.json)')

# (d) ingest path
tgt_any = set(int(t) for ts in targets for t in ts)
g = next(i for i in range(N) if i not in tgt_any and files[i].lower().endswith('.wav'))
dup = 'gallery_submitted/__verify_duplicate__.wav'
S.register_new_item(dup, C.SFX_GALLERY / files[g])
fl, G_A, G_B, r_T, bank = S.d1.snapshot()
d1_ = float(np.abs(G_A[-1] - G_A[g]).max()) + float(np.abs(G_B[-1] - G_B[g]).max()); d2 = abs(float(r_T[-1]) - float(r_T[g]))
out = S.search(pairs[0]['mimic_path'], k=N + 1); sc = {r['file']: r['system_score'] for r in out['results']}
d3 = abs(sc[dup] - sc[files[g]])
del S; gc.collect(); torch.cuda.empty_cache()
S2 = SfxSearch(**OV); fl2, A2, B2, r2, b2 = S2.d1.snapshot()
d4 = float(np.abs(A2[-1] - G_A[-1]).max()) + float(np.abs(B2[-1] - G_B[-1]).max()) + abs(float(r2[-1]) - float(r_T[-1]))
from sfxdeploy import submit as submit_pipeline       # the /ingest captioner shares the gate's AF3: check it still fits next to everything
torch.cuda.reset_peak_memory_stats()
cap = submit_pipeline._get_flamingo().describe(str(C.SFX_GALLERY / files[g]), submit_pipeline.CAPTION_PROMPT, max_new_tokens=96)
peak_caption = torch.cuda.max_memory_allocated() / 2**30
print(f'    caption test: {cap!r} | peak GPU memory allocated during captioning {peak_caption:.2f} GiB '
      f'of {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}')
d_ok = fl2 == fl and fl2[-1] == dup and d1_ < 1e-5 and d2 < 1e-4 and d3 < 1e-4 and d4 < 1e-6; ok &= d_ok
print(f'(d) ingest of base clip {files[g]!r} as a duplicate: tower rows |diff| {d1_:.2e}, r_T |diff| {d2:.2e}, system score |diff| {d3:.2e}, '
      f'reload from overlay |diff| {d4:.2e}, file lists aligned {fl2 == fl} -> {"OK" if d_ok else "FAIL"}')

# (e) qvim-dev through the served path
dq = load_qdev(); Q = QdevSearch(S2.model)
qrows = []
for pth in dq['query_paths']:
    w = load_audio(pth, cap=True); qrows.append(Q.score(S2.model.query(w))[0])
m = metrics(np.stack(qrows), dq); rq = cfg['zero_shot_research_numbers']['qvim_dev_MRR']['tower+head+CSLS']; rh = cfg['zero_shot_research_numbers']['qvim_dev_HardNeg']['tower+head+CSLS']
e_ok = abs(m['MRR'] - rq) < 1e-3 and abs(m['HardNeg'] - rh) < 1e-3; ok &= e_ok
print(f'(e) qvim-dev served path (979 queries): MRR {m["MRR"]:.4f} (research {rq:.4f}), HardNeg {m["HardNeg"]:.4f} (research {rh:.4f}) -> {"OK" if e_ok else "FAIL"}')
report = {'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'backend_tree': str(ROOT), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
          'a_system_rows_vs_research': {'ok': a_ok, 'max_abs_diff': float(np.abs(rows - Fcr).max()), 'target_ranks_identical': int((r_live == r_ref).sum()), 'queries': 84},
          'b_af3_gate_vs_logged_study': {'ok': b_ok, 'max_abs_logit_diff': float(np.abs(logits - RG['c_sfx']).max()), 'decisions_identical': int((dec_live == dec_ref).sum()), 'speech_queries': int(dec_live.sum()), 'queries': 84},
          'c_search_end_to_end_vs_offline_gated_ranks': {'ok': c_ok, 'identical': int((live_ranks[keep] == exp_ranks[keep]).sum()), 'compared': int(keep.sum()), 'excluded_different_live_transcript': [int(x) for x in tr_diff]},
          'd_ingest_duplicate': {'ok': d_ok, 'tower_rows_max_abs_diff': d1_, 'r_T_abs_diff': d2, 'system_score_abs_diff': d3, 'reload_from_overlay_abs_diff': d4, 'file_lists_aligned': bool(fl2 == fl), 'caption_example': cap},
          'e_qvim_dev_served_path': {'ok': e_ok, 'MRR': m['MRR'], 'HardNeg': m['HardNeg'], 'research_MRR': rq, 'research_HardNeg': rh, 'queries': len(dq['query_paths'])},
          'peak_gpu_gib_allocated': {'during_84_searches': peak_search, 'during_captioning': peak_caption},
          'plumbing_not_accuracy_84_queries': {'gated_MRR': float(np.mean(1 / live_ranks)), 'system_alone_MRR': float(np.mean(1 / r_live))},
          'gate': 'PASS' if ok else 'FAIL'}
(SCR / 'verify_d1_report.json').write_text(json.dumps(report, indent=1)); print(f'report: {SCR / "verify_d1_report.json"}')
print('\nGATE: ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
