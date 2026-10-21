#!/usr/bin/env python3
"""Build the D1 SFX index (data/indexes/sfx_gallery_d1.npz + .meta.json) and verify it against the research export (models/d1/reference_tables.npz = the score matrices
the thesis numbers were computed from).

Contents: every base gallery clip (the 9,708 files of the benchmark gallery) encoded ALONE by the two reference towers (MATPAC++ 3,840-d, BEATs 768-d) and the head's reference side
(frames); the 84 SFX benchmark queries encoded by the two imitation towers and the head's query side (the CSLS reference set); their score rows' per-row statistics (to standardise a
later-ingested clip against them); and the CSLS hub term r_T(g) = mean of g's top-10 fused scores over the 84 queries, each query's own targets left out.

Refuses to write the index unless the served code reproduces the export: identical gallery / query order, the tower and head matrices equal to the research matrices (max |diff| reported),
and every query's target rank under the served fused + CSLS rule identical to the rank computed from the research matrices with the same rule.
It also prints the head+CSLS MRR of the served rule next to the thesis's per-fold number (0.5007). That is a PLUMBING number, not an accuracy: the served r_T leaves all 84 queries' own targets out (the right rule for new queries),
so on these 84 queries it is higher than the per-fold evaluation (which leaves only the other 83 out). The accuracy estimate is the thesis's per-fold number.
usage: .venv/bin/python scripts/build_d1_index.py"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import sfxdeploy  # noqa: F401
from sfxdeploy import config as C
from sfxdeploy.audio import load_many
from sfxdeploy.benchmark import load_pairs, gallery_listing
from sfxdeploy.softdtw_head import sdtw_matrix, cos_np, zrow, frames_to_arrays
from sfxdeploy.d1 import D1Model, INDEX_PATH, D1_DIR, fuse, csls_hub

t0 = time.time()
ref = np.load(D1_DIR / 'reference_tables.npz', allow_pickle=True)
listing = gallery_listing(C.SFX_GALLERY)                                    # the benchmark gallery, top level only (ingested clips live in gallery_submitted/)
all_waves = load_many([str(C.SFX_GALLERY / f) for f in listing], 'decode gallery')
files = [f for f, w in zip(listing, all_waves) if w is not None]            # the clips the audio front end accepts (the 15 it rejects are not part of the benchmark gallery)
g_waves = [w for w in all_waves if w is not None]; dropped = [f for f, w in zip(listing, all_waves) if w is None]; del all_waves
assert files == [str(f) for f in ref['gallery_files']], 'gallery differs from the research export'
pairs, _, _ = load_pairs()
assert [p['mimic_file'] for p in pairs] == [str(f) for f in ref['query_files']], 'query order differs from the research export'
pos = {f: i for i, f in enumerate(files)}
targets = [np.array([pos[t] for t in p['target_files'] if t in pos]) for p in pairs]
excl = np.zeros((len(pairs), len(files)), bool)
for q, t in enumerate(targets):
    excl[q, t] = True

model = D1Model()
G_A = np.zeros((len(files), 3840), 'float32'); G_B = np.zeros((len(files), 768), 'float32'); frames = []
for i, w in enumerate(g_waves):
    G_A[i], G_B[i], f = model.item(w); frames.append(f)
    if (i + 1) % 1000 == 0:
        print(f'  gallery {i + 1}/{len(files)} ({time.time() - t0:.0f}s)', flush=True)
print(f'gallery encoded: {len(files)} clips, {sum(f.shape[0] for f in frames)} head frames ({time.time() - t0:.0f}s)', flush=True)
durations = [len(w) / 16000 for w in g_waves]; del g_waves

q_waves = load_many([p['mimic_path'] for p in pairs], 'decode queries'); assert all(w is not None for w in q_waves)
Q = [model.query(w) for w in q_waves]
Q_A = np.stack([q[0] for q in Q]); Q_B = np.stack([q[1] for q in Q]); Q_F = [q[2] for q in Q]
S_a, S_b = cos_np(Q_A, G_A), cos_np(Q_B, G_B); S_h = sdtw_matrix(Q_F, frames, model.device)
T, Fz = fuse(S_a, S_b, S_h); r_T = csls_hub(Fz, excl); Fc = Fz - 0.5 * r_T[None, :]
Tr, Fzr = fuse(ref['S_a'], ref['S_b'], ref['S_h']); r_Tr = csls_hub(Fzr, excl); Fcr = Fzr - 0.5 * r_Tr[None, :]      # the same rule on the research matrices


def tranks(S):
    return np.array([1 + int((S[q] > S[q][t].max()).sum()) for q, t in enumerate(targets)])


# Fidelity check. A gallery column is an OUTLIER when any of the three base tables differs from the research matrix by more than 1e-4 there. One clip of the gallery (liam-h-54272.wav, 4.4 s) is
# numerically sensitive in BEATs: its embedding moves by ~1e-3 in cosine between the research run, this GPU run and a CPU run alike (TF32 on or off makes no difference), so it is reported, not hidden.
# Everything else must match to < 1e-4; the derived tables are compared on the other columns; target ranks must be identical.
base = {'S_a (MATPAC++ tower)': (S_a, ref['S_a']), 'S_b (BEATs tower)': (S_b, ref['S_b']), 'S_h (head)': (S_h, ref['S_h'])}
col = np.max([np.nanmax(np.abs(a - b), 0) for a, b in base.values()], 0)
outliers = [int(i) for i in np.where(col > 1e-4)[0]]; keep = np.ones(len(files), bool); keep[outliers] = False
chk = {name: float(np.nanmax(np.abs(a - b)[:, keep])) for name, (a, b) in base.items()}
chk.update({name: float(np.nanmax(np.abs(a - b)[..., keep])) for name, a, b in (('tower z', T, Tr), ('fused', Fz, Fzr), ('r_T', r_T, r_Tr), ('fused_csls', Fc, Fcr))})
chk['outlier_columns'] = {files[i]: {'seconds': round(durations[i], 2), 'max_abs_diff_S_a': float(np.abs(S_a - ref['S_a'])[:, i].max()), 'max_abs_diff_S_b': float(np.abs(S_b - ref['S_b'])[:, i].max()),
                                     'max_abs_diff_S_h': float(np.nanmax(np.abs(S_h - ref['S_h'])[:, i]))} for i in outliers}
r_new, r_ref = tranks(Fc), tranks(Fcr)
chk['target_rank_mismatches'] = int((r_new != r_ref).sum())
mrr = float(np.mean(1 / r_new)); mrr_tower = float(np.mean(1 / tranks(T)))
print('verification vs research export (max |diff|, outlier columns excluded): ' + ', '.join(f'{k} {v:.2e}' for k, v in chk.items() if isinstance(v, float)) + f' | target ranks differing: {chk["target_rank_mismatches"]}/84', flush=True)
print(f'outlier columns ({len(outliers)}): {chk["outlier_columns"]}', flush=True)
print(f'served rule on the 84 queries (plumbing, NOT accuracy: r_T leaves each query own targets out): tower MRR {mrr_tower:.4f} (thesis 0.3302), head + CSLS MRR {mrr:.4f} (the accuracy estimate is the thesis per-fold 0.5007)', flush=True)
assert chk['target_rank_mismatches'] == 0 and len(outliers) <= 2 and max(v for v in chk.values() if isinstance(v, float)) < 1e-4 and all(max(o['max_abs_diff_S_a'], o['max_abs_diff_S_b'], o['max_abs_diff_S_h']) < 5e-3 for o in chk['outlier_columns'].values()), \
    'served code does not reproduce the export'

U = zrow(S_a) + zrow(S_b)
stats = {'mu_a': np.nanmean(S_a, 1), 'sd_a': np.nanstd(S_a, 1), 'mu_b': np.nanmean(S_b, 1), 'sd_b': np.nanstd(S_b, 1), 'mu_u': np.nanmean(U, 1), 'sd_u': np.nanstd(U, 1),
         'mu_t': np.nanmean(T, 1), 'sd_t': np.nanstd(T, 1), 'mu_h': np.nanmean(S_h, 1), 'sd_h': np.nanstd(S_h, 1)}
cat, lens = frames_to_arrays(frames); rcat, rlens = frames_to_arrays([f.cpu() for f in Q_F])
INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
tmp = INDEX_PATH.with_name(INDEX_PATH.stem + '.tmp.npz')
np.savez(tmp, files=np.array(files, dtype=object), G_A=G_A, G_B=G_B, frames=cat, lengths=lens, r_T=r_T,
         ref_query_files=np.array([p['mimic_file'] for p in pairs], dtype=object), ref_A=Q_A, ref_B=Q_B, ref_frames=rcat, ref_lengths=rlens, **stats)
tmp.replace(INDEX_PATH)
meta = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'model_sha12': model.cfg['sha12'], 'n_files': len(files), 'dropped_by_front_end': dropped, 'n_frames': int(lens.sum()),
        'encoding': 'every clip alone, fp32, towers see the clip cut to 10 s, head frames natural length (D1 protocol)', 'csls': 'k=10, r_T over the 84 SFX queries, own targets excluded',
        'verification_vs_export': chk, 'served_rule_MRR_84_queries': {'tower': mrr_tower, 'head+CSLS': mrr}, 'torch': torch.__version__, 'seconds': round(time.time() - t0)}
INDEX_PATH.with_suffix('.meta.json').write_text(json.dumps(meta, indent=1))
print(f'wrote {INDEX_PATH} ({time.time() - t0:.0f}s)')
