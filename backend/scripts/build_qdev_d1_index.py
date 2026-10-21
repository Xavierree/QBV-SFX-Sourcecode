#!/usr/bin/env python3
"""Build the qvim-dev index of the D1 system (data/indexes/qdev_d1.npz + .meta.json) and verify it against the research numbers of this same system
(models/d1/reference_tables.npz, models/d1/config.json -> zero_shot_research_numbers).

Contents: the 121 benchmark items encoded ALONE by the two reference towers (cut to 10 s, no padding) and the head's reference side (natural length, clips > 30 s capped), and the CSLS
hub term r_T(g) = mean of g's top-10 fused scores over the 979 benchmark queries (no exclusion; the research `csls(f)`).

Refuses to write the index unless (1) the offline tower / tower+head / tower+CSLS / tower+head+CSLS MRR and within-class HardNeg reproduce the research values to < 1e-3 and at most 10
per-query ranks differ from the research arrays (the research head frames were computed on the CPU, the service uses the GPU), and (2) the SERVED scoring path (QdevSearch.score, one query at a time) reproduces the offline rows.
usage: .venv/bin/python scripts/build_qdev_d1_index.py"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, torch
import sfxdeploy  # noqa: F401
from sfxdeploy import config as C
from sfxdeploy.audio import load_many
from sfxdeploy.softdtw_head import sdtw_matrix, cos_np, zrow, frames_to_arrays
from sfxdeploy.d1 import D1Model, D1_DIR, fuse, csls_hub
from sfxdeploy.qdev_bench import load_qdev, metrics, rank_arrays
from sfxdeploy.qdev_search import QDEV_INDEX_PATH, QdevSearch

t0 = time.time()
dq = load_qdev(); nQ, nI = len(dq['query_names']), len(dq['item_names'])
assert (nQ, nI) == (979, 121), (nQ, nI)
model = D1Model(); ref = np.load(D1_DIR / 'reference_tables.npz', allow_pickle=True); cfg_ref = model.cfg['zero_shot_research_numbers']

# Protocol of the research numbers: the TOWERS saw every clip capped at 30 s (and cut to 10 s), the head similarities came from a frame cache that did NOT cap (one item, a rain recording longer than 30 s,
# differs: 0.2774 instead of 0.2777 head-stage MRR if both are capped). Reproduced here so the served qvim-dev tab reproduces the thesis numbers; queries are short and unaffected.
i_waves = load_many(dq['item_paths'], 'decode items', cap=True); assert all(w is not None for w in i_waves)
i_head = load_many(dq['item_paths'], 'decode items (head, no cap)', unlimited=True); assert all(w is not None for w in i_head)
print('items whose head wave is longer than the capped one:', [(dq['item_names'][i], round(len(h) / 16000, 1), round(len(w) / 16000, 1)) for i, (h, w) in enumerate(zip(i_head, i_waves)) if len(h) != len(w)])
G_A = np.zeros((nI, 3840), 'float32'); G_B = np.zeros((nI, 768), 'float32'); frames = []
for i, w in enumerate(i_waves):
    G_A[i], G_B[i], f = model.item(w, w_head=i_head[i]); frames.append(f)
q_waves = load_many(dq['query_paths'], 'decode queries', cap=True); assert all(w is not None for w in q_waves)
Q = [model.query(w) for w in q_waves]
Q_A = np.stack([q[0] for q in Q]); Q_B = np.stack([q[1] for q in Q]); Q_F = [q[2] for q in Q]
print(f'encoded {nI} items, {nQ} queries ({time.time() - t0:.0f}s)', flush=True)

S_a, S_b = cos_np(Q_A, G_A), cos_np(Q_B, G_B); S_h = sdtw_matrix(Q_F, frames, model.device)


def variants(S_a, S_b, S_h):
    T, Fz = fuse(S_a, S_b, S_h); r_T = csls_hub(Fz); zt = zrow(T)
    return {'X': T, 'X+H': Fz, 'X+C': zt - 0.5 * csls_hub(zt)[None, :], 'X+H+C': Fz - 0.5 * r_T[None, :]}, r_T


V, r_T = variants(S_a, S_b, S_h)
nz = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
Vr, _ = variants(nz(ref['qd_Q_a'].astype('float64')) @ nz(ref['qd_I_a'].astype('float64')).T, nz(ref['qd_Q_b'].astype('float64')) @ nz(ref['qd_I_b'].astype('float64')).T, ref['qd_S_h'].astype('float64'))
worst, rows = 0.0, {}
for k, S in V.items():
    m = metrics(S, dq); ra = rank_arrays(S.astype('float64'), dq)
    d_rr = np.abs(ra['rr_full'] - ref[f'rr_qd_{k}']); d_hn = np.abs(ra['rr_pool'] - ref[f'rr_qd_hn_{k}'])
    n_diff = int((d_rr > 1e-9).sum() + (d_hn > 1e-9).sum()); d_row = float(np.nanmax(np.abs(S - Vr[k])))
    d_mean = max(abs(m['MRR'] - float(np.nanmean(ref[f'rr_qd_{k}']))), abs(m['HardNeg'] - float(np.nanmean(ref[f'rr_qd_hn_{k}']))))
    rows[k] = {**m, 'research_MRR': float(np.nanmean(ref[f'rr_qd_{k}'])), 'research_HardNeg': float(np.nanmean(ref[f'rr_qd_hn_{k}'])), 'queries_with_a_different_rank': n_diff, 'max_per_query_rr_diff': float(max(np.nanmax(d_rr), np.nanmax(d_hn))),
               'max_row_diff_vs_research_matrices': d_row}
    print(f'  {k:6s} MRR {m["MRR"]:.4f} (research {rows[k]["research_MRR"]:.4f})  HardNeg {m["HardNeg"]:.4f} (research {rows[k]["research_HardNeg"]:.4f})  queries with a different rank (MRR + HardNeg lists) {n_diff}  max |row diff| {d_row:.1e}')
    assert d_mean < 1e-3 and n_diff <= 10, f'{k}: offline qvim-dev numbers do not reproduce the research values'      # a few ranks differ (research head frames: CPU frame cache; served: GPU)
    worst = max(worst, d_mean)

# the served path, one query at a time, must reproduce the offline rows
INDEX_TMP = QDEV_INDEX_PATH.with_name(QDEV_INDEX_PATH.stem + '.tmp.npz')
cat, lens = frames_to_arrays(frames)
np.savez(INDEX_TMP, files=np.array(dq['item_files'], dtype=object), G_A=G_A, G_B=G_B, frames=cat, lengths=lens, r_T=r_T)
INDEX_TMP.with_suffix('.meta.json').write_text(json.dumps({'model_sha12': model.cfg['sha12']}))
srv = QdevSearch(model, INDEX_TMP)
served = np.stack([srv.score(q)[0] for q in Q]); dmax = float(np.nanmax(np.abs(served - V['X+H+C'])))
sm = metrics(served, dq)
print(f'  served path MRR {sm["MRR"]:.4f} HardNeg {sm["HardNeg"]:.4f}; max |row diff| vs offline {dmax:.1e}')
assert dmax < 1e-3 and abs(sm['MRR'] - rows['X+H+C']['MRR']) < 1e-3, 'served scoring path differs from the offline tables'

INDEX_TMP.replace(QDEV_INDEX_PATH); INDEX_TMP.with_suffix('.meta.json').unlink()
meta = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'model_sha12': model.cfg['sha12'], 'n_items': nI, 'n_queries_for_csls': nQ,
        'protocol': 'every clip alone; towers see clips capped at 30 s and cut to 10 s (no padding), head frames natural length (items: no cap, as the research frame cache); CSLS k=10 over the 979 queries, no exclusion',
        'verification_vs_research': rows, 'served_path': {**sm, 'max_row_diff_vs_offline': dmax}, 'torch': torch.__version__, 'seconds': round(time.time() - t0)}
QDEV_INDEX_PATH.with_suffix('.meta.json').write_text(json.dumps(meta, indent=1))
print(f'wrote {QDEV_INDEX_PATH} ({time.time() - t0:.0f}s)')
