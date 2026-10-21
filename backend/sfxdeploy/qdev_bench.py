"""qvim-dev structure and metrics, copied from SFX-fusion/sfx_adapt's qdev_common.py (research workspace) with the paths pointed at
data/qvim-dev. Canonical order = `drop_duplicates('Items')` / `drop_duplicates('Query')` on the melted DEV csv: 121 items, 979 queries.
Metrics: full-gallery MRR over the 121 items, and QVIM within-class HardNeg (pool = every item listed under the QUERY's class;
rank = #pool items with a STRICTLY higher score + 1)."""
import numpy as np, pandas as pd
from collections import defaultdict
from . import config as C


def load_qdev(root=None):
    root = root or C.QVIM_DEV
    p = pd.read_csv(f'{root}/DEV Dataset.csv', skiprows=1)[['Label', 'Class', 'Items', 'Query 1', 'Query 2', 'Query 3']]
    p = p.melt(id_vars=[c for c in p.columns if 'Query' not in c], value_vars=['Query 1', 'Query 2', 'Query 3'], var_name='Query Type', value_name='Query').dropna()
    item_rows = p.drop_duplicates('Items')[['Items', 'Class']].values.tolist()
    query_rows = p.drop_duplicates('Query')[['Query', 'Class']].values.tolist()
    d = {'item_names': [n for n, _ in item_rows], 'item_class': [c for _, c in item_rows], 'query_names': [n for n, _ in query_rows], 'query_class': [c for _, c in query_rows]}
    d['item_files'] = [f'{c}/{n}' for n, c in item_rows]                       # relative to Items/ (what the app plays at /audio/qdev/...)
    d['item_paths'] = [f'{root}/Items/{c}/{n}' for n, c in item_rows]; d['query_paths'] = [f'{root}/Queries/{c}/{n}' for n, c in query_rows]
    gt = defaultdict(set)
    for q, i in zip(p['Query'], p['Items']): gt[q].add(i)
    assert all(len(v) == 1 for v in gt.values()), 'a query maps to >1 item'
    ipos = {n: k for k, n in enumerate(d['item_names'])}; d['target'] = np.array([ipos[next(iter(gt[q]))] for q in d['query_names']])
    cls_items = defaultdict(set)
    for c, i in zip(p['Class'], p['Items']): cls_items[c].add(ipos[i])
    d['pool_mask'] = np.zeros((len(d['query_names']), len(d['item_names'])), bool)
    for q, c in enumerate(d['query_class']): d['pool_mask'][q, sorted(cls_items[c])] = True
    assert d['pool_mask'][np.arange(len(d['target'])), d['target']].all()
    return d


def rank_arrays(S, d):
    """S [nQ, nI] scores (NaN row = failed query) -> per-query reciprocal ranks: rr_full (all items) and rr_pool (within-class)."""
    nQ = S.shape[0]; tgt = d['target']; rr_pool = np.full(nQ, np.nan); rr_full = np.full(nQ, np.nan)
    for q in range(nQ):
        if np.isnan(S[q]).any(): continue
        t = S[q, tgt[q]]; rr_full[q] = 1 / (1 + (S[q] > t).sum()); pool = np.where(d['pool_mask'][q])[0]
        if len(pool) < 2: continue
        rr_pool[q] = 1 / (1 + (S[q, pool] > t).sum())
    return {'rr_pool': rr_pool, 'rr_full': rr_full}


def metrics(S, d):
    ra = rank_arrays(S.astype('float64'), d)
    return {'MRR': float(np.nanmean(ra['rr_full'])), 'HardNeg': float(np.nanmean(ra['rr_pool']))}
