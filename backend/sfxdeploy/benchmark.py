"""The SFX benchmark: query -> target pairing, ranking, metrics. Matcher / rank / summarize are copied VERBATIM from
`SFX Zero-Shot Eval - DistilHuBERT-LoRA + EfficientAT dual tower.ipynb` (cells 3 and 10), so numbers computed here are the notebook's numbers."""
import os, re, json
from collections import Counter
import numpy as np
from scipy.special import gammaln
from . import config as C
AUDIO_EXTS = ('.wav', '.mp3', '.m4a', '.ogg', '.flac'); KS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]; N_BOOT = 5000

def _norm(s):  return re.sub(r'[^a-z0-9]', '', s.lower())
def _trailing_id(stem):
    m = re.search(r'(\d{3,})$', stem); return m.group(1) if m else None
def _split_category(stem):
    m = re.search(r'\s*-\s*(phrase|sound)\s*$', stem, flags=re.I)
    return (stem[:m.start()], m.group(1).lower()) if m else (stem, 'unknown')
def _strip_prefix(stem): return re.sub(r'^audio\.?\s+', '', stem.strip(), flags=re.I)

def gallery_listing(gallery_dir=C.SFX_GALLERY): return sorted(f for f in os.listdir(gallery_dir) if f.lower().endswith(AUDIO_EXTS))

def load_pairs(gallery_dir=C.SFX_GALLERY, mimic_dir=C.SFX_QUERIES):
    """-> (pairs, unmatched, gallery_all). pairs[i] = {mimic_file, mimic_path, category(phrase|sound), target_files[list]}"""
    gallery_all = gallery_listing(gallery_dir); mimic_names = sorted(f for f in os.listdir(mimic_dir) if f.lower().endswith(AUDIO_EXTS))
    _by_norm, _by_id = {}, {}
    for f in gallery_all:
        st = f.rsplit('.', 1)[0]; _by_norm.setdefault(_norm(st), []).append(f)
        t = _trailing_id(st)
        if t: _by_id.setdefault(t, []).append(f)
    def _match_segment(seg):
        seg = seg.strip(); ns = _norm(seg)
        if not ns: return []
        if ns in _by_norm: return _by_norm[ns]                                            # 1) exact
        cand = [(k, v) for k, v in _by_norm.items() if len(k) >= 6 and ns.endswith(k) and ns[:len(ns) - len(k)].isdigit()]
        if cand:                                                                          # 2) digit-prefix suffix
            cand.sort(key=lambda kv: -len(kv[0])); return _by_norm[cand[0][0]]
        t = _trailing_id(seg)                                                             # 3) unique trailing id
        return _by_id[t] if t and len(_by_id.get(t, [])) == 1 else []
    pairs, unmatched = [], []
    for f in mimic_names:
        core, category = _split_category(f.rsplit('.', 1)[0])
        targets = sorted({t for seg in _strip_prefix(core).split(';') for t in _match_segment(seg)})
        rec = {'mimic_file': f, 'mimic_path': str(mimic_dir / f), 'category': category, 'target_files': targets}
        (pairs if targets else unmatched).append(rec)
    return pairs, unmatched, gallery_all

def rank_queries(gal, gal_files, q_emb, pairs):
    pos, rows = {f: i for i, f in enumerate(gal_files)}, []
    for qv, d in zip(q_emb, pairs):
        base = {'mimic_file': d['mimic_file'], 'category': d['category'], 'target_files': d['target_files']}
        if qv is None:                       rows.append({**base, 'rank': None, 'why': 'unreadable_query'}); continue
        tg = [t for t in d['target_files'] if t in pos]
        if not tg:                           rows.append({**base, 'rank': None, 'why': 'no_target_in_gallery'}); continue
        sims = gal @ (qv / (np.linalg.norm(qv) + 1e-9)); ts = np.array([sims[pos[t]] for t in tg]); b = int(ts.argmax())
        rows.append({**base, 'rank': int((sims > ts[b]).sum()) + 1, 'best_target': tg[b], 'n_targets': len(tg)})
    return rows

def wilson(k, n, z=1.96):
    if n == 0: return [float('nan')] * 2
    p, den = k / n, 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, float(c - h)), min(1.0, float(c + h))]

_cc = {}
def _chance_one(N, m):
    """Exact chance for a query with m targets among N clips, random ranking. S(r) = P(best rank >= r) = C(N-r+1, m) / C(N, m)."""
    if (N, m) not in _cc:
        r = np.arange(1, N + 2); top = N - r + 1
        S = np.exp(np.where(top >= m, gammaln(top + 1) - gammaln(np.maximum(top - m, 0) + 1) - gammaln(N + 1) + gammaln(N - m + 1), -np.inf))
        _cc[(N, m)] = (float(((S[:-1] - S[1:]) / r[:-1]).sum()), {k: float(1 - S[k]) if k < N else 1.0 for k in KS})
    return _cc[(N, m)]

def summarize(rows, N):
    ok = [x for x in rows if x['rank'] is not None]; r = np.array([x['rank'] for x in ok], float); n = len(r)
    if n == 0: return {'n_queries': 0, 'gallery_size': N}
    out = {'n_queries': n, 'gallery_size': N}
    for k in KS:
        h = int((r <= k).sum()); out[f'R@{k}'] = h / n; out[f'R@{k}_ci'] = wilson(h, n); out[f'R@{k}_hits'] = h
    rr = 1.0 / r; out['MRR'] = float(rr.mean())
    b = rr[np.random.default_rng(0).integers(0, n, (N_BOOT, n))].mean(1); out['MRR_ci'] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out['median_rank'] = float(np.median(r)); out['median_rank_pct'] = float(np.median(r) / N * 100); out['mean_rank'] = float(r.mean())
    ch = [_chance_one(N, x['n_targets']) for x in ok]
    out['chance'] = {'MRR': float(np.mean([c[0] for c in ch])), **{f'R@{k}': float(np.mean([c[1][k] for c in ch])) for k in KS}}
    return out
