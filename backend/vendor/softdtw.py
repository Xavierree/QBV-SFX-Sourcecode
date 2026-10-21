"""Batched soft-DTW (Cuturi & Blondel 2017) in plain PyTorch, anti-diagonal vectorised, variable lengths via per-pair (lq, lr).
    soft_dtw(cost, lq, lr, gamma)  ->  [P]   value = softmin over monotone paths of the summed cost from (0,0) to (lq-1, lr-1)
cost: [P, n, m] (cells beyond a pair's own lengths are ignored), lq/lr: LongTensor [P].  gamma -> 0 recovers ordinary DTW.
No in-place writes on tensors that require grad (each diagonal is its own tensor), so autograd memory is ~ the diagonals themselves."""
import torch

_BIG = 1e9


def _softmin3(a, b, c, gamma):
    x = torch.stack([a, b, c], dim=-1)
    return -gamma * torch.logsumexp(-x / gamma, dim=-1)


def soft_dtw(cost, lq, lr, gamma=0.1):
    P, n, m = cost.shape
    dev, dt = cost.device, cost.dtype
    K = n + m
    ii = torch.arange(n, device=dev)                                     # 0-based row index i0 = i-1
    # skew: cost on diagonal k (k = i + j with i, j 1-based) at row i0 -> cost[i0, j0], j0 = k - i - 1 = k - i0 - 2
    d_prev2 = torch.full((P, n + 1), _BIG, device=dev, dtype=dt); d_prev2[:, 0] = 0.0     # diagonal k=0: R[0,0] = 0
    d_prev1 = torch.full((P, n + 1), _BIG, device=dev, dtype=dt)                           # diagonal k=1: all boundary
    out = torch.zeros(P, device=dev, dtype=dt)
    target_k = lq + lr
    pidx = torch.arange(P, device=dev)
    for k in range(2, K + 1):
        j0 = k - ii - 2                                                  # [n]
        valid = (j0 >= 0) & (j0 < m)
        c_k = cost[:, ii.clamp(max=n - 1), j0.clamp(0, m - 1)]           # [P, n] gather along the diagonal
        sm = _softmin3(d_prev2[:, :-1], d_prev1[:, :-1], d_prev1[:, 1:], gamma)          # from (i-1,j-1), (i-1,j), (i,j-1)
        new = torch.where(valid[None, :], c_k + sm, torch.full_like(c_k, _BIG))
        d_k = torch.cat([torch.full((P, 1), _BIG, device=dev, dtype=dt), new], dim=1)    # index 0 = boundary row
        hit = (target_k == k)
        if hit.any(): out = out + torch.where(hit, d_k[pidx, lq], torch.zeros_like(out))
        d_prev2, d_prev1 = d_prev1, d_k
    return out


def soft_dtw_similarity(cos_sim, lq, lr, gamma=0.1):
    """cos_sim: [P, n, m] frame cosine similarities -> similarity in ~[-2, 0]: -softDTW(1 - cos) / (lq + lr)."""
    return -soft_dtw(1.0 - cos_sim, lq, lr, gamma) / (lq + lr).to(cos_sim.dtype)
