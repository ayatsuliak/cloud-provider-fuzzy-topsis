"""Sensitivity and robustness analysis of the fuzzy TOPSIS ranking."""
import numpy as np

from .fuzzy import scale_spread
from .topsis import fuzzy_topsis, normalize_weights, rank


def vary_weight(W, j, factor):
    """Multiplies the weight of criterion j by factor and renormalizes all weights (Σm = 1)."""
    W2 = W.copy()
    W2[j] *= factor
    return normalize_weights(W2)


def weight_sweep(X, types, W, j, factors, mode="minmax"):
    """CC of all alternatives while the weight of criterion j varies. Shape: (len(factors), m)."""
    return np.array([fuzzy_topsis(X, types, vary_weight(W, j, f), mode)["cc"] for f in factors])


def leader_change_thresholds(X, types, W, factors, mode="minmax"):
    """For each criterion, the weight multipliers closest to 1 (below and above)
    at which the top-ranked alternative changes. None means no change within the range."""
    base_leader = int(np.argmax(fuzzy_topsis(X, types, W, mode)["cc"]))
    factors = np.asarray(factors)
    out = []
    for j in range(W.shape[0]):
        leaders = np.argmax(weight_sweep(X, types, W, j, factors, mode), axis=1)
        down = [f for f, l in zip(factors, leaders) if f < 1 and l != base_leader]
        up = [f for f, l in zip(factors, leaders) if f > 1 and l != base_leader]
        new_down = leaders[factors == max(down)][0] if down else None
        new_up = leaders[factors == min(up)][0] if up else None
        out.append({"lower": max(down) if down else None, "lower_leader": new_down,
                    "upper": min(up) if up else None, "upper_leader": new_up})
    return base_leader, out


def spread_sweep(X, types, W, alphas, target="data", mode="minmax", lower=None, upper=None):
    """CC when the TFN spread (level of uncertainty) is scaled by α.
    target: 'data' - alternative ratings, 'weights' - weights, 'both' - both.
    lower, upper - admissible bounds per criterion, shape (n,), so that widened TFNs
    stay physically meaningful (no negative cost, no rating above the scale maximum)."""
    n = X.shape[1]
    lo = np.full(n, 1e-9) if lower is None else np.asarray(lower, float)
    hi = np.full(n, np.inf) if upper is None else np.asarray(upper, float)
    res = []
    for a in alphas:
        Xa = scale_spread(X, a) if target in ("data", "both") else X
        Wa = scale_spread(W, a) if target in ("weights", "both") else W
        Xa = np.clip(Xa, lo[None, :, None], hi[None, :, None])
        Wa = np.maximum(Wa, 0)
        res.append(fuzzy_topsis(Xa, types, Wa, mode)["cc"])
    return np.array(res)


def sample_triangular(T, rng):
    """Draws one random value per TFN in T from the triangular distribution (l, m, u)."""
    l, m, u = T[..., 0], T[..., 1], T[..., 2]
    U = rng.random(l.shape)
    width = u - l
    safe = np.where(width > 0, width, 1)
    fc = np.where(width > 0, (m - l) / safe, 0.5)
    left = l + np.sqrt(U * width * (m - l))
    right = u - np.sqrt((1 - U) * width * (u - m))
    return np.where(width > 0, np.where(U < fc, left, right), m)


def monte_carlo(X, types, W, n_iter=5000, vary_data=True, vary_weights=True,
                mode="minmax", seed=42):
    """Monte Carlo simulation: on each iteration criterion values and/or weights are
    sampled within their TFNs (triangular distribution) and TOPSIS is run.
    Returns rank and CC matrices of shape (n_iter, m)."""
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    ranks = np.zeros((n_iter, m), dtype=int)
    ccs = np.zeros((n_iter, m))
    for k in range(n_iter):
        Xk = X
        if vary_data:
            x = np.maximum(sample_triangular(X, rng), 1e-9)
            Xk = np.repeat(x[..., None], 3, axis=-1)
        Wk = W
        if vary_weights:
            w = sample_triangular(W, rng)
            Wk = np.repeat((w / w.sum())[:, None], 3, axis=1)
        cc = fuzzy_topsis(Xk, types, Wk, mode)["cc"]
        ccs[k], ranks[k] = cc, rank(cc)
    return ranks, ccs


def rank_frequencies(ranks):
    """Share of iterations in which alternative i took rank r: matrix (m, m)."""
    m = ranks.shape[1]
    return np.array([[np.mean(ranks[:, i] == r + 1) for r in range(m)] for i in range(m)])


def spearman(r1, r2):
    """Spearman rank correlation between two rankings (no ties)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    n = len(r1)
    return 1 - 6 * np.sum((r1 - r2) ** 2) / (n * (n ** 2 - 1))
