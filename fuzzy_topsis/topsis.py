"""Fuzzy TOPSIS (Chen, 2000) with triangular fuzzy numbers.

Notation: X - fuzzy decision matrix of shape (m alternatives, n criteria, 3),
W - fuzzy weights of shape (n, 3), types - list of 'benefit' / 'cost'.
"""
import numpy as np

from .fuzzy import distance, validate


def normalize(X, types):
    """Linear normalization of the fuzzy decision matrix.

    Benefit criterion: r_ij = (l_ij / u_j*, m_ij / u_j*, u_ij / u_j*),  u_j* = max_i u_ij
    Cost criterion:    r_ij = (l_j⁻ / u_ij, l_j⁻ / m_ij, l_j⁻ / l_ij), l_j⁻ = min_i l_ij
    The result consists of TFNs within [0, 1]."""
    X = validate(X)
    R = np.empty_like(X)
    for j, t in enumerate(types):
        col = X[:, j, :]
        if t == "benefit":
            R[:, j, :] = col / col[:, 2].max()
        elif t == "cost":
            if np.any(col <= 0):
                raise ValueError(f"Cost criterion #{j + 1} must have positive values")
            R[:, j, :] = col[:, 0].min() / col[:, ::-1]
        else:
            raise ValueError(f"Unknown criterion type: {t}")
    return R


def normalize_weights(W):
    """Normalizes fuzzy weights so that the modal values sum to 1."""
    W = validate(W)
    return W / W[:, 1].sum()


def ideal_solutions(V, mode="minmax"):
    """Fuzzy positive (FPIS, A*) and negative (FNIS, A⁻) ideal solutions.

    mode='chen'   - A*_j = (1, 1, 1),  A⁻_j = (0, 0, 0)  (Chen, 2000);
    mode='minmax' - A*_j = (max_i u_ij)×3,  A⁻_j = (min_i l_ij)×3, i.e. the best/worst
                    observed values; separates close alternatives better."""
    n = V.shape[1]
    if mode == "chen":
        return np.ones((n, 3)), np.zeros((n, 3))
    if mode == "minmax":
        pos = np.repeat(V[:, :, 2].max(axis=0)[:, None], 3, axis=1)
        neg = np.repeat(V[:, :, 0].min(axis=0)[:, None], 3, axis=1)
        return pos, neg
    raise ValueError(f"Unknown ideal solution mode: {mode}")


def rank(scores):
    """Ranks (1 = best) in descending order of scores."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    r = np.empty(len(order), dtype=int)
    r[order] = np.arange(1, len(order) + 1)
    return r


def fuzzy_topsis(X, types, W, mode="minmax"):
    """Full fuzzy TOPSIS algorithm. Returns a dict with all intermediate results.

    1. R = normalize(X)                    - normalized matrix;
    2. V = R ⊗ W                            - weighted normalized matrix;
    3. A*, A⁻                              - ideal solutions;
    4. d_i* = Σ_j d(v_ij, A*_j), d_i⁻ = Σ_j d(v_ij, A⁻_j);
    5. CC_i = d_i⁻ / (d_i* + d_i⁻)        - closeness coefficient, higher is better."""
    R = normalize(X, types)
    V = R * validate(W)[None, :, :]
    pos, neg = ideal_solutions(V, mode)
    d_pos = distance(V, pos[None, :, :]).sum(axis=1)
    d_neg = distance(V, neg[None, :, :]).sum(axis=1)
    cc = d_neg / (d_pos + d_neg)
    return {"R": R, "V": V, "fpis": pos, "fnis": neg,
            "d_pos": d_pos, "d_neg": d_neg, "cc": cc, "rank": rank(cc)}


def crisp_topsis(x, types, w):
    """Classic (crisp) TOPSIS with vector normalization, used for comparison."""
    x = np.asarray(x, float)
    r = x / np.sqrt((x ** 2).sum(axis=0))
    v = r * np.asarray(w, float)
    benefit = np.array([t == "benefit" for t in types])
    pos = np.where(benefit, v.max(axis=0), v.min(axis=0))
    neg = np.where(benefit, v.min(axis=0), v.max(axis=0))
    d_pos = np.sqrt(((v - pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - neg) ** 2).sum(axis=1))
    cc = d_neg / (d_pos + d_neg)
    return {"cc": cc, "rank": rank(cc)}
