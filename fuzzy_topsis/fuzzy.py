"""Triangular fuzzy numbers (TFN) and linguistic variables.

A TFN Ã = (l, m, u), l <= m <= u, has the membership function
    μ(x) = (x - l)/(m - l),  l <= x <= m
           (u - x)/(u - m),  m <= x <= u
           0,                otherwise.
Arrays of TFNs are stored as numpy arrays whose last dimension has length 3.
"""
import numpy as np


def tfn(l, m, u):
    """Creates a TFN and checks that l <= m <= u."""
    if not (l <= m <= u):
        raise ValueError(f"Invalid TFN ({l}, {m}, {u}): l <= m <= u is required")
    return np.array([l, m, u], dtype=float)


def validate(a):
    a = np.asarray(a, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError("The last dimension of a TFN array must be 3")
    if np.any(a[..., 0] > a[..., 1]) or np.any(a[..., 1] > a[..., 2]):
        raise ValueError("Condition l <= m <= u is violated")
    return a


def aggregate(tfns):
    """Aggregates the ratings of several experts (Chen, 2000):
    l = min l_k,  m = mean m_k,  u = max u_k.
    Disagreement between experts widens the resulting TFN instead of being averaged out."""
    t = validate(tfns)
    return np.array([t[:, 0].min(), t[:, 1].mean(), t[:, 2].max()])


def distance(a, b):
    """Vertex distance between TFNs: d(a, b) = sqrt(1/3 * Σ (a_i - b_i)^2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))


def defuzzify(a):
    """Centroid defuzzification: (l + m + u) / 3."""
    return np.asarray(a, float).mean(axis=-1)


def scale_spread(a, alpha):
    """Scales the spread (uncertainty) of a TFN around its mode m:
    l' = m - α(m - l),  u' = m + α(u - m).  α = 0 gives a crisp number, α = 1 leaves it unchanged."""
    a = np.asarray(a, float)
    m = a[..., 1:2]
    return m + alpha * (a - m)


def linguistic_to_tfn(term, scale):
    """Converts a linguistic term into a TFN using the given scale."""
    if term not in scale:
        raise KeyError(f"Unknown linguistic term '{term}'. Allowed: {list(scale)}")
    return tfn(*scale[term])


def membership(a, x):
    """Membership function of TFN a evaluated at points x (used for plotting)."""
    l, m, u = a
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    if m > l:
        mask = (x >= l) & (x <= m)
        y[mask] = (x[mask] - l) / (m - l)
    if u > m:
        mask = (x >= m) & (x <= u)
        y[mask] = (u - x[mask]) / (u - m)
    y[np.isclose(x, m)] = 1.0
    return y
