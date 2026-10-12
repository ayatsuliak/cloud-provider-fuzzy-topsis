"""Analytic Hierarchy Process (AHP, T. Saaty) for deriving criteria weights."""
from fractions import Fraction

import numpy as np

# Saaty's Random Index (RI) by matrix size
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def parse_matrix(rows):
    """Converts a matrix with entries like "1/3" into a numeric array and validates it."""
    A = np.array([[float(Fraction(str(v))) for v in row] for row in rows])
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("Pairwise comparison matrix must be square")
    if not np.allclose(np.diag(A), 1):
        raise ValueError("Pairwise comparison matrix must have ones on the diagonal")
    if not np.allclose(A * A.T, 1, atol=1e-6):
        raise ValueError("Pairwise comparison matrix is not reciprocal (a_ji = 1/a_ij)")
    return A


def priority_vector(A):
    """Priority vector: the normalized principal eigenvector of A.
    Returns (w, λ_max)."""
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    return w / w.sum(), float(vals[k].real)


def consistency(A):
    """Consistency index CI = (λ_max - n)/(n - 1) and consistency ratio CR = CI/RI.
    A matrix is considered acceptably consistent when CR <= 0.1."""
    n = A.shape[0]
    _, lam = priority_vector(A)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return {"lambda_max": lam, "CI": ci, "CR": cr, "consistent": cr <= 0.1}


def aggregate_matrices(matrices):
    """Group matrix: element-wise geometric mean (AIJ; preserves reciprocity)."""
    stack = np.array(matrices)
    return np.exp(np.log(stack).mean(axis=0))


def group_fuzzy_weights(matrices):
    """Fuzzy criteria weights from group AHP.

    m    - weights of the aggregated (geometric mean) matrix;
    l, u - minimum and maximum weights across individual experts.
    The width of each weight TFN therefore reflects expert disagreement."""
    individual = np.array([priority_vector(A)[0] for A in matrices])
    group = aggregate_matrices(matrices)
    w_m, _ = priority_vector(group)
    l = np.minimum(individual.min(axis=0), w_m)
    u = np.maximum(individual.max(axis=0), w_m)
    return np.stack([l, w_m, u], axis=1), individual, group
