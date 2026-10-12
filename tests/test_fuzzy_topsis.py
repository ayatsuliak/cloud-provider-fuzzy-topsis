"""Unit tests for the package. Run: python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzy_topsis import (aggregate, consistency, distance, fuzzy_topsis, load_problem, normalize,
                          parse_matrix, priority_vector, scale_spread)
from fuzzy_topsis.sensitivity import monte_carlo, sample_triangular, spearman


class TestFuzzy(unittest.TestCase):
    def test_aggregate(self):
        np.testing.assert_allclose(aggregate([[7, 9, 10], [5, 7, 9], [9, 10, 10]]), [5, 26 / 3, 10])

    def test_distance(self):
        self.assertAlmostEqual(distance([0, 0, 0], [1, 1, 1]), 1.0)
        self.assertAlmostEqual(distance([1, 2, 3], [1, 2, 3]), 0.0)

    def test_spread(self):
        np.testing.assert_allclose(scale_spread([2, 4, 8], 0), [4, 4, 4])
        np.testing.assert_allclose(scale_spread([2, 4, 8], 0.5), [3, 4, 6])

    def test_triangular_sampling_in_bounds(self):
        T = np.array([[1.0, 2.0, 5.0], [3.0, 3.0, 3.0]])
        s = np.array([sample_triangular(T, np.random.default_rng(k)) for k in range(2000)])
        self.assertTrue(np.all(s[:, 0] >= 1) and np.all(s[:, 0] <= 5))
        self.assertTrue(np.all(s[:, 1] == 3))
        self.assertAlmostEqual(s[:, 0].mean(), 8 / 3, delta=0.1)  # E = (l+m+u)/3


class TestAHP(unittest.TestCase):
    def test_perfectly_consistent_matrix(self):
        w = np.array([0.5, 0.3, 0.2])
        A = w[:, None] / w[None, :]
        w2, lam = priority_vector(A)
        np.testing.assert_allclose(w2, w, atol=1e-9)
        self.assertAlmostEqual(lam, 3)
        self.assertAlmostEqual(consistency(A)["CR"], 0, places=9)

    def test_parse_and_reciprocity(self):
        A = parse_matrix([["1", "3"], ["1/3", "1"]])
        self.assertAlmostEqual(A[1, 0], 1 / 3)
        with self.assertRaises(ValueError):
            parse_matrix([["1", "3"], ["3", "1"]])


class TestTopsis(unittest.TestCase):
    def test_cost_normalization(self):
        X = np.array([[[2, 4, 5]], [[1, 2, 4]]], float)
        R = normalize(X, ["cost"])
        np.testing.assert_allclose(R[0, 0], [1 / 5, 1 / 4, 1 / 2])
        np.testing.assert_allclose(R[1, 0], [1 / 4, 1 / 2, 1])

    def test_dominated_alternative_is_last(self):
        X = np.array([[[7, 9, 10], [1, 2, 3]],     # best: high benefit, low cost
                      [[5, 7, 9], [2, 3, 4]],
                      [[1, 3, 5], [4, 5, 6]]], float)  # dominated
        W = np.array([[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]])
        for mode in ("minmax", "chen"):
            res = fuzzy_topsis(X, ["benefit", "cost"], W, mode)
            np.testing.assert_array_equal(res["rank"], [1, 2, 3])
            self.assertTrue(np.all((res["cc"] >= 0) & (res["cc"] <= 1)))

    def test_spearman(self):
        self.assertEqual(spearman([1, 2, 3], [1, 2, 3]), 1)
        self.assertEqual(spearman([1, 2, 3], [3, 2, 1]), -1)


class TestProblem(unittest.TestCase):
    def test_input_file(self):
        p = load_problem(os.path.join(os.path.dirname(__file__), "..", "data", "input.json"))
        W, _, cons = p.ahp_weights()
        self.assertAlmostEqual(W[:, 1].sum(), 1)
        self.assertTrue(all(c["consistent"] for c in cons.values()))
        X = p.decision_matrix()
        self.assertEqual(X.shape, (len(p.alternatives), len(p.criteria), 3))
        ranks, _ = monte_carlo(X, p.types, W, n_iter=50)
        self.assertEqual(ranks.shape, (50, len(p.alternatives)))


if __name__ == "__main__":
    unittest.main()
