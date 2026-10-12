"""Problem model: loads input data and builds the fuzzy decision matrix and weights."""
import json
from dataclasses import dataclass, field

import numpy as np

from . import ahp
from .fuzzy import aggregate, linguistic_to_tfn, tfn
from .topsis import fuzzy_topsis, normalize_weights


@dataclass
class Problem:
    name: str
    alternatives: list
    criteria: list
    experts: list
    rating_scale: dict
    weight_scale: dict
    quantitative: dict
    linguistic: dict
    ahp_matrices: dict = field(default_factory=dict)
    linguistic_weights: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # ---- convenience properties ----
    @property
    def alt_ids(self):
        return [a["id"] for a in self.alternatives]

    @property
    def alt_names(self):
        return [a["name"] for a in self.alternatives]

    @property
    def crit_ids(self):
        return [c["id"] for c in self.criteria]

    @property
    def crit_names(self):
        return [c["name"] for c in self.criteria]

    @property
    def types(self):
        return [c["type"] for c in self.criteria]

    # ---- model construction ----
    def decision_matrix(self, rating_scale=None):
        """Fuzzy decision matrix X of shape (m, n, 3).

        Quantitative criteria are given directly as TFNs (min, most likely, max),
        which models uncertainty in prices, workload and downtime.
        Qualitative criteria are linguistic expert ratings converted to TFNs and aggregated."""
        scale = rating_scale or self.rating_scale
        m, n = len(self.alternatives), len(self.criteria)
        X = np.zeros((m, n, 3))
        for j, c in enumerate(self.criteria):
            cid = c["id"]
            for i, aid in enumerate(self.alt_ids):
                if c["source"] == "quantitative":
                    X[i, j] = tfn(*self.quantitative[cid][aid])
                elif c["source"] == "linguistic":
                    terms = self.linguistic[cid][aid]
                    X[i, j] = aggregate([linguistic_to_tfn(t, scale) for t in terms])
                else:
                    raise ValueError(f"Unknown data source for criterion {cid}: {c['source']}")
        return X

    def ahp_weights(self):
        """Fuzzy weights from group AHP plus a consistency report for every matrix."""
        mats = [ahp.parse_matrix(self.ahp_matrices[e["id"]]) for e in self.experts]
        W, individual, group = ahp.group_fuzzy_weights(mats)
        report = {e["id"]: ahp.consistency(A) for e, A in zip(self.experts, mats)}
        report["group"] = ahp.consistency(group)
        return normalize_weights(W), individual, report

    def linguistic_expert_weights(self):
        """Alternative weighting: weights from linguistic importance ratings (VL ... VH)."""
        W = np.array([
            aggregate([linguistic_to_tfn(self.linguistic_weights[e["id"]][cid], self.weight_scale)
                       for e in self.experts])
            for cid in self.crit_ids])
        return normalize_weights(W)

    def scenario_weights(self, W, scenario):
        """Scenario weights: base fuzzy weights times scenario multipliers, renormalized."""
        k = np.array([scenario.get("multipliers", {}).get(cid, 1.0) for cid in self.crit_ids])
        return normalize_weights(W * k[:, None])

    def solve(self, W, mode="minmax", X=None):
        return fuzzy_topsis(self.decision_matrix() if X is None else X, self.types, W, mode)


def _scale(d):
    return {k: v["tfn"] if isinstance(v, dict) else v for k, v in d.items()}


def load_problem(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    p = Problem(
        name=d.get("name", ""),
        alternatives=d["alternatives"],
        criteria=d["criteria"],
        experts=d.get("experts", []),
        rating_scale=_scale(d["linguistic_scales"]["rating"]),
        weight_scale=_scale(d["linguistic_scales"].get("weight", {})),
        quantitative=d.get("quantitative", {}),
        linguistic=d.get("linguistic", {}),
        ahp_matrices=d.get("ahp", {}),
        linguistic_weights=d.get("linguistic_weights", {}),
        scenarios=d.get("scenarios", [{"id": "base", "name": "Базовий", "multipliers": {}}]),
        raw=d,
    )
    _check(p)
    return p


def _check(p):
    for c in p.criteria:
        if c["type"] not in ("benefit", "cost"):
            raise ValueError(f"{c['id']}: type must be 'benefit' or 'cost'")
        src = p.quantitative if c["source"] == "quantitative" else p.linguistic
        missing = [a for a in p.alt_ids if a not in src.get(c["id"], {})]
        if missing:
            raise ValueError(f"Criterion {c['id']} has no ratings for alternatives {missing}")
