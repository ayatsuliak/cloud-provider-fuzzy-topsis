# Cloud Provider Selection under Uncertainty (AHP + Fuzzy TOPSIS)

*Вибір постачальника хмарних сервісів в умовах невизначеності — індивідуальне завдання 3.*

A small decision support system that ranks cloud service providers when the input data is
incomplete, fuzzy and contradictory. Uncertainty is represented with triangular fuzzy numbers
(TFNs), criteria weights are derived with group AHP, alternatives are ranked with fuzzy TOPSIS,
and the result is stress-tested with scenario, sensitivity and Monte Carlo analysis.

## Method

1. **Input**: 5 providers (AWS, Azure, Google Cloud, Oracle Cloud, Hetzner), 8 criteria
   (cost, performance, availability, reliability, security, scalability, support, compliance),
   3 experts (CTO, CISO, CFO).
2. **Uncertainty**: quantitative criteria (cost, benchmark score, downtime) are given as TFNs
   `(min, most likely, max)`; qualitative criteria are linguistic ratings (`VP`…`VG`)
   converted to TFNs and aggregated as `(min l, mean m, max u)`, so expert disagreement widens
   the number instead of being averaged out.
3. **Weights**: each expert fills an 8×8 Saaty pairwise comparison matrix; consistency is checked
   (CR ≤ 0.1); the fuzzy weight is `(min individual, group geometric-mean matrix, max individual)`.
4. **Fuzzy TOPSIS** (Chen, 2000): linear normalization → weighted matrix → fuzzy ideal solutions →
   vertex distances → closeness coefficient `CC = d⁻ / (d* + d⁻)`.
5. **Scenarios**: base, cost-minimization, reliability/security-maximization (weight multipliers).
6. **Sensitivity**: weight sweep (×0.05…×5) with leader-change thresholds, TFN spread scaling
   (α = 0…2), alternative scales / weighting / ideal solutions / crisp TOPSIS, Monte Carlo
   (5000 iterations).
7. **Robustness**: integral rank = mean rank over scenarios, model variants and simulations.

**Result:** AWS is the most robust choice (integral rank 1.48), Google Cloud is the runner-up
(2.28); Hetzner wins only when the cost weight is raised to ≈ 0.23 or more.

> The provider data in `data/input.json` is illustrative (expert estimates for a reference
> workload), not live pricing. The method itself does not depend on it.

## Project structure

```
fuzzy_topsis/
  fuzzy.py         triangular fuzzy numbers: aggregation, distance, defuzzification, spread scaling
  ahp.py           AHP: priority vector, CI/CR, group matrix, fuzzy weights
  topsis.py        fuzzy TOPSIS and classic crisp TOPSIS (for comparison)
  problem.py       loads and validates input, builds the decision matrix and scenario weights
  sensitivity.py   weight / spread sensitivity, Monte Carlo, Spearman correlation
data/input.json    all input data (alternatives, criteria, scales, expert ratings, AHP matrices, scenarios)
main.py            runs the whole pipeline and writes tables and figures
tests/             unit tests
results/           output of `python main.py` (tables, CSVs, figures)
```

## Running

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python main.py                                   # full analysis -> results/
python main.py --mode chen --iter 10000 --out results_chen
python -m unittest discover tests
```

Options: `--input` (input JSON), `--out` (output directory), `--mode minmax|chen` (how fuzzy
ideal solutions are defined), `--iter` (Monte Carlo iterations), `--seed`.

Outputs: `results/results.md` with every table, CSV files for each step, and figures in
`results/figures/`. Report text and figure labels are in Ukrainian.

## Using your own data

Edit `data/input.json`:

- `alternatives`, `criteria` (`type`: `benefit` / `cost`, `source`: `quantitative` / `linguistic`);
- `quantitative`: TFN `[l, m, u]` per alternative;
- `linguistic`: terms (`VP`…`VG`) from each expert;
- `ahp`: expert pairwise comparison matrices (strings like `"1/3"` are allowed);
- `linguistic_weights`: alternative linguistic importance ratings (`VL`…`VH`);
- `scenarios`: criteria weight multipliers for each scenario.
