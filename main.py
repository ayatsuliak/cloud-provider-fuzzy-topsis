"""Decision support system: cloud service provider selection under uncertainty.

Usage:  python main.py [--input data/input.json] [--out results] [--mode minmax|chen] [--iter 5000]

Reads provider characteristics, AHP pairwise comparison matrices and fuzzy linguistic
expert ratings, ranks the alternatives with fuzzy TOPSIS for several scenarios, runs
sensitivity analysis and produces the final recommendation.
Outputs: results/results.md (all tables), results/*.csv, results/figures/*.png.
Report text, tables and figure labels are generated in Ukrainian.
"""
import argparse
import csv
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fuzzy_topsis import crisp_topsis, defuzzify, fuzzy_topsis, load_problem
from fuzzy_topsis import sensitivity as sa
from fuzzy_topsis.fuzzy import membership

COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]
WEIGHT_FACTORS = np.round(np.concatenate([np.linspace(0.05, 1, 20), np.linspace(1.1, 5, 40)]), 3)
ALPHAS = np.round(np.linspace(0, 2, 21), 2)


# ---------------------------------------------------------------- output helpers
def f3(a, p=3):
    return "(" + ", ".join(f"{v:.{p}f}" for v in a) + ")"


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


class Report:
    def __init__(self):
        self.parts = []

    def h(self, text, level=2):
        self.parts.append("#" * level + " " + text)
        print("\n" + "=" * 80 + "\n" + text + "\n" + "=" * 80)

    def p(self, text):
        self.parts.append(text)
        print(text)

    def table(self, header, rows, csv_path=None):
        t = md_table(header, rows)
        self.parts.append(t)
        print(t)
        if csv_path:
            write_csv(csv_path, header, rows)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.parts) + "\n")


# ---------------------------------------------------------------- alternative scales
def uniform_scale(base):
    """Uniform scale: 7 terms centred at 0, 1.67, ..., 10 with half-width 1.67."""
    step = 10 / 6
    return {t: (max(0, k * step - step), k * step, min(10, k * step + step))
            for k, t in enumerate(base)}


def wide_scale(base):
    """Wide scale: the uncertainty of every term is doubled."""
    return {t: (max(0, m - 2 * (m - l)), m, min(10, m + 2 * (u - m))) for t, (l, m, u) in base.items()}


# ---------------------------------------------------------------- plots
def plot_scales(p, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (scale, title, xmax) in zip(axes, [(p.rating_scale, "Шкала оцінок альтернатив", 10),
                                               (p.weight_scale, "Шкала важливості критеріїв", 1)]):
        x = np.linspace(0, xmax, 801)
        for k, (term, t) in enumerate(scale.items()):
            ax.plot(x, membership(t, x), color=COLORS[k % 8], lw=2, label=term)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("μ(x)")
        ax.set_ylim(0, 1.08)
        ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty(p, X, path):
    quant = [j for j, c in enumerate(p.criteria) if c["source"] == "quantitative"]
    fig, axes = plt.subplots(1, len(quant), figsize=(5 * len(quant), 3.6))
    for ax, j in zip(np.atleast_1d(axes), quant):
        lo, hi = X[:, j, 0].min(), X[:, j, 2].max()
        x = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 800)
        for i, name in enumerate(p.alt_names):
            ax.plot(x, membership(X[i, j], x), color=COLORS[i], lw=2, label=name)
        c = p.criteria[j]
        ax.set_title(f"{c['id']}: {c['name']}")
        ax.set_xlabel(c.get("unit", ""))
        ax.set_ylabel("μ(x)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(p, W_ahp, W_ling, path):
    n = len(p.criteria)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, (W, label, dx) in enumerate([(W_ahp, "МАІ (груповий)", -0.15), (W_ling, "Лінгвістичні оцінки", 0.15)]):
        ax.errorbar(x + dx, W[:, 1], yerr=[W[:, 1] - W[:, 0], W[:, 2] - W[:, 1]], fmt="o",
                    color=COLORS[k], capsize=5, lw=2, label=label)
    ax.set_xticks(x, [f"{c['id']}\n" + textwrap.fill(c["name"], 14) for c in p.criteria], fontsize=8)
    ax.set_ylabel("Вага (l, m, u)")
    ax.set_title("Нечіткі вагові коефіцієнти критеріїв")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(p, results, path):
    m, S = len(p.alternatives), len(results)
    width = 0.8 / S
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, (sc, res) in enumerate(results):
        pos = np.arange(m) + (s - (S - 1) / 2) * width
        bars = ax.bar(pos, res["cc"], width, color=COLORS[s], label=sc["name"])
        for b, r in zip(bars, res["rank"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, str(r), ha="center", fontsize=8)
    ax.set_xticks(np.arange(m), p.alt_names)
    ax.set_ylabel("Коефіцієнт близькості CC")
    ax.set_title("Рейтинг постачальників за сценаріями (цифри — ранг)")
    ax.set_ylim(0, max(r["cc"].max() for _, r in results) * 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weight_sensitivity(p, sweeps, path):
    n = len(p.criteria)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.6 * rows), sharey=True)
    for j, ax in enumerate(axes.flat):
        if j >= n:
            ax.axis("off")
            continue
        for i, name in enumerate(p.alt_names):
            ax.plot(WEIGHT_FACTORS, sweeps[j][:, i], color=COLORS[i], lw=2, label=name)
        ax.axvline(1, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"{p.criteria[j]['id']}: {p.criteria[j]['name']}", fontsize=10)
        ax.set_xlabel("множник ваги")
        ax.grid(alpha=0.3)
    axes.flat[0].set_ylabel("CC")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Чутливість CC до зміни ваги кожного критерію (базовий сценарій)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spread(p, spreads, path):
    fig, axes = plt.subplots(1, len(spreads), figsize=(5 * len(spreads), 3.8), sharey=True)
    for ax, (title, cc) in zip(axes, spreads):
        for i, name in enumerate(p.alt_names):
            ax.plot(ALPHAS, cc[:, i], color=COLORS[i], lw=2, label=name)
        ax.axvline(1, color="gray", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("α — коефіцієнт ширини ТНЧ")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("CC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monte_carlo(p, mc, path):
    fig, axes = plt.subplots(1, len(mc), figsize=(5 * len(mc), 3.8), sharey=True)
    m = len(p.alternatives)
    for ax, (sc, freq) in zip(np.atleast_1d(axes), mc):
        bottom = np.zeros(m)
        for r in range(m):
            ax.bar(p.alt_names, freq[:, r], bottom=bottom, color=COLORS[r], label=f"ранг {r + 1}")
            bottom += freq[:, r]
        ax.set_title(sc["name"], fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Частка ітерацій")
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Імітаційне моделювання (Монте-Карло): розподіл рангів")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main pipeline
def main():
    ap = argparse.ArgumentParser(description="Fuzzy TOPSIS for cloud provider selection")
    ap.add_argument("--input", default="data/input.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mode", default="minmax", choices=["minmax", "chen"],
                    help="how fuzzy ideal solutions are defined")
    ap.add_argument("--iter", type=int, default=5000, help="number of Monte Carlo iterations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    # Windows consoles default to a legacy code page that cannot print λ, ⁻, etc.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out, figs = args.out, os.path.join(args.out, "figures")
    os.makedirs(figs, exist_ok=True)
    p = load_problem(args.input)
    names, cids, m, n = p.alt_names, p.crit_ids, len(p.alternatives), len(p.criteria)
    rep = Report()
    rep.h(p.name, 1)
    rep.p(f"Режим ідеальних рішень: `{args.mode}`; ітерацій Монте-Карло: {args.iter}; seed = {args.seed}.")

    # 1-2. Alternatives and criteria
    rep.h("1. Альтернативи")
    rep.table(["Код", "Постачальник", "Опис"], [[a["id"], a["name"], a.get("description", "")] for a in p.alternatives])
    rep.h("2. Критерії оцінювання")
    rep.table(["Код", "Критерій", "Тип", "Джерело даних", "Одиниці"],
              [[c["id"], c["name"], "вигода (max)" if c["type"] == "benefit" else "витрати (min)",
                "кількісні ТНЧ" if c["source"] == "quantitative" else "лінгвістичні оцінки експертів",
                c.get("unit", "—")] for c in p.criteria])

    # 3. Weights (AHP)
    rep.h("3. Вагові коефіцієнти: метод аналізу ієрархій")
    W, individual, cons = p.ahp_weights()
    rep.table(["Експерт", "λmax", "CI", "CR", "Узгодженість (CR ≤ 0.1)"],
              [[e["id"] + " — " + e["role"], f"{cons[e['id']]['lambda_max']:.4f}", f"{cons[e['id']]['CI']:.4f}",
                f"{cons[e['id']]['CR']:.4f}", "так" if cons[e["id"]]["consistent"] else "НІ"] for e in p.experts]
              + [["Групова матриця", f"{cons['group']['lambda_max']:.4f}", f"{cons['group']['CI']:.4f}",
                  f"{cons['group']['CR']:.4f}", "так" if cons["group"]["consistent"] else "НІ"]],
              os.path.join(out, "ahp_consistency.csv"))
    W_ling = p.linguistic_expert_weights()
    rep.p("Індивідуальні ваги експертів (МАІ), групова нечітка вага (l — мін., m — з групової матриці, "
          "u — макс.; нормовано за Σm = 1) та, для порівняння, ваги з лінгвістичних оцінок важливості:")
    rep.table(["Критерій"] + [e["id"] for e in p.experts] + ["Нечітка вага МАІ (l, m, u)", "Чітка вага", "Лінгвістичні ваги (l, m, u)"],
              [[f"{c['id']} {c['name']}"] + [f"{individual[k, j]:.3f}" for k in range(len(p.experts))]
               + [f3(W[j]), f"{defuzzify(W[j]):.3f}", f3(W_ling[j])] for j, c in enumerate(p.criteria)],
              os.path.join(out, "weights.csv"))
    plot_scales(p, os.path.join(figs, "01_linguistic_scales.png"))
    plot_weights(p, W, W_ling, os.path.join(figs, "02_weights.png"))

    # 4-5. Uncertainty and the fuzzy decision matrix
    X = p.decision_matrix()
    rep.h("4–5. Нечітка матриця рішень")
    rep.p("Лінгвістичні оцінки експертів (E1/E2/E3):")
    ling = [c for c in p.criteria if c["source"] == "linguistic"]
    rep.table(["Постачальник"] + [f"{c['id']} {c['name']}" for c in ling],
              [[a["name"]] + ["/".join(p.linguistic[c["id"]][a["id"]]) for c in ling] for a in p.alternatives])
    rep.p("Агреговані ТНЧ (l = min, m = середнє, u = max):")
    rep.table(["Постачальник"] + cids, [[names[i]] + [f3(X[i, j], 1) for j in range(n)] for i in range(m)],
              os.path.join(out, "decision_matrix.csv"))
    rep.p("Відносна ширина ТНЧ (u − l)/m — міра невизначеності кожної оцінки:")
    rel = (X[:, :, 2] - X[:, :, 0]) / X[:, :, 1]
    rep.table(["Постачальник"] + cids, [[names[i]] + [f"{rel[i, j]:.2f}" for j in range(n)] for i in range(m)])
    plot_uncertainty(p, X, os.path.join(figs, "03_quantitative_uncertainty.png"))

    # 6-7. Normalization and fuzzy TOPSIS (base scenario)
    base = fuzzy_topsis(X, p.types, W, args.mode)
    rep.h("6. Нормалізована нечітка матриця рішень")
    rep.table(["Постачальник"] + cids, [[names[i]] + [f3(base["R"][i, j]) for j in range(n)] for i in range(m)],
              os.path.join(out, "normalized_matrix.csv"))
    rep.h("7. Нечіткий TOPSIS — базовий сценарій")
    rep.p("Зважена нормалізована матриця V = R ⊗ W:")
    rep.table(["Постачальник"] + cids, [[names[i]] + [f3(base["V"][i, j], 4) for j in range(n)] for i in range(m)],
              os.path.join(out, "weighted_matrix.csv"))
    rep.table(["", *cids], [["FPIS A*"] + [f"{base['fpis'][j, 0]:.4f}" for j in range(n)],
                            ["FNIS A⁻"] + [f"{base['fnis'][j, 0]:.4f}" for j in range(n)]])
    order = np.argsort(base["rank"])
    rep.table(["Ранг", "Постачальник", "d*", "d⁻", "CC"],
              [[base["rank"][i], names[i], f"{base['d_pos'][i]:.4f}", f"{base['d_neg'][i]:.4f}",
                f"{base['cc'][i]:.4f}"] for i in order], os.path.join(out, "topsis_base.csv"))

    # 8. Scenarios
    rep.h("8. Сценарії")
    results = []
    for sc in p.scenarios:
        Ws = p.scenario_weights(W, sc)
        results.append((sc, fuzzy_topsis(X, p.types, Ws, args.mode), Ws))
    rep.p("Множники ваг сценаріїв (до перенормування) та підсумкові модальні ваги m:")
    rep.table(["Сценарій"] + cids,
              [[sc["name"]] + [f"×{sc.get('multipliers', {}).get(c, 1):g} → {Ws[j, 1]:.3f}" for j, c in enumerate(cids)]
               for sc, _, Ws in results])
    rep.table(["Постачальник"] + [f"{sc['name']}: CC (ранг)" for sc, _, _ in results],
              [[names[i]] + [f"{r['cc'][i]:.4f} ({r['rank'][i]})" for _, r, _ in results] for i in range(m)],
              os.path.join(out, "scenarios.csv"))
    plot_scenarios(p, [(sc, r) for sc, r, _ in results], os.path.join(figs, "04_scenarios.png"))

    # 9. Sensitivity analysis
    rep.h("9. Аналіз чутливості")
    rep.h("9.1. Зміна вагових коефіцієнтів", 3)
    sweeps = [sa.weight_sweep(X, p.types, W, j, WEIGHT_FACTORS, args.mode) for j in range(n)]
    plot_weight_sensitivity(p, sweeps, os.path.join(figs, "05_weight_sensitivity.png"))
    leader, thr = sa.leader_change_thresholds(X, p.types, W, WEIGHT_FACTORS, args.mode)
    rep.p(f"Лідер базового сценарію: **{names[leader]}**. Межі множника ваги критерію "
          f"(діапазон перевірки {WEIGHT_FACTORS.min()}–{WEIGHT_FACTORS.max()}), за яких змінюється лідер:")
    rows = []
    for j, t in enumerate(thr):
        lo = f"≤ {t['lower']:g} → {names[t['lower_leader']]}" if t["lower"] is not None else "не змінюється"
        hi = f"≥ {t['upper']:g} → {names[t['upper_leader']]}" if t["upper"] is not None else "не змінюється"
        rows.append([f"{cids[j]} {p.crit_names[j]}", f"{W[j, 1]:.3f}", lo, hi])
    rep.table(["Критерій", "Вага m", "Зменшення ваги", "Збільшення ваги"], rows,
              os.path.join(out, "weight_thresholds.csv"))

    rep.h("9.2. Зміна параметрів нечітких оцінок (ширини ТНЧ)", 3)
    # Physical bounds: cost-type values stay >= 10% of the smallest mode (cost/downtime never vanish),
    # benefit-type values stay non-negative, linguistic ratings stay within the scale.
    scale_max = max(t[2] for t in p.rating_scale.values())
    lower = [0.1 * X[:, j, 1].min() if t == "cost" else 0.0 for j, t in enumerate(p.types)]
    upper = [scale_max if c["source"] == "linguistic" else np.inf for c in p.criteria]
    spreads = [(t, sa.spread_sweep(X, p.types, W, ALPHAS, target, args.mode, lower, upper)) for t, target in
               [("Ширина ТНЧ оцінок альтернатив", "data"), ("Ширина ТНЧ ваг", "weights"), ("Обидва", "both")]]
    plot_spread(p, spreads, os.path.join(figs, "06_spread_sensitivity.png"))
    pick = [0, 5, 10, 15, 20]
    rep.table(["α"] + [f"{t}: ранги" for t, _ in spreads],
              [[f"{ALPHAS[k]:.1f}"] + [" ".join(str(r) for r in sa.rank(cc[k])) for _, cc in spreads] for k in pick])
    rep.p(f"Ранги наведено в порядку: {', '.join(names)}. α = 0 — чіткі значення (моди ТНЧ), α = 1 — вихідні дані, "
          "α = 2 — подвоєна невизначеність.")

    rep.h("9.3. Варіанти лінгвістичної шкали, способу задання ваг і методу", 3)
    variants = {
        "Базова модель": base,
        "Рівномірна шкала оцінок": fuzzy_topsis(p.decision_matrix(uniform_scale(p.rating_scale)), p.types, W, args.mode),
        "Широка шкала оцінок (×2 невизначеність)": fuzzy_topsis(p.decision_matrix(wide_scale(p.rating_scale)), p.types, W, args.mode),
        "Ваги з лінгвістичних оцінок": fuzzy_topsis(X, p.types, W_ling, args.mode),
        "Ідеальні рішення за Chen (1,1,1)/(0,0,0)": fuzzy_topsis(X, p.types, W, "chen" if args.mode == "minmax" else "minmax"),
        "Чіткий TOPSIS, лінійна нормалізація (центри ваги ТНЧ)": fuzzy_topsis(
            np.repeat(defuzzify(X)[..., None], 3, -1), p.types, np.repeat(defuzzify(W)[:, None], 3, 1), args.mode),
        "Чіткий TOPSIS, векторна нормалізація (класичний)": crisp_topsis(defuzzify(X), p.types, defuzzify(W)),
    }
    rep.table(["Варіант"] + names + ["ρ Спірмена з базовою"],
              [[k] + [f"{v['cc'][i]:.4f} ({v['rank'][i]})" for i in range(m)] + [f"{sa.spearman(v['rank'], base['rank']):.2f}"]
               for k, v in variants.items()], os.path.join(out, "model_variants.csv"))

    rep.h("9.4. Імітаційне моделювання (Монте-Карло)", 3)
    rep.p(f"На кожній з {args.iter} ітерацій значення критеріїв і ваги обираються випадково в межах своїх ТНЧ "
          "(трикутний розподіл), після чого розраховується рейтинг.")
    mc, mc_rows = [], []
    for sc, _, Ws in results:
        ranks, ccs = sa.monte_carlo(X, p.types, Ws, args.iter, mode=args.mode, seed=args.seed)
        freq = sa.rank_frequencies(ranks)
        mc.append((sc, freq, ranks))
        for i in range(m):
            mc_rows.append([sc["name"], names[i], f"{freq[i, 0]:.3f}", f"{freq[i, :2].sum():.3f}",
                            f"{ranks[:, i].mean():.2f}", f"{ccs[:, i].mean():.4f} ± {ccs[:, i].std():.4f}"])
    rep.table(["Сценарій", "Постачальник", "P(ранг 1)", "P(ранг ≤ 2)", "Середній ранг", "CC (сер. ± σ)"], mc_rows,
              os.path.join(out, "monte_carlo.csv"))
    plot_monte_carlo(p, [(sc, f) for sc, f, _ in mc], os.path.join(figs, "07_monte_carlo.png"))

    # 10. Scenario comparison and robustness
    rep.h("10. Порівняння рейтингів і визначення стійкого рішення")
    S = len(results)
    rep.p("Коефіцієнти рангової кореляції Спірмена між сценаріями:")
    rep.table([""] + [sc["name"] for sc, _, _ in results],
              [[results[a][0]["name"]] + [f"{sa.spearman(results[a][1]['rank'], results[b][1]['rank']):.2f}" for b in range(S)]
               for a in range(S)])
    scen_ranks = np.array([r["rank"] for _, r, _ in results])          # (S, m)
    var_ranks = np.array([v["rank"] for v in variants.values()])       # model variants
    mc_top1 = np.array([f[:, 0] for _, f, _ in mc])                    # (S, m)
    mc_top2 = np.array([f[:, :2].sum(axis=1) for _, f, _ in mc])
    mc_mean_rank = np.array([rk.mean(axis=0) for _, _, rk in mc])
    sweep_ranks = np.array([sa.rank(cc) for sw in sweeps for cc in sw])
    in_top2_sweep = (sweep_ranks <= 2).mean(axis=0)
    regret = (scen_ranks - 1).max(axis=0)
    # Integral robustness score: mean rank over scenarios, model variants and simulations (lower is better)
    robust = (scen_ranks.mean(axis=0) + mc_mean_rank.mean(axis=0) + var_ranks.mean(axis=0)) / 3
    rows = []
    for i in np.argsort(robust):
        rows.append([names[i], " / ".join(str(r) for r in scen_ranks[:, i]), f"{scen_ranks[:, i].mean():.2f}",
                     regret[i], f"{var_ranks[:, i].mean():.2f}", f"{mc_mean_rank[:, i].mean():.2f}",
                     f"{mc_top2[:, i].mean():.3f}", f"{in_top2_sweep[i]:.3f}", f"{robust[i]:.2f}"])
    rep.table(["Постачальник", "Ранги у сценаріях", "Сер. ранг (сценарії)", "Макс. втрата позицій",
               "Сер. ранг (варіанти моделі)", "Сер. ранг (Монте-Карло)", "P(топ-2), МК", "Частка топ-2 при зміні ваг",
               "Інтегральний ранг стійкості"], rows, os.path.join(out, "robustness.csv"))
    best = int(np.argmin(robust))
    rep.p(f"**Найбільш стійке рішення: {names[best]}** — найменший інтегральний ранг стійкості "
          f"({robust[best]:.2f}); у сценаріях посідає місця {', '.join(map(str, scen_ranks[:, best]))}.")

    # 12. Recommendations (generated automatically)
    rep.h("12. Підсумкова рекомендація")
    leaders = {sc["name"]: names[int(np.argmin(r["rank"]))] for sc, r, _ in results}
    for sc_name, ld in leaders.items():
        rep.p(f"- Сценарій «{sc_name}»: лідер — **{ld}**.")
    rep.p(f"- Рекомендована альтернатива за критерієм стійкості: **{names[best]}**.")

    summary = {
        "mode": args.mode,
        "weights_ahp": W.tolist(),
        "weights_linguistic": W_ling.tolist(),
        "consistency": cons,
        "scenarios": {sc["id"]: {"cc": r["cc"].tolist(), "rank": r["rank"].tolist()} for sc, r, _ in results},
        "monte_carlo_top1": {sc["id"]: f[:, 0].tolist() for sc, f, _ in mc},
        "robustness_score": robust.tolist(),
        "most_robust": names[best],
        "alternatives": names,
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    rep.save(os.path.join(out, "results.md"))
    print(f"\nРезультати збережено в каталозі '{out}'.")


if __name__ == "__main__":
    main()
