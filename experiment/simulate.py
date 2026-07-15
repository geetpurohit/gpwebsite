#!/usr/bin/env python
"""
Experimentation demonstrations for the "Significance" project.

The interactive math (power, sample size, two-proportion z-test, confidence
intervals) runs live in the browser. This script generates the supporting
result charts via Monte Carlo simulation, so the claims on the page are backed
by real numbers rather than assertions:

    ../public/experiment/power.png    power vs. sample size, by effect size
    ../public/experiment/cuped.png    CUPED variance reduction
    ../public/experiment/peeking.png  false-positive inflation from peeking
    ../public/experiment/experiment.json   meta + simulated headline numbers

Usage:
    python simulate.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

OUT = (Path(__file__).resolve().parents[1] / "public" / "experiment")
RNG = np.random.default_rng(42)


def required_n(p, mde_rel, alpha=0.05, power=0.8):
    """Sample size per arm for a two-proportion test (relative MDE)."""
    p2 = p * (1 + mde_rel)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(2 * p * (1 - p)) + z_b * np.sqrt(p * (1 - p) + p2 * (1 - p2))) ** 2
    return num / (p2 - p) ** 2


def power_at_n(p, mde_rel, n, alpha=0.05):
    p2 = p * (1 + mde_rel)
    z_a = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n + p2 * (1 - p2) / n)
    return 1 - stats.norm.cdf(z_a - abs(p2 - p) / se)


def styled_ax(ax, title):
    ax.set_facecolor("#0b0713")
    ax.set_title(title, color="#f1edfb", fontsize=12)
    ax.tick_params(colors="#948cad", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2a2440")
    ax.grid(color="#1c1730", lw=0.6)


def sim_cuped(n=4000, rho=0.7, true_lift=0.0, trials=3000):
    """Simulate ATE estimates with and without CUPED variance reduction.

    A pre-experiment covariate X correlates with the outcome Y at rho. CUPED
    subtracts theta*X (theta chosen to minimise variance). Expected variance
    reduction is 1 - rho^2.
    """
    naive, cuped = [], []
    for _ in range(trials):
        x_c = RNG.normal(0, 1, n)
        x_t = RNG.normal(0, 1, n)
        y_c = rho * x_c + np.sqrt(1 - rho ** 2) * RNG.normal(0, 1, n)
        y_t = rho * x_t + np.sqrt(1 - rho ** 2) * RNG.normal(0, 1, n) + true_lift
        naive.append(y_t.mean() - y_c.mean())
        x_all = np.r_[x_c, x_t]; y_all = np.r_[y_c, y_t]
        theta = np.cov(x_all, y_all)[0, 1] / np.var(x_all)
        yc_adj = y_c - theta * (x_c - x_all.mean())
        yt_adj = y_t - theta * (x_t - x_all.mean())
        cuped.append(yt_adj.mean() - yc_adj.mean())
    return np.array(naive), np.array(cuped)


def sim_peeking(n_final=4000, peeks=(1, 2, 5, 10, 20), alpha=0.05, trials=4000):
    """False-positive rate under repeated significance testing (A/A, no effect)."""
    fpr = []
    for k in peeks:
        checkpoints = np.linspace(n_final / k, n_final, k).astype(int)
        hits = 0
        for _ in range(trials):
            # streaming Bernoulli(0.1) for both arms, same underlying rate
            c = RNG.random(n_final) < 0.10
            t = RNG.random(n_final) < 0.10
            flagged = False
            for m in checkpoints:
                pc, pt = c[:m].mean(), t[:m].mean()
                se = np.sqrt(pc * (1 - pc) / m + pt * (1 - pt) / m)
                if se > 0 and abs(pt - pc) / se > stats.norm.ppf(1 - alpha / 2):
                    flagged = True
                    break
            hits += flagged
        fpr.append(hits / trials)
    return list(peeks), fpr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Power vs sample size, by effect size
    ns = np.linspace(200, 40000, 200)
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
    for mde, color in [(0.02, "#c084fc"), (0.05, "#a855f7"), (0.10, "#e879f9")]:
        ax.plot(ns, [power_at_n(0.10, mde, n) for n in ns], color=color, lw=2,
                label=f"{int(mde*100)}% relative lift")
    ax.axhline(0.8, color="#34d399", lw=1.2, ls="--", label="80% power target")
    styled_ax(ax, "Statistical power vs. sample size (baseline 10%)")
    ax.set_xlabel("Users per arm", color="#c7c0e0", fontsize=9)
    ax.set_ylabel("Power (chance of detecting the lift)", color="#c7c0e0", fontsize=9)
    ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
    fig.savefig(OUT / "power.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

    # 2. CUPED variance reduction
    rho = 0.7
    naive, cuped = sim_cuped(rho=rho)
    var_red = 1 - np.var(cuped) / np.var(naive)
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
    ax.hist(naive, bins=50, color="#6b6688", alpha=0.7, label=f"Standard (SD {naive.std():.3f})")
    ax.hist(cuped, bins=50, color="#a855f7", alpha=0.8, label=f"CUPED (SD {cuped.std():.3f})")
    ax.axvline(0, color="#f43f5e", lw=1.2, ls="--")
    styled_ax(ax, f"CUPED tightens the estimate ({var_red*100:.0f}% less variance)")
    ax.set_xlabel("Estimated treatment effect (true effect = 0)", color="#c7c0e0", fontsize=9)
    ax.set_ylabel("Simulated experiments", color="#c7c0e0", fontsize=9)
    ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
    fig.savefig(OUT / "cuped.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

    # 3. Peeking problem
    peeks, fpr = sim_peeking()
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
    ax.plot(peeks, [f * 100 for f in fpr], "o-", color="#e879f9", lw=2, ms=6, label="Peeking, uncorrected")
    ax.axhline(5, color="#34d399", lw=1.2, ls="--", label="Intended 5% false-positive rate")
    styled_ax(ax, "Peeking inflates false positives")
    ax.set_xlabel("Number of times you check for significance", color="#c7c0e0", fontsize=9)
    ax.set_ylabel("Actual false-positive rate (%)", color="#c7c0e0", fontsize=9)
    ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
    fig.savefig(OUT / "peeking.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

    # ---- chart data for the in-browser animated SVG charts ----
    ns_c = np.linspace(200, 40000, 60)
    power_curves = {f"mde{int(m*100)}": [round(float(power_at_n(0.10, m, n)), 4) for n in ns_c]
                    for m in (0.02, 0.05, 0.10)}
    lo, hi = float(min(naive.min(), cuped.min())), float(max(naive.max(), cuped.max()))
    nc_counts, nc_edges = np.histogram(naive, bins=40, range=(lo, hi))
    cc_counts, _ = np.histogram(cuped, bins=40, range=(lo, hi))
    charts = {
        "power": {"ns": [round(float(n)) for n in ns_c],
                  "mde2": power_curves["mde2"], "mde5": power_curves["mde5"], "mde10": power_curves["mde10"]},
        "cuped": {"edges": [round(float(e), 4) for e in nc_edges],
                  "naive": [int(x) for x in nc_counts], "cuped": [int(x) for x in cc_counts]},
        "peeking": {"peeks": list(peeks), "fpr": [round(float(f), 4) for f in fpr]},
    }

    out = {
        "meta": {
            "generated": "trained",
            "note": "Headline numbers below are from Monte Carlo simulation in simulate.py.",
        },
        "cuped": {"rho": rho, "variance_reduction": round(float(var_red), 3),
                  "effective_sample_multiplier": round(1 / (1 - float(var_red)), 2)},
        "peeking": {"peeks": peeks, "false_positive_rate": [round(f, 3) for f in fpr]},
        "defaults": {"baseline_rate": 0.10, "mde_rel": 0.05, "alpha": 0.05, "power": 0.8,
                     "daily_traffic": 20000},
        "reference_n": {
            "mde_2pct": round(float(required_n(0.10, 0.02))),
            "mde_5pct": round(float(required_n(0.10, 0.05))),
            "mde_10pct": round(float(required_n(0.10, 0.10))),
        },
        "charts": charts,
    }
    (OUT / "experiment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote charts + experiment.json to", OUT)
    print(f"  CUPED variance reduction: {var_red*100:.1f}%   peeking FPR: {fpr}")


if __name__ == "__main__":
    main()
