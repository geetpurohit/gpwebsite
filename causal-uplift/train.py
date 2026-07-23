#!/usr/bin/env python
"""
Uplift-model trainer (Hillstrom MineThatData email experiment).

Estimates the *incremental* effect of sending a marketing email on whether a
customer visits the site, using causal meta-learners (T-learner and X-learner),
and evaluates them with the Qini curve, the metric that actually matters for
targeting. Exports a small two-model logistic T-learner that runs entirely in
the browser, plus result charts:

    ../public/uplift/uplift.json   (model consumed by the live web demo)
    ../public/uplift/*.png         (Qini, decile lift, uplift distribution, ATE)

Usage:
    python train.py --data data/hillstrom.csv

Notes:
  * Treatment = customer received an email (Men's or Women's arm); control = no email.
  * Outcome   = website visit within two weeks (the strongest signal in the data;
    conversion is far sparser). The experiment is randomised, so treatment
    assignment is independent of the features (clean causal identification).
  * The browser model is the T-learner (two logistic regressions). The X-learner is
    trained as a stronger challenger and reported alongside it via the Qini score.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 2.0 renamed trapz

OUT = (Path(__file__).resolve().parents[1] / "public" / "uplift")

# ---- feature config: key -> display + input widget spec for the web demo ----
# Encoded column order the browser reproduces exactly:
#   recency_z, history_log_z, mens, womens, newbie,
#   zip_Urban, zip_Rural, channel_Web, channel_Multichannel
NUMERIC = {
    "recency": dict(label="Months since last purchase",
                    inp=dict(min=1, max=12, step=1, default=6)),
    "history": dict(label="Spend last year",
                    inp=dict(min=0, max=3000, step=25, default=250), log=True),
}
BINARY = {
    "mens":   dict(label="Bought men's merchandise"),
    "womens": dict(label="Bought women's merchandise"),
    "newbie": dict(label="New customer (last 12 months)"),
}
CATEG = {
    "zip_code": dict(label="Neighborhood", base="Surburban",
                     options=[("Surburban", "Suburban"), ("Urban", "Urban"), ("Rural", "Rural")]),
    "channel":  dict(label="Past purchase channel", base="Phone",
                     options=[("Phone", "Phone"), ("Web", "Web"), ("Multichannel", "Multichannel")]),
}


def encode(df, enc):
    """Build the fixed model matrix from raw columns using shared encoding params."""
    cols = {}
    cols["recency_z"] = (df["recency"] - enc["recency"]["mean"]) / enc["recency"]["std"]
    hl = np.log1p(df["history"])
    cols["history_log_z"] = (hl - enc["history"]["mean"]) / enc["history"]["std"]
    cols["mens"] = df["mens"].astype(float)
    cols["womens"] = df["womens"].astype(float)
    cols["newbie"] = df["newbie"].astype(float)
    cols["zip_Urban"] = (df["zip_code"] == "Urban").astype(float)
    cols["zip_Rural"] = (df["zip_code"] == "Rural").astype(float)
    cols["channel_Web"] = (df["channel"] == "Web").astype(float)
    cols["channel_Multichannel"] = (df["channel"] == "Multichannel").astype(float)
    return np.column_stack([cols[c] for c in COLUMNS])


COLUMNS = ["recency_z", "history_log_z", "mens", "womens", "newbie",
           "zip_Urban", "zip_Rural", "channel_Web", "channel_Multichannel"]


def qini_curve(y, t, uplift, n_points=100):
    """Qini curve: cumulative incremental outcomes as we target by predicted uplift.

    Returns (fractions, qini_values, qini_coefficient). The curve at fraction f is
    (Y_t/N_t - Y_c/N_c) * (N_t+N_c) over the top-f ranked customers, i.e. incremental
    responses if we treated that fraction. The coefficient is the area between the
    model curve and the random (diagonal) line.
    """
    order = np.argsort(-uplift)
    y, t = y[order], t[order]
    n = len(y)
    cum_t = np.cumsum(t)                    # treated seen so far
    cum_c = np.cumsum(1 - t)                # control seen so far
    cum_yt = np.cumsum(y * t)               # treated responders so far
    cum_yc = np.cumsum(y * (1 - t))         # control responders so far
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = cum_yt - cum_yc * (cum_t / np.maximum(cum_c, 1))
    gain = np.nan_to_num(gain)

    idx = np.linspace(0, n - 1, n_points).astype(int)
    fr = (idx + 1) / n
    q = gain[idx]
    # random baseline: straight line to the endpoint
    rand = q[-1] * fr
    # normalised Qini coefficient (area between curve and random, over random area)
    area_model = trapz(q, fr)
    area_rand = trapz(rand, fr)
    coeff = (area_model - area_rand) / abs(area_rand) if area_rand else 0.0
    return fr, q, rand, float(coeff)


def styled_ax(ax, title):
    ax.set_facecolor("#0b0713")
    ax.set_title(title, color="#f1edfb", fontsize=12)
    ax.tick_params(colors="#948cad", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2a2440")
    ax.grid(color="#1c1730", lw=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to Hillstrom email CSV")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    df["t"] = (df["segment"] != "No E-Mail").astype(int)   # treatment: email sent
    df["y"] = df["visit"].astype(int)                      # outcome: site visit
    print(f"  rows: {len(df):,}   treated: {df['t'].mean():.3f}   visit rate: {df['y'].mean():.3f}")

    # shared encoding params (fit on full data; standardisation only)
    enc = {
        "recency": {"mean": float(df["recency"].mean()), "std": float(df["recency"].std())},
        "history": {"mean": float(np.log1p(df["history"]).mean()),
                    "std": float(np.log1p(df["history"]).std())},
    }
    X = encode(df, enc)
    y = df["y"].values
    t = df["t"].values

    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
        X, y, t, test_size=0.30, random_state=42, stratify=t)

    prop = float(ttr.mean())  # propensity (randomised => ~constant)
    ate = float(ytr[ttr == 1].mean() - ytr[ttr == 0].mean())
    print(f"  ATE (treated - control visit rate): {ate:+.4f}")

    # ---- T-learner: two outcome models ----
    m1 = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[ttr == 1], ytr[ttr == 1])
    m0 = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[ttr == 0], ytr[ttr == 0])
    up_t_te = m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1]
    _, q_t, rand_t, coeff_t = qini_curve(yte, tte, up_t_te)
    print(f"  T-learner Qini coefficient: {coeff_t:.4f}")

    # ---- X-learner challenger ----
    mu1 = m1.predict_proba(Xtr)[:, 1]
    mu0 = m0.predict_proba(Xtr)[:, 1]
    d_treated = ytr[ttr == 1] - mu0[ttr == 1]        # treated: Y - mu0(X)
    d_control = mu1[ttr == 0] - ytr[ttr == 0]        # control: mu1(X) - Y
    tau1 = Ridge(alpha=1.0).fit(Xtr[ttr == 1], d_treated)
    tau0 = Ridge(alpha=1.0).fit(Xtr[ttr == 0], d_control)
    up_x_te = prop * tau0.predict(Xte) + (1 - prop) * tau1.predict(Xte)
    fr, q_x, rand_x, coeff_x = qini_curve(yte, tte, up_x_te)
    print(f"  X-learner Qini coefficient: {coeff_x:.4f}")

    # population uplift stats (from T-learner, the exported model) for the demo
    up_all = m1.predict_proba(X)[:, 1] - m0.predict_proba(X)[:, 1]
    thr = float(np.median(np.r_[m1.predict_proba(X)[:, 1], m0.predict_proba(X)[:, 1]]))

    # ---- chart data for the in-browser animated SVG charts ----
    def _ds(a, n=48):
        a = np.asarray(a, float)
        if len(a) <= n:
            return [round(float(x), 3) for x in a]
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        return [round(float(a[i]), 3) for i in idx]

    _dec = pd.qcut(up_t_te, 10, labels=False, duplicates="drop")
    _df_dec = pd.DataFrame({"dec": _dec, "y": yte, "t": tte})
    _lift = (_df_dec.groupby("dec")
             .apply(lambda g: (g[g.t == 1].y.mean() - g[g.t == 0].y.mean()) * 100)
             .reindex(range(10)).fillna(0))
    _counts, _edges = np.histogram(up_all * 100, bins=28)
    charts = {
        "qini": {"fr": _ds(fr), "t": _ds(q_t), "x": _ds(q_x), "rand": _ds(rand_t)},
        "deciles": {"lift": [round(float(v), 2) for v in _lift.values], "ate": round(ate * 100, 2)},
        "distribution": {"counts": [int(c) for c in _counts],
                         "edges": [round(float(e), 2) for e in _edges]},
        "ate": {"control": round(float(df[df.t == 0].y.mean() * 100), 2),
                "treated": round(float(df[df.t == 1].y.mean() * 100), 2)},
    }

    # ---- charts ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Qini curves
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.plot(fr, q_t, color="#c084fc", lw=2, label=f"T-learner (Qini {coeff_t:.3f})")
        ax.plot(fr, q_x, color="#e879f9", lw=1.8, ls="--", label=f"X-learner (Qini {coeff_x:.3f})")
        ax.plot(fr, rand_t, color="#4b4368", lw=1.2, ls=":", label="Target at random")
        styled_ax(ax, "Qini curve: incremental visits vs. fraction targeted")
        ax.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)",
                      color="#c7c0e0", fontsize=9)
        ax.set_ylabel("Cumulative incremental visits", color="#c7c0e0", fontsize=9)
        ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
        fig.savefig(OUT / "qini.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

        # 2. Actual uplift by predicted-uplift decile (does the ranking hold up?)
        dec = pd.qcut(up_t_te, 10, labels=False, duplicates="drop")
        d = pd.DataFrame({"dec": dec, "y": yte, "t": tte})
        real = d.groupby("dec").apply(
            lambda g: g[g.t == 1].y.mean() - g[g.t == 0].y.mean())
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        colors = ["#f43f5e" if v < 0 else "#a855f7" for v in real.values]
        ax.bar(range(len(real)), real.values * 100, color=colors)
        ax.axhline(ate * 100, color="#34d399", lw=1.2, ls="--", label=f"Average lift ({ate*100:.1f}%)")
        styled_ax(ax, "Actual lift by predicted-uplift decile")
        ax.set_xlabel("Decile (0 = model says least persuadable, 9 = most)",
                      color="#c7c0e0", fontsize=9)
        ax.set_ylabel("Actual visit lift (pp)", color="#c7c0e0", fontsize=9)
        ax.set_xticks(range(len(real)))
        ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
        fig.savefig(OUT / "deciles.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

        # 3. Distribution of predicted uplift across the population
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.hist(up_all * 100, bins=40, color="#8b5cf6", alpha=0.85)
        ax.axvline(0, color="#f43f5e", lw=1.2, ls="--", label="Zero uplift")
        styled_ax(ax, "Predicted uplift across all customers")
        ax.set_xlabel("Predicted incremental visit probability (pp)", color="#c7c0e0", fontsize=9)
        ax.set_ylabel("Customers", color="#c7c0e0", fontsize=9)
        ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
        fig.savefig(OUT / "distribution.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

        # 4. Raw treated vs control response (the experiment itself)
        rt = df[df.t == 1].y.mean(); rc = df[df.t == 0].y.mean()
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.bar(["No email\n(control)", "Email\n(treated)"], [rc * 100, rt * 100],
               color=["#4b4368", "#a855f7"])
        ax.text(0, rc * 100 + 0.4, f"{rc*100:.1f}%", ha="center", color="#f1edfb", fontsize=10)
        ax.text(1, rt * 100 + 0.4, f"{rt*100:.1f}%", ha="center", color="#f1edfb", fontsize=10)
        styled_ax(ax, f"The experiment: +{ate*100:.1f}pp average lift from emailing")
        ax.set_ylabel("Visit rate (%)", color="#c7c0e0", fontsize=9)
        fig.savefig(OUT / "ate.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)
        print("  wrote qini.png, deciles.png, distribution.png, ate.png")
    except Exception as e:
        print(f"Charts skipped: {e}")

    # ---- build feature specs + export the T-learner ----
    features = []
    for k, cfg in NUMERIC.items():
        features.append({"key": k, "label": cfg["label"], "type": "numeric",
                         "input": cfg["inp"]})
    for k, cfg in BINARY.items():
        features.append({"key": k, "label": cfg["label"], "type": "binary",
                         "input": {"default": 0}})
    for k, cfg in CATEG.items():
        features.append({"key": k, "label": cfg["label"], "type": "categorical",
                         "input": {"default": cfg["base"]},
                         "categories": [{"value": v, "label": lab} for v, lab in cfg["options"]]})

    def model_dict(m):
        return {"intercept": round(float(m.intercept_[0]), 6),
                "coef": {c: round(float(w), 6) for c, w in zip(COLUMNS, m.coef_[0])}}

    out = {
        "meta": {
            "dataset": "Hillstrom MineThatData email experiment (64k customers)",
            "generated": "trained",
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "treatment": "marketing email sent", "outcome": "website visit",
            "ate": round(ate, 4), "propensity": round(prop, 4),
            "response_threshold": round(thr, 4),
            "avg_uplift": round(float(up_all.mean()), 4),
        },
        "metrics": {
            "t_learner": {"qini": round(coeff_t, 4)},
            "x_learner": {"qini": round(coeff_x, 4)},
        },
        "columns": COLUMNS,
        "encoding": enc,
        "features": features,
        "models": {"treat": model_dict(m1), "control": model_dict(m0)},
        "charts": charts,
    }
    (OUT / "uplift.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT / 'uplift.json'}")
    print("Done. Rebuild / redeploy the site to publish the trained model.")


if __name__ == "__main__":
    main()
