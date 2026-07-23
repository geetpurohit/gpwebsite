#!/usr/bin/env python
"""
Credit-default scorecard trainer (LendingClub loan-level data).

Builds an interpretable WOE + logistic-regression scorecard, benchmarks it against
an XGBoost challenger (with SHAP), and writes:
    ../public/credit-risk/scorecard.json   (model consumed by the live web demo)
    ../public/credit-risk/*.png            (ROC, calibration, KS, IV, SHAP charts)

Usage:
    python train.py --data data/accepted_2007_to_2018Q4.csv
    python train.py --data data/accepted_2007_to_2018Q4.csv --sample 300000

Notes:
  * Only origination-time, applicant-centric features are used (no int_rate/grade),
    so the model predicts default from the borrower's profile rather than from
    LendingClub's own risk assessment.
  * XGBoost + SHAP are optional; if not installed the scorecard is still produced.
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

OUT = (Path(__file__).resolve().parents[1] / "public" / "credit-risk")
EPS = 0.5  # Laplace smoothing for WOE

# ---- feature config: key -> display + input widget spec for the web demo ----
NUMERIC = {
    "fico":       dict(label="FICO score",              inp=dict(min=600, max=850, step=5, default=690)),
    "term":       None,  # handled as categorical below
    "dti":        dict(label="Debt-to-income",           inp=dict(min=0, max=45, step=0.5, default=18)),
    "annual_inc": dict(label="Annual income",            inp=dict(min=10000, max=300000, step=1000, default=65000)),
    "revol_util": dict(label="Revolving utilization",    inp=dict(min=0, max=150, step=1, default=45)),
    "loan_amnt":  dict(label="Loan amount",               inp=dict(min=1000, max=40000, step=500, default=15000)),
    "emp_length": dict(label="Employment length",        inp=dict(min=0, max=10, step=1, default=5)),
}
NUMERIC.pop("term")
CATEG = {
    "term":           dict(label="Term",           default="36"),
    "home_ownership": dict(label="Home ownership",  default="MORTGAGE"),
    "purpose":        dict(label="Loan purpose",    default="debt_consolidation"),
}
NUM_KEYS = list(NUMERIC.keys())
CAT_KEYS = list(CATEG.keys())

RAW_COLS = ["loan_status", "fico_range_low", "fico_range_high", "dti", "annual_inc",
            "revol_util", "loan_amnt", "emp_length", "term", "home_ownership", "purpose"]

BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


def load(path: str, sample: int | None) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, usecols=lambda c: c in RAW_COLS, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    df = df[df["loan_status"].isin(BAD | GOOD)].copy()
    df["y"] = df["loan_status"].isin(BAD).astype(int)

    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0]
    df["emp_length"] = (df["emp_length"].astype(str)
                        .str.replace("10+ years", "10", regex=False)
                        .str.replace("< 1 year", "0", regex=False)
                        .str.extract(r"(\d+)")[0].astype(float))
    df["revol_util"] = pd.to_numeric(
        df["revol_util"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    for c in ["dti", "annual_inc", "loan_amnt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["home_ownership"] = df["home_ownership"].where(
        df["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"]), "OTHER")
    top_purpose = df["purpose"].value_counts().head(7).index
    df["purpose"] = df["purpose"].where(df["purpose"].isin(top_purpose), "other")

    keep = NUM_KEYS + CAT_KEYS + ["y"]
    df = df[keep].dropna(subset=["y"])
    for c in NUM_KEYS:
        df[c] = df[c].fillna(df[c].median())
    for c in CAT_KEYS:
        df[c] = df[c].fillna(df[c].mode()[0])

    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    print(f"  modelled rows: {len(df):,}   default rate: {df['y'].mean():.3f}")
    return df


def woe_table(good, bad):
    """WOE + IV from per-bin good/bad counts (Series indexed by bin)."""
    tg, tb = good.sum(), bad.sum()
    dg = (good + EPS) / (tg + EPS * len(good))
    db = (bad + EPS) / (tb + EPS * len(bad))
    woe = np.log(dg / db)
    iv = float(((dg - db) * woe).sum())
    return woe, iv


def bin_numeric(x: pd.Series, y: pd.Series, n_bins=6):
    cats, edges = pd.qcut(x, q=n_bins, duplicates="drop", retbins=True, labels=False)
    g = pd.DataFrame({"b": cats, "y": y})
    good = g[g.y == 0].groupby("b").size().reindex(range(len(edges) - 1), fill_value=0)
    bad = g[g.y == 1].groupby("b").size().reindex(range(len(edges) - 1), fill_value=0)
    woe, iv = woe_table(good, bad)
    bins, transform = [], {}
    for i in range(len(edges) - 1):
        top = None if i == len(edges) - 2 else round(float(edges[i + 1]), 4)
        lo = round(float(edges[i]), 2)
        hi = "∞" if top is None else round(float(edges[i + 1]), 2)
        bins.append({"max": top, "woe": round(float(woe.iloc[i]), 5),
                     "label": f"{lo}-{hi}"})
        transform[i] = float(woe.iloc[i])
    x_woe = pd.Series(cats).map(transform).fillna(0).values
    return bins, iv, x_woe


def bin_categ(x: pd.Series, y: pd.Series):
    g = pd.DataFrame({"c": x.astype(str), "y": y})
    good = g[g.y == 0].groupby("c").size()
    bad = g[g.y == 1].groupby("c").size()
    idx = good.index.union(bad.index)
    good = good.reindex(idx, fill_value=0)
    bad = bad.reindex(idx, fill_value=0)
    woe, iv = woe_table(good, bad)
    cats = [{"value": str(k), "label": str(k).replace("_", " ").title(), "woe": round(float(woe[k]), 5)}
            for k in idx]
    mp = {str(k): float(woe[k]) for k in idx}
    x_woe = x.astype(str).map(mp).fillna(0).values
    return cats, iv, mp, x_woe


def styled_ax(ax, title):
    ax.set_facecolor("#0b0713")
    ax.set_title(title, color="#f1edfb", fontsize=12, fontfamily="DejaVu Sans")
    ax.tick_params(colors="#948cad", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#2a2440")
    ax.grid(color="#1c1730", lw=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to LendingClub accepted-loans CSV")
    ap.add_argument("--sample", type=int, default=400000, help="max rows to model (speed)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load(args.data, args.sample)
    y = df["y"].values

    # ---- WOE transform every feature ----
    features, X_cols, iv_map = [], {}, {}
    cat_maps = {}
    for k in NUM_KEYS:
        bins, iv, xw = bin_numeric(df[k], df["y"])
        X_cols[k] = xw
        iv_map[k] = iv
        features.append({"key": k, "label": NUMERIC[k]["label"], "type": "numeric",
                         "input": NUMERIC[k]["inp"], "iv": round(iv, 4), "bins": bins})
    for k in CAT_KEYS:
        cats, iv, mp, xw = bin_categ(df[k], df["y"])
        X_cols[k] = xw
        iv_map[k] = iv
        cat_maps[k] = mp
        features.append({"key": k, "label": CATEG[k]["label"], "type": "categorical",
                         "input": {"default": CATEG[k]["default"]}, "iv": round(iv, 4),
                         "categories": cats})

    order = NUM_KEYS + CAT_KEYS
    X = np.column_stack([X_cols[k] for k in order])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # ---- scorecard: logistic regression on WOE ----
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Xtr, ytr)
    p_te = lr.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te)
    fpr, tpr, _ = roc_curve(yte, p_te)
    ks = float(np.max(tpr - fpr))
    gini = 2 * auc - 1
    brier = brier_score_loss(yte, p_te)
    print(f"Scorecard  AUC={auc:.3f}  KS={ks:.3f}  Gini={gini:.3f}  Brier={brier:.3f}")

    coef = {k: float(c) for k, c in zip(order, lr.coef_[0])}
    for f in features:
        f["coef"] = round(coef[f["key"]], 5)

    # ---- chart data (downsampled) for the in-browser animated SVG charts ----
    def _ds(a, n=60):
        a = np.asarray(a, float)
        if len(a) <= n:
            return [round(float(x), 4) for x in a]
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        return [round(float(a[i]), 4) for i in idx]

    _fpos, _mpred = calibration_curve(yte, p_te, n_bins=10)
    _kthr = np.linspace(0, 1, 100)
    _cbad = np.array([(p_te[yte == 1] <= tt).mean() for tt in _kthr])
    _cgood = np.array([(p_te[yte == 0] <= tt).mean() for tt in _kthr])
    _iv_sorted = sorted([(f["label"], f["iv"]) for f in features], key=lambda kv: kv[1])
    label_of = {f["key"]: f["label"] for f in features}
    charts = {
        "roc": {"fpr": _ds(fpr), "tpr": _ds(tpr)},
        "calibration": {"mean_pred": [round(float(x), 4) for x in _mpred],
                        "frac_pos": [round(float(x), 4) for x in _fpos]},
        "ks": {"thr": _ds(_kthr), "good": _ds(_cgood), "bad": _ds(_cbad)},
        "iv": {"labels": [k for k, _ in _iv_sorted], "values": [round(float(v), 4) for _, v in _iv_sorted]},
    }

    # points scaling (PDO)
    pdo, base_score, base_odds = 20, 600, 20
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)

    # ---- XGBoost challenger + SHAP (optional) ----
    xgb_auc = None
    try:
        import xgboost as xgb
        Xr = df[NUM_KEYS].copy()
        for k in CAT_KEYS:
            Xr[k] = df[k].astype("category").cat.codes
        Xrtr, Xrte, _, _ = train_test_split(Xr.values, y, test_size=0.25, random_state=42, stratify=y)
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                                n_jobs=4, tree_method="hist")
        clf.fit(Xrtr, ytr)
        xgb_p = clf.predict_proba(Xrte)[:, 1]
        xgb_auc = float(roc_auc_score(yte, xgb_p))
        print(f"XGBoost    AUC={xgb_auc:.3f}")
        _xf, _xt, _ = roc_curve(yte, xgb_p)
        charts["roc"]["xgb_fpr"] = _ds(_xf)
        charts["roc"]["xgb_tpr"] = _ds(_xt)

        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            samp = Xr.sample(min(4000, len(Xr)), random_state=1)
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(samp)
            _svm = sv[-1] if isinstance(sv, list) else sv
            _imp = np.abs(_svm).mean(axis=0)
            _names = NUM_KEYS + CAT_KEYS
            _sorder = np.argsort(_imp)
            charts["shap"] = {"labels": [label_of.get(_names[i], _names[i]) for i in _sorder],
                              "values": [round(float(_imp[i]), 4) for i in _sorder]}
            plt.figure(facecolor="#020104")
            shap.summary_plot(sv, samp, feature_names=NUM_KEYS + CAT_KEYS, show=False, plot_size=(7, 4.5))
            plt.gcf().set_facecolor("#020104")
            plt.savefig(OUT / "shap.png", dpi=140, bbox_inches="tight", facecolor="#020104")
            plt.close()
            print("  wrote shap.png")
        except Exception as e:
            print(f"  SHAP skipped: {e}")
    except Exception as e:
        print(f"XGBoost skipped: {e}")

    # ---- charts ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.plot(fpr, tpr, color="#c084fc", lw=2, label=f"Scorecard (AUC {auc:.3f})")
        if xgb_auc:
            xf, xt, _ = roc_curve(yte, xgb_p)
            ax.plot(xf, xt, color="#e879f9", lw=1.6, ls="--", label=f"XGBoost (AUC {xgb_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#4b4368", lw=1, ls=":")
        styled_ax(ax, "ROC curve")
        ax.set_xlabel("False positive rate", color="#c7c0e0", fontsize=9)
        ax.set_ylabel("True positive rate", color="#c7c0e0", fontsize=9)
        ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
        fig.savefig(OUT / "roc.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

        frac_pos, mean_pred = calibration_curve(yte, p_te, n_bins=10)
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.plot([0, 1], [0, 1], color="#4b4368", lw=1, ls=":")
        ax.plot(mean_pred, frac_pos, "o-", color="#a855f7", lw=2, ms=5)
        styled_ax(ax, "Calibration")
        ax.set_xlabel("Predicted default probability", color="#c7c0e0", fontsize=9)
        ax.set_ylabel("Observed default rate", color="#c7c0e0", fontsize=9)
        fig.savefig(OUT / "calibration.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

        thr = np.linspace(0, 1, 200)
        cum_bad = np.array([(p_te[yte == 1] <= t).mean() for t in thr])
        cum_good = np.array([(p_te[yte == 0] <= t).mean() for t in thr])
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.plot(thr, cum_good, color="#34d399", lw=2, label="Goods")
        ax.plot(thr, cum_bad, color="#f43f5e", lw=2, label="Bads")
        ax.fill_between(thr, cum_good, cum_bad, color="#a855f7", alpha=0.12)
        styled_ax(ax, f"KS separation (KS = {ks:.3f})")
        ax.set_xlabel("Score threshold (P default)", color="#c7c0e0", fontsize=9)
        ax.legend(facecolor="#0b0713", edgecolor="#2a2440", labelcolor="#f1edfb", fontsize=8)
        fig.savefig(OUT / "ks.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)

        ivs = sorted(iv_map.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="#020104")
        ax.barh([k for k, _ in ivs], [v for _, v in ivs], color="#8b5cf6")
        styled_ax(ax, "Information value by feature")
        ax.set_xlabel("IV", color="#c7c0e0", fontsize=9)
        fig.savefig(OUT / "iv.png", dpi=140, bbox_inches="tight", facecolor="#020104"); plt.close(fig)
        print("  wrote roc.png, calibration.png, ks.png, iv.png")
    except Exception as e:
        print(f"Charts skipped: {e}")

    # ---- export scorecard.json ----
    out = {
        "meta": {
            "dataset": "LendingClub accepted loans (2007-2018)",
            "generated": "trained",
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "default_rate": round(float(y.mean()), 4),
        },
        "metrics": {
            "scorecard": {"auc": round(float(auc), 4), "ks": round(ks, 4),
                          "gini": round(float(gini), 4), "brier": round(float(brier), 4)},
            "xgboost": {"auc": round(xgb_auc, 4)} if xgb_auc else {},
        },
        "scaling": {"base_score": base_score, "base_odds": base_odds, "pdo": pdo,
                    "factor": round(factor, 4), "offset": round(offset, 4)},
        "intercept": round(float(lr.intercept_[0]), 5),
        "features": features,
        "charts": charts,
    }
    (OUT / "scorecard.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT / 'scorecard.json'}")
    print("Done. Rebuild / redeploy the site to publish the trained model.")


if __name__ == "__main__":
    main()
