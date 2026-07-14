# Credit-Default Scorecard

An explainable credit-risk model trained on public LendingClub loan-level data, powering the
live demo at **geetpurohit.com/projects/credit-risk**.

- **Scorecard:** weight-of-evidence (WOE) binning plus logistic regression, giving calibrated,
  fully interpretable probabilities (the industry-standard credit-scoring method).
- **Challenger:** an XGBoost model with SHAP, as a performance ceiling and a sanity check
  that the drivers agree with the scorecard's logic.
- **Deploy:** the model exports to a small `scorecard.json`, and the web page runs it entirely
  in the browser (no server, no cost). See [`../public/credit-risk/`](../public/credit-risk/).

## 1. Get the data

Download the LendingClub **accepted loans** CSV from Kaggle
(free account required):

> https://www.kaggle.com/datasets/wordsforthewise/lending-club

Grab `accepted_2007_to_2018Q4.csv` and drop it in a `data/` folder here:

```
credit-risk-scorecard/
  data/
    accepted_2007_to_2018Q4.csv
  train.py
```

## 2. Train

Python is already installed via Miniconda; `xgboost` and `shap` were added. Run:

```bash
C:/Users/sikep/miniconda3/python.exe train.py --data data/accepted_2007_to_2018Q4.csv
```

Options:
- `--sample 400000` (default) caps rows for speed; raise or drop it for the full set.

This writes into `../public/credit-risk/`:
- `scorecard.json`: the trained model (replaces the placeholder the demo ships with)
- `roc.png`, `calibration.png`, `ks.png`, `iv.png`, `shap.png`: result charts

## 3. Publish

```bash
cd ..            # website root
npm run build    # or just redeploy, it's a static site
```

The page reads `scorecard.json` and the charts as static assets, so once they're committed and
the site redeploys, the trained model and plots go live automatically.

## Modelling notes
- Uses **origination-time, applicant features only** (FICO, DTI, income, revolving utilization,
  loan amount, employment length, term, home ownership, purpose). It deliberately **excludes**
  `int_rate` and `grade`, since those are LendingClub's own risk assessment and would leak the target.
- Target: completed loans labelled **default** (Charged Off / Default) vs. **paid** (Fully Paid).
  In-flight (`Current`) loans are dropped to avoid unknown outcomes.
- Metrics reported on a 25% held-out test set: ROC-AUC, KS, Gini, Brier (calibration).
