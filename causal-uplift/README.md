# Persuade: Causal Uplift Modeling

An uplift model that estimates the *incremental* effect of a marketing email on each
customer (not just who will convert, but who converts **because** you reached them),
powering the live demo at **geetpurohit.com/projects/uplift**.

- **Meta-learners:** a T-learner (two outcome models, treated vs. control) and an
  X-learner challenger, the standard heterogeneous-treatment-effect estimators.
- **Evaluation:** the **Qini curve**. Accuracy is meaningless for uplift (you never
  observe both the treated and untreated outcome for the same person), so the model
  is judged on whether ranking by predicted uplift captures more incremental visits
  per email than random targeting.
- **Deploy:** the T-learner exports to a small `uplift.json` (two logistic models plus
  an encoding spec) and the web page runs it entirely in the browser (no server, no cost).
  See [`../public/uplift/`](../public/uplift/).

## 1. Get the data

The Hillstrom MineThatData email experiment is a public CSV (no login needed):

```bash
curl -o data/hillstrom.csv \
  http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
```

64,000 customers randomly assigned to a men's email, a women's email, or no email,
with two-week visit / conversion / spend outcomes.

## 2. Train

```bash
C:/Users/sikep/miniconda3/python.exe train.py --data data/hillstrom.csv
```

This writes into `../public/uplift/`:

- `uplift.json`: the trained T-learner (replaces the placeholder the demo ships with)
- `qini.png`, `deciles.png`, `distribution.png`, `ate.png`: result charts

## 3. Publish

```bash
cd ..            # website root
npm run build    # or just redeploy, it's a static site
```

## Modelling notes

- **Treatment** = an email was sent (men's or women's arm pooled); **control** = no email.
- **Outcome** = website visit within two weeks. Conversion is far sparser (~1%), so visit
  is the primary, learnable signal; the same pipeline runs on conversion by swapping the label.
- Because assignment was **randomized**, treatment is independent of the features, so the
  difference in modeled response is an unbiased causal effect (no confounding to adjust for).
- The four-quadrant labels (Persuadables / Sure Things / Lost Causes / Sleeping Dogs) come from
  crossing the treated and control response predictions against a fixed threshold.
