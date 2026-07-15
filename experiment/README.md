# Significance: A/B Test Designer & Analyzer

An interactive experimentation toolkit powering the live demo at
**geetpurohit.com/projects/experiment**.

- **Design:** power and sample-size calculation for a two-proportion test.
- **Analyze:** two-proportion z-test with a confidence interval on the lift.
- **The two traps:** live demonstrations of **peeking** (repeated looks inflating the
  false-positive rate past 25%) and **CUPED** variance reduction (~50% tighter estimates).

The interactive math runs entirely in the browser. `simulate.py` runs the Monte Carlo
simulations behind the result charts and writes the chart data the page animates:

```bash
C:/Users/sikep/miniconda3/python.exe simulate.py
```

This writes `../public/experiment/experiment.json` (chart data + headline numbers) plus
static `*.png` fallbacks. The live site renders animated SVG from the JSON, so the PNGs
are a byproduct and are gitignored.

## Notes
- No external dataset: the A/A and CUPED demonstrations are simulated from scratch, so the
  script is fully reproducible with `numpy` + `scipy` + `matplotlib`.
- The peeking simulation runs many A/A tests (no true effect) and counts how often repeated
  significance checks trip p < 0.05; CUPED simulates a correlated pre-period covariate.
