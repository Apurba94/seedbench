# SeedBench

"Model A beats Model B by 0.4%" means little without seeds, confidence intervals and a fair test. SeedBench turns multi-seed results into statistics a reviewer will accept, with LaTeX tables ready for a paper. It uses only the Python standard library.

```python
from seedbench import Results

r = Results({
    "YOLOv8n": {"mAP": [0.801, 0.812, 0.795]},
    "YOLO11n": {"mAP": [0.818, 0.820, 0.809]},
})
print(r.markdown("mAP"))
print(r.compare("mAP", baseline="YOLOv8n", test="t", adjust="holm"))
print(r.power_warning("mAP"))
print(r.latex())
```

From a CSV with the columns `model, metric, seed, value`:

```
python -m seedbench results.csv YOLOv8n
```

## What it gives you

- Mean, standard deviation and t-based confidence intervals, which are right for a handful of seeds
- Paired t-test, Welch's t-test, the Wilcoxon signed-rank test (exact when possible) and an exact sign-flip permutation test
- Effect sizes (paired Cohen's d, Hedges' g) and a confidence interval on each difference
- Friedman test across many models
- Holm, Bonferroni and Benjamini-Hochberg corrections
- Bootstrap intervals for any statistic
- A warning when your number of seeds makes significance impossible
- Booktabs LaTeX and Markdown tables, with the best mean in bold

## A trap this catches

With 3 seeds, a Wilcoxon or permutation test **cannot** go below p = 0.25, however large the improvement. With 5 seeds the floor is 0.0625. `power_warning` tells you before a reviewer does. For small numbers of seeds, report the effect size and the interval on the difference, and treat the p-value with caution.

## How it is tested

When SciPy is installed, the t and chi-squared distributions, paired and Welch t-tests, exact Wilcoxon, Friedman, the sign-flip test and Benjamini-Hochberg are compared with SciPy on hundreds of random cases, to at least 8 decimal places. Simulations confirm that the false positive rate stays near 5% and that 95% intervals cover the truth about 95% of the time.

## Limits

Paired tests assume each seed means the same thing across models, so use the same seed for the same split and initialisation. Tests over seeds measure stability to randomness on one dataset, and they say nothing about a new dataset. The Wilcoxon normal approximation used with ties is rough at very small sample sizes.

## Tests

```
python -m unittest discover -s tests
```

MIT licence.
