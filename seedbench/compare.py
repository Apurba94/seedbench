"""Compare models over seeds, and write the result as tables a paper can use."""
import csv
import io
import itertools

from .stats import benjamini_hochberg, bonferroni, cohen_d_paired, confidence_interval, friedman, holm, mean, paired_t, sign_flip, std, wilcoxon


class Results:
    """scores: {model: {metric: [one value per seed, in the same seed order for every model]}}"""

    def __init__(self, scores=None):
        self.scores = scores or {}

    @classmethod
    def from_csv(cls, text):
        """Read long-form rows with the columns model, metric, seed and value."""
        rows = {}
        for r in csv.DictReader(io.StringIO(text)):
            rows.setdefault(r["model"], {}).setdefault(r["metric"], {})[r["seed"]] = float(r["value"])
        seeds = None
        out = {}
        for model, metrics in rows.items():
            out[model] = {}
            for metric, by_seed in metrics.items():
                if seeds is None:
                    seeds = sorted(by_seed)
                if sorted(by_seed) != seeds:
                    raise ValueError(f"{model} / {metric} has seeds {sorted(by_seed)}, expected {seeds}")
                out[model][metric] = [by_seed[s] for s in seeds]
        return cls(out)

    def models(self):
        return list(self.scores)

    def metrics(self):
        return list(next(iter(self.scores.values()))) if self.scores else []

    def summary(self, metric):
        out = {}
        for model, m in self.scores.items():
            c, lo, hi = confidence_interval(m[metric])
            out[model] = {"mean": c, "std": std(m[metric]), "low": lo, "high": hi, "n": len(m[metric])}
        return out

    def compare(self, metric, baseline=None, test="t", adjust="holm"):
        """Pairwise tests, against a baseline or among all models, with multiplicity correction."""
        models = self.models()
        pairs = [(baseline, m) for m in models if m != baseline] if baseline else list(itertools.combinations(models, 2))
        rows = []
        for a, b in pairs:
            x, y = self.scores[a][metric], self.scores[b][metric]
            if test == "t":
                p = paired_t(x, y)[1]
            elif test == "wilcoxon":
                p = wilcoxon(x, y)[1]
            elif test == "permutation":
                p = sign_flip(x, y)
            else:
                raise ValueError("test must be t, wilcoxon or permutation")
            diffs = [i - j for i, j in zip(x, y)]
            c, lo, hi = confidence_interval(diffs)
            rows.append({"a": a, "b": b, "difference": c, "low": lo, "high": hi, "effect_size": cohen_d_paired(x, y), "p": p})
        raw = [r["p"] for r in rows]
        fixed = {"holm": holm, "bonferroni": bonferroni, "bh": benjamini_hochberg, "none": lambda v: v}[adjust](raw) if raw else []
        for r, q in zip(rows, fixed):
            r["p_adjusted"] = q
        return rows

    def omnibus(self, metric):
        """Friedman test across all models, to ask first whether any difference exists."""
        names = self.models()
        table = list(zip(*[self.scores[m][metric] for m in names]))
        return friedman([list(r) for r in table])

    def power_warning(self, metric, alpha=0.05):
        """Warn if the test cannot reach significance, whatever the data, with this many seeds."""
        n = len(next(iter(self.scores.values()))[metric])
        smallest = 2 / 2 ** n
        if smallest > alpha:
            return f"with {n} seeds, a Wilcoxon or permutation test cannot give p below {smallest:.3f}"
        return None

    def latex(self, metrics=None, higher_better=True, digits=3, caption="Results over seeds.", label="tab:results"):
        """A booktabs table of mean ± std with the best mean in bold for each metric."""
        metrics = metrics or self.metrics()
        out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule"]
        out.append("Model & " + " & ".join(escape(m) for m in metrics) + " \\\\")
        out.append("\\midrule")
        best = {}
        for m in metrics:
            means = {name: mean(s[m]) for name, s in self.scores.items()}
            best[m] = max(means.values()) if higher_better else min(means.values())
        for name, s in self.scores.items():
            cells = []
            for m in metrics:
                text = f"{mean(s[m]):.{digits}f} $\\pm$ {std(s[m]):.{digits}f}"
                cells.append(f"\\textbf{{{text}}}" if mean(s[m]) == best[m] else text)
            out.append(escape(name) + " & " + " & ".join(cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        return "\n".join(out)

    def markdown(self, metric, digits=3):
        s = self.summary(metric)
        lines = [f"| Model | {metric} (mean ± std) | 95% CI |", "|---|---|---|"]
        for name, r in sorted(s.items(), key=lambda kv: -kv[1]["mean"]):
            lines.append(f"| {name} | {r['mean']:.{digits}f} ± {r['std']:.{digits}f} | {r['low']:.{digits}f} to {r['high']:.{digits}f} |")
        return "\n".join(lines)


def escape(text):
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}")):
        if a == "\\":
            text = text.replace(a, "\0")
        else:
            text = text.replace(a, b)
    return text.replace("\0", "\\textbackslash{}")
