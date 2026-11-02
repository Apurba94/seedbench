"""Summarise a results CSV with the columns model, metric, seed and value.

Usage: python -m seedbench results.csv [baseline model]
"""
import sys

from . import Results


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8", newline="") as f:
        r = Results.from_csv(f.read())
    baseline = argv[2] if len(argv) > 2 else None
    for metric in r.metrics():
        print(f"\n## {metric}\n")
        print(r.markdown(metric))
        warning = r.power_warning(metric)
        if warning:
            print(f"\nWarning: {warning}.")
        print()
        for row in r.compare(metric, baseline):
            print(f"{row['a']} vs {row['b']}: difference {row['difference']:+.4f} [{row['low']:+.4f}, {row['high']:+.4f}], d = {row['effect_size']:.2f}, adjusted p = {row['p_adjusted']:.4f}")
    print("\n" + r.latex())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
