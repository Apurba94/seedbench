import math
import random
import unittest

try:
    from scipy import stats as sp
except ImportError:
    sp = None

from seedbench import Results, benjamini_hochberg, betainc, bonferroni, bootstrap_ci, chi2_sf, cohen_d_paired, confidence_interval, escape, friedman, hedges_g, holm, mean, paired_t, sign_flip, std, t_cdf, t_quantile, welch_t, wilcoxon


class TestDistributions(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(t_cdf(0, 5), 0.5)
        self.assertAlmostEqual(t_cdf(2.228, 10), 0.975, places=3)
        self.assertAlmostEqual(t_quantile(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(chi2_sf(3.841, 1), 0.05, places=3)
        self.assertAlmostEqual(betainc(1, 1, 0.3), 0.3)

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_against_scipy(self):
        rng = random.Random(1)
        for _ in range(300):
            df = rng.choice([1, 2, 3, 5, 10, 30, 100])
            t = rng.uniform(-8, 8)
            self.assertAlmostEqual(t_cdf(t, df), sp.t.cdf(t, df), places=9)
            x = rng.uniform(0, 30)
            self.assertAlmostEqual(chi2_sf(x, df), sp.chi2.sf(x, df), places=9)


class TestTests(unittest.TestCase):
    def pairs(self, rng, n):
        a = [rng.gauss(0.8, 0.02) for _ in range(n)]
        return a, [v + rng.gauss(0.005, 0.01) for v in a]

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_paired_t_against_scipy(self):
        rng = random.Random(2)
        for _ in range(100):
            a, b = self.pairs(rng, rng.randint(3, 20))
            t, p = paired_t(a, b)
            r = sp.ttest_rel(a, b)
            self.assertAlmostEqual(t, r.statistic, places=9)
            self.assertAlmostEqual(p, r.pvalue, places=9)

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_welch_against_scipy(self):
        rng = random.Random(3)
        for _ in range(100):
            a = [rng.gauss(0, 1) for _ in range(rng.randint(3, 15))]
            b = [rng.gauss(0.5, 2) for _ in range(rng.randint(3, 15))]
            t, p, _ = welch_t(a, b)
            r = sp.ttest_ind(a, b, equal_var=False)
            self.assertAlmostEqual(t, r.statistic, places=9)
            self.assertAlmostEqual(p, r.pvalue, places=9)

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_wilcoxon_exact_against_scipy(self):
        rng = random.Random(4)
        for _ in range(100):
            a, b = self.pairs(rng, rng.randint(4, 20))
            w, p = wilcoxon(a, b)
            r = sp.wilcoxon(a, b, method="exact")
            self.assertAlmostEqual(w, r.statistic)
            self.assertAlmostEqual(p, r.pvalue, places=9)

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_friedman_against_scipy(self):
        rng = random.Random(5)
        for _ in range(100):
            rows = [[round(rng.gauss(j * 0.1, 0.3), 1) for j in range(4)] for _ in range(rng.randint(3, 12))]
            s, p = friedman(rows)
            r = sp.friedmanchisquare(*zip(*rows))
            self.assertAlmostEqual(s, r.statistic, places=8)
            self.assertAlmostEqual(p, r.pvalue, places=8)

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_hedges_style_sign_flip_is_valid(self):
        rng = random.Random(6)
        a, b = self.pairs(rng, 10)
        r = sp.permutation_test((a, b), lambda x, y: sum(x) / len(x) - sum(y) / len(y), permutation_type="samples", alternative="two-sided", n_resamples=2000, random_state=1)
        self.assertAlmostEqual(sign_flip(a, b), r.pvalue, delta=1e-9)

    def test_identical_results(self):
        self.assertEqual(paired_t([1, 2, 3], [1, 2, 3]), (0.0, 1.0))
        self.assertEqual(wilcoxon([1, 2], [1, 2]), (0.0, 1.0))
        self.assertEqual(paired_t([2, 3, 4], [1, 2, 3])[1], 0.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            paired_t([1, 2], [1])
        with self.assertRaises(ValueError):
            paired_t([1], [1])
        with self.assertRaises(ValueError):
            confidence_interval([1])

    def test_sign_flip_minimum_with_three_seeds(self):
        self.assertAlmostEqual(sign_flip([3, 4, 5], [1, 1, 1]), 0.25)

    def test_false_positive_rate_is_about_alpha(self):
        rng = random.Random(7)
        hits = 0
        for _ in range(4000):
            a = [rng.gauss(0, 1) for _ in range(8)]
            b = [rng.gauss(0, 1) for _ in range(8)]
            hits += paired_t(a, b)[1] < 0.05
        self.assertAlmostEqual(hits / 4000, 0.05, delta=0.01)


class TestIntervalsAndEffects(unittest.TestCase):
    def test_ci_hand_example(self):
        m, lo, hi = confidence_interval([1, 2, 3, 4, 5])
        half = 2.7764 * math.sqrt(2.5 / 5)
        self.assertEqual(m, 3)
        self.assertAlmostEqual(hi - 3, half, places=3)
        self.assertAlmostEqual(lo, 3 - half, places=3)

    def test_ci_coverage(self):
        rng = random.Random(8)
        hit = 0
        for _ in range(4000):
            x = [rng.gauss(10, 3) for _ in range(5)]
            _, lo, hi = confidence_interval(x)
            hit += lo <= 10 <= hi
        self.assertAlmostEqual(hit / 4000, 0.95, delta=0.01)

    def test_bootstrap(self):
        rng = random.Random(9)
        x = [rng.gauss(0, 1) for _ in range(200)]
        lo, hi = bootstrap_ci(x, rounds=2000)
        self.assertLess(lo, mean(x))
        self.assertGreater(hi, mean(x))
        self.assertAlmostEqual(hi - lo, 2 * 1.96 / math.sqrt(200), delta=0.1)

    def test_effect_sizes(self):
        self.assertAlmostEqual(cohen_d_paired([2, 3, 5], [1, 1, 2]), 2 / 1.0)
        self.assertLess(abs(hedges_g([1, 2, 3, 4], [2, 3, 4, 5])), 1)


class TestCorrections(unittest.TestCase):
    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_holm_is_monotone_and_keeps_order(self):
        rng = random.Random(10)
        for _ in range(200):
            p = [rng.random() for _ in range(8)]
            q = holm(p)
            pairs = sorted(zip(p, q))
            self.assertTrue(all(a[1] <= b[1] for a, b in zip(pairs, pairs[1:])))
            self.assertTrue(all(y >= x for x, y in zip(p, q)))

    def test_bonferroni_caps(self):
        self.assertEqual(bonferroni([0.5, 0.01]), [1.0, 0.02])

    @unittest.skipIf(sp is None, "SciPy is not installed")
    def test_bh_against_scipy(self):
        rng = random.Random(11)
        for _ in range(100):
            p = [rng.random() for _ in range(rng.randint(1, 20))]
            for x, y in zip(benjamini_hochberg(p), sp.false_discovery_control(p)):
                self.assertAlmostEqual(x, y, places=9)


class TestResults(unittest.TestCase):
    def sample(self):
        return Results({"yolo_n": {"mAP": [0.80, 0.81, 0.79, 0.80, 0.82], "F1": [0.7, 0.71, 0.72, 0.7, 0.7]},
                        "yolo_s": {"mAP": [0.84, 0.86, 0.83, 0.85, 0.86], "F1": [0.72, 0.71, 0.73, 0.7, 0.72]},
                        "base": {"mAP": [0.70, 0.72, 0.71, 0.70, 0.73], "F1": [0.6, 0.61, 0.6, 0.6, 0.62]}})

    def test_summary(self):
        s = self.sample().summary("mAP")["yolo_n"]
        self.assertAlmostEqual(s["mean"], 0.804)
        self.assertLess(s["low"], s["mean"])

    def test_compare_all_pairs(self):
        rows = self.sample().compare("mAP")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["p_adjusted"] >= r["p"] for r in rows))

    def test_compare_with_baseline(self):
        rows = self.sample().compare("mAP", "base")
        self.assertEqual({r["b"] for r in rows}, {"yolo_n", "yolo_s"})
        self.assertTrue(all(r["difference"] < 0 and r["p_adjusted"] < 0.05 for r in rows))

    def test_other_tests(self):
        for t in ("wilcoxon", "permutation"):
            self.assertEqual(len(self.sample().compare("F1", "base", t, "bh")), 2)
        with self.assertRaises(ValueError):
            self.sample().compare("F1", "base", "nope")

    def test_omnibus(self):
        s, p = self.sample().omnibus("mAP")
        self.assertLess(p, 0.05)

    def test_three_seeds_warning(self):
        r = Results({"a": {"m": [1, 2, 3]}, "b": {"m": [2, 3, 4]}})
        self.assertIn("cannot give p below 0.250", r.power_warning("m"))
        self.assertIsNone(self.sample().power_warning("mAP", alpha=0.1))

    def test_latex_marks_the_best(self):
        t = self.sample().latex(digits=2)
        self.assertIn("\\textbf{0.85 $\\pm$ 0.01}", t)
        self.assertIn("yolo\\_n", t)
        self.assertEqual(t.count("\\textbf"), 2)
        self.assertIn("\\toprule", t)

    def test_lower_is_better(self):
        r = Results({"a": {"err": [1, 1.5]}, "b": {"err": [3, 3.5]}})
        self.assertIn("\\textbf{1.250", r.latex(higher_better=False))

    def test_escape(self):
        self.assertEqual(escape("a_b & c%"), "a\\_b \\& c\\%")
        self.assertEqual(escape("a\\b"), "a\\textbackslash{}b")

    def test_csv(self):
        text = "model,metric,seed,value\na,m,1,0.5\na,m,2,0.6\nb,m,2,0.7\nb,m,1,0.8\n"
        r = Results.from_csv(text)
        self.assertEqual(r.scores["b"]["m"], [0.8, 0.7])

    def test_csv_with_missing_seed_is_refused(self):
        with self.assertRaises(ValueError):
            Results.from_csv("model,metric,seed,value\na,m,1,0.5\na,m,2,0.6\nb,m,1,0.7\n")

    def test_markdown_is_sorted(self):
        lines = self.sample().markdown("mAP").split("\n")
        self.assertTrue(lines[2].startswith("| yolo_s"))


if __name__ == "__main__":
    unittest.main()
