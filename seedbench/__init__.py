"""SeedBench: honest model comparison over multiple seeds."""
from .compare import Results, escape
from .stats import benjamini_hochberg, betainc, bonferroni, bootstrap_ci, chi2_sf, cohen_d_paired, confidence_interval, friedman, hedges_g, holm, mean, paired_t, sign_flip, std, t_cdf, t_quantile, welch_t, wilcoxon

__all__ = ["Results", "escape", "benjamini_hochberg", "betainc", "bonferroni", "bootstrap_ci", "chi2_sf", "cohen_d_paired", "confidence_interval", "friedman", "hedges_g", "holm", "mean", "paired_t", "sign_flip", "std", "t_cdf", "t_quantile", "welch_t", "wilcoxon"]
