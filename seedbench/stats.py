"""Statistics written from the definitions, using only the standard library."""
import itertools
import math
import random


def mean(x):
    return math.fsum(x) / len(x)


def variance(x):
    m = mean(x)
    return math.fsum((v - m) ** 2 for v in x) / (len(x) - 1)


def std(x):
    return math.sqrt(variance(x))


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h


def betainc(a, b, x):
    """The regularised incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    p = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return p if t < 0 else 1 - p


def t_quantile(q, df):
    """Inverse of t_cdf, by bisection."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gammainc_upper(a, x):
    """The regularised upper incomplete gamma function Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1 / a
        n = a
        for _ in range(10000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c = x + 1 - a, 1e300
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    return gammainc_upper(df / 2, x / 2)


def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def confidence_interval(x, level=0.95):
    """Mean with a t-based confidence interval. The right choice for a few seeds."""
    n = len(x)
    if n < 2:
        raise ValueError("needs at least two values")
    m = mean(x)
    half = t_quantile(1 - (1 - level) / 2, n - 1) * std(x) / math.sqrt(n)
    return m, m - half, m + half


def paired_t(a, b):
    """Paired two-sided t-test. Returns (t, p)."""
    if len(a) != len(b):
        raise ValueError("paired tests need equal lengths")
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        raise ValueError("needs at least two pairs")
    sd = std(d)
    if sd == 0:
        return (0.0, 1.0) if mean(d) == 0 else (math.copysign(math.inf, mean(d)), 0.0)
    t = mean(d) / (sd / math.sqrt(n))
    return t, 2 * (1 - t_cdf(abs(t), n - 1))


def welch_t(a, b):
    """Two-sided t-test without assuming equal variances. Returns (t, p, df)."""
    va, vb, na, nb = variance(a), variance(b), len(a), len(b)
    se2 = va / na + vb / nb
    t = (mean(a) - mean(b)) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, 2 * (1 - t_cdf(abs(t), df)), df


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def wilcoxon(a, b):
    """Wilcoxon signed-rank test. Zero differences are dropped. Exact without ties when n ≤ 25,
    otherwise a normal approximation. Returns (W, p), where W is the smaller rank sum."""
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    r = _ranks([abs(v) for v in d])
    plus = math.fsum(rk for rk, v in zip(r, d) if v > 0)
    minus = math.fsum(rk for rk, v in zip(r, d) if v < 0)
    w = min(plus, minus)
    if len(set(r)) == n and n <= 25:
        counts = [1]
        for k in range(1, n + 1):
            new = [0] * (len(counts) + k)
            for s, c in enumerate(counts):
                new[s] += c
                new[s + k] += c
            counts = new
        tail = sum(counts[:int(w) + 1])
        return w, min(1.0, 2 * tail / 2 ** n)
    expect = n * (n + 1) / 4
    ties = {}
    for v in r:
        ties[v] = ties.get(v, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
    z = (w - expect) / math.sqrt(var)
    return w, min(1.0, 2 * normal_cdf(z))


def sign_flip(a, b, rounds=100000, seed=0):
    """Exact (for up to 20 pairs) or sampled permutation test on the mean paired difference."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    observed = abs(mean(d))
    if n <= 20:
        hits = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            hits += abs(math.fsum(s * v for s, v in zip(signs, d))) / n >= observed - 1e-12
        return hits / total
    rng = random.Random(seed)
    hits = sum(abs(math.fsum(v if rng.random() < 0.5 else -v for v in d)) / n >= observed - 1e-12 for _ in range(rounds))
    return (hits + 1) / (rounds + 1)


def bootstrap_ci(x, stat=mean, rounds=10000, level=0.95, seed=0):
    """Percentile bootstrap interval for any statistic of one sample."""
    rng = random.Random(seed)
    n = len(x)
    vals = sorted(stat([x[rng.randrange(n)] for _ in range(n)]) for _ in range(rounds))
    a = (1 - level) / 2
    return vals[int(a * rounds)], vals[min(rounds - 1, int((1 - a) * rounds))]


def cohen_d_paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    sd = std(d)
    return 0.0 if sd == 0 else mean(d) / sd


def hedges_g(a, b):
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * variance(a) + (nb - 1) * variance(b)) / (na + nb - 2))
    g = (mean(a) - mean(b)) / pooled
    return g * (1 - 3 / (4 * (na + nb) - 9))


def holm(pvalues):
    """Holm's step-down adjustment. Controls the family-wise error, and beats Bonferroni."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    out = [0.0] * m
    best = 0.0
    for rank, i in enumerate(order):
        best = max(best, min(1.0, (m - rank) * pvalues[i]))
        out[i] = best
    return out


def bonferroni(pvalues):
    return [min(1.0, p * len(pvalues)) for p in pvalues]


def benjamini_hochberg(pvalues):
    """Adjusted p-values that control the false discovery rate."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i], reverse=True)
    m = len(pvalues)
    out = [0.0] * m
    best = 1.0
    for k, i in enumerate(order):
        best = min(best, pvalues[i] * m / (m - k))
        out[i] = best
    return out


def friedman(table):
    """Friedman test. table: rows are datasets or seeds, columns are methods. Returns (statistic, p)."""
    n, k = len(table), len(table[0])
    rank_sum = [0.0] * k
    tie = 0.0
    for row in table:
        r = _ranks(row)
        for j, v in enumerate(r):
            rank_sum[j] += v
        counts = {}
        for v in r:
            counts[v] = counts.get(v, 0) + 1
        tie += sum(c ** 3 - c for c in counts.values())
    stat = 12 / (n * k * (k + 1)) * math.fsum(s * s for s in rank_sum) - 3 * n * (k + 1)
    correction = 1 - tie / (n * k * (k * k - 1))
    if correction == 0:
        return 0.0, 1.0
    stat /= correction
    return stat, chi2_sf(stat, k - 1)
