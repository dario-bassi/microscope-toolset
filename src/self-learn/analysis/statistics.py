"""Statistical testing for microscopy experiments.

Hypothesis tests and confidence intervals for comparing populations,
conditions, and time points in microscopy experiments.

Functions:
    compare_two          -- Compare two populations (t-test or Mann-Whitney)
    compare_multiple     -- Compare multiple groups (ANOVA or Kruskal-Wallis)
    pairwise_comparisons -- Post-hoc pairwise tests with correction
    correct_pvalues      -- Multiple testing correction (Bonferroni/FDR/Holm)
    confidence_interval  -- Compute confidence interval for a mean
    effect_size          -- Cohen's d effect size
    bootstrap_ci         -- Bootstrap confidence interval for any statistic
"""

import numpy as np
from scipy import stats as sp_stats


def compare_two(group1, group2, test='auto', alternative='two-sided'):
    """Compare two populations statistically.

    Args:
        group1: 1D array of values.
        group2: 1D array of values.
        test: 'ttest' (parametric), 'mannwhitney' (non-parametric),
              or 'auto' (choose based on normality and sample size).
        alternative: 'two-sided', 'greater', 'less'.

    Returns:
        dict with:
            test_used: Name of the test performed.
            statistic: Test statistic value.
            p_value: p-value.
            significant: Whether p < 0.05.
            mean1, mean2: Group means.
            effect_size: Cohen's d.
            summary: Human-readable summary string.
    """
    g1 = np.asarray(group1, dtype=np.float64).ravel()
    g2 = np.asarray(group2, dtype=np.float64).ravel()

    if len(g1) < 2 or len(g2) < 2:
        return {
            'test_used': 'insufficient_data',
            'statistic': 0.0, 'p_value': 1.0,
            'significant': False,
            'mean1': float(g1.mean()) if len(g1) > 0 else 0.0,
            'mean2': float(g2.mean()) if len(g2) > 0 else 0.0,
            'effect_size': 0.0,
            'summary': 'Insufficient data for statistical test',
        }

    m1, m2 = float(g1.mean()), float(g2.mean())
    d = effect_size(g1, g2)

    if test == 'auto':
        # Use parametric if both groups are roughly normal and n >= 20
        if len(g1) >= 20 and len(g2) >= 20:
            _, p_norm1 = sp_stats.shapiro(g1[:50])  # cap at 50 for speed
            _, p_norm2 = sp_stats.shapiro(g2[:50])
            if p_norm1 > 0.05 and p_norm2 > 0.05:
                test = 'ttest'
            else:
                test = 'mannwhitney'
        elif len(g1) >= 5 and len(g2) >= 5:
            test = 'mannwhitney'
        else:
            test = 'ttest'

    if test == 'ttest':
        stat_result = sp_stats.ttest_ind(g1, g2, alternative=alternative)
        stat_val = float(stat_result.statistic)
        p_val = float(stat_result.pvalue)
        test_name = "Welch's t-test"

    elif test == 'mannwhitney':
        alt_map = {'two-sided': 'two-sided', 'greater': 'greater',
                   'less': 'less'}
        stat_result = sp_stats.mannwhitneyu(
            g1, g2, alternative=alt_map.get(alternative, 'two-sided'))
        stat_val = float(stat_result.statistic)
        p_val = float(stat_result.pvalue)
        test_name = 'Mann-Whitney U'

    else:
        raise ValueError(f"Unknown test: {test}. Use 'ttest', 'mannwhitney', or 'auto'.")

    sig = p_val < 0.05
    direction = "higher" if m1 > m2 else "lower"
    summary = (f"{test_name}: p={p_val:.4f} "
               f"({'significant' if sig else 'not significant'}), "
               f"group1 {direction} (d={d:.2f})")

    return {
        'test_used': test_name,
        'statistic': round(stat_val, 4),
        'p_value': round(p_val, 6),
        'significant': sig,
        'mean1': round(m1, 4),
        'mean2': round(m2, 4),
        'effect_size': round(d, 4),
        'summary': summary,
    }


def compare_multiple(groups, test='auto'):
    """Compare multiple groups statistically.

    Args:
        groups: List of 1D arrays.
        test: 'anova' (parametric), 'kruskal' (non-parametric),
              or 'auto'.

    Returns:
        dict with:
            test_used: Name of test.
            statistic: Test statistic.
            p_value: p-value.
            significant: Whether p < 0.05.
            group_means: List of group means.
            n_groups: Number of groups.
            summary: Human-readable summary.
    """
    arrays = [np.asarray(g, dtype=np.float64).ravel() for g in groups]

    if len(arrays) < 2:
        return {
            'test_used': 'insufficient_groups',
            'statistic': 0.0, 'p_value': 1.0,
            'significant': False,
            'group_means': [float(a.mean()) for a in arrays if len(a) > 0],
            'n_groups': len(arrays),
            'summary': 'Need at least 2 groups',
        }

    means = [float(a.mean()) for a in arrays]

    # Check if all groups have enough data
    if any(len(a) < 2 for a in arrays):
        return {
            'test_used': 'insufficient_data',
            'statistic': 0.0, 'p_value': 1.0,
            'significant': False,
            'group_means': means,
            'n_groups': len(arrays),
            'summary': 'All groups need at least 2 observations',
        }

    if test == 'auto':
        # Kruskal-Wallis if any group has < 20 observations or non-normal
        if all(len(a) >= 20 for a in arrays):
            test = 'anova'
        else:
            test = 'kruskal'

    if test == 'anova':
        stat_result = sp_stats.f_oneway(*arrays)
        stat_val = float(stat_result.statistic)
        p_val = float(stat_result.pvalue)
        test_name = 'One-way ANOVA'

    elif test == 'kruskal':
        stat_result = sp_stats.kruskal(*arrays)
        stat_val = float(stat_result.statistic)
        p_val = float(stat_result.pvalue)
        test_name = 'Kruskal-Wallis'

    else:
        raise ValueError(f"Unknown test: {test}. Use 'anova', 'kruskal', or 'auto'.")

    sig = p_val < 0.05
    summary = (f"{test_name}: p={p_val:.4f} "
               f"({'significant' if sig else 'not significant'}), "
               f"{len(arrays)} groups")

    return {
        'test_used': test_name,
        'statistic': round(stat_val, 4),
        'p_value': round(p_val, 6),
        'significant': sig,
        'group_means': [round(m, 4) for m in means],
        'n_groups': len(arrays),
        'summary': summary,
    }


def confidence_interval(values, confidence=0.95):
    """Compute confidence interval for the mean.

    Args:
        values: 1D array of values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        dict with:
            mean: Sample mean.
            ci_low: Lower bound of CI.
            ci_high: Upper bound of CI.
            margin: Half-width of the CI.
            n: Sample size.
            sem: Standard error of the mean.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = len(arr)

    if n < 2:
        m = float(arr[0]) if n == 1 else 0.0
        return {
            'mean': m, 'ci_low': m, 'ci_high': m,
            'margin': 0.0, 'n': n, 'sem': 0.0,
        }

    m = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * sem

    return {
        'mean': round(m, 4),
        'ci_low': round(m - margin, 4),
        'ci_high': round(m + margin, 4),
        'margin': round(margin, 4),
        'n': n,
        'sem': round(sem, 4),
    }


def effect_size(group1, group2):
    """Compute Cohen's d effect size between two groups.

    Args:
        group1: 1D array.
        group2: 1D array.

    Returns:
        float: Cohen's d (positive if group1 > group2).
    """
    g1 = np.asarray(group1, dtype=np.float64).ravel()
    g2 = np.asarray(group2, dtype=np.float64).ravel()

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)

    # Pooled standard deviation
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if sp < 1e-10:
        return 0.0

    return float((m1 - m2) / sp)


def bootstrap_ci(values, statistic_fn=np.mean, n_bootstrap=1000,
                 confidence=0.95, seed=42):
    """Compute bootstrap confidence interval for any statistic.

    Args:
        values: 1D array of values.
        statistic_fn: Function to compute the statistic (default: mean).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        dict with:
            estimate: Point estimate of the statistic.
            ci_low: Lower bound.
            ci_high: Upper bound.
            bootstrap_std: Standard deviation of bootstrap distribution.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = len(arr)

    if n == 0:
        return {'estimate': 0.0, 'ci_low': 0.0, 'ci_high': 0.0,
                'bootstrap_std': 0.0}

    estimate = float(statistic_fn(arr))

    if n < 2:
        return {'estimate': estimate, 'ci_low': estimate, 'ci_high': estimate,
                'bootstrap_std': 0.0}

    rng = np.random.RandomState(seed)
    boot_stats = np.array([
        float(statistic_fn(rng.choice(arr, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(boot_stats, alpha * 100))
    ci_high = float(np.percentile(boot_stats, (1 - alpha) * 100))

    return {
        'estimate': round(estimate, 4),
        'ci_low': round(ci_low, 4),
        'ci_high': round(ci_high, 4),
        'bootstrap_std': round(float(boot_stats.std()), 4),
    }


# ---------------------------------------------------------------------------
# Post-hoc pairwise comparisons
# ---------------------------------------------------------------------------

def pairwise_comparisons(groups, group_names=None, test='auto',
                         correction='holm'):
    """Post-hoc pairwise comparisons between all group pairs.

    After a significant ANOVA/Kruskal-Wallis, run this to find which
    specific pairs differ. Applies multiple testing correction automatically.

    Args:
        groups: List of 1D arrays (one per group).
        group_names: Optional list of names (default: 'Group 0', 'Group 1', ...).
        test: 'ttest', 'mannwhitney', or 'auto' (same as compare_two).
        correction: 'bonferroni', 'holm', 'fdr', or 'none'.

    Returns:
        dict with:
            comparisons: List of dicts, each with:
                pair: (name_a, name_b) tuple.
                p_value: Raw p-value.
                p_adjusted: Corrected p-value.
                significant: Whether adjusted p < 0.05.
                effect_size: Cohen's d.
                mean_diff: Difference in means.
            n_significant: Number of significant pairs.
            correction_method: Correction method used.
            summary: Human-readable summary string.
    """
    n_groups = len(groups)
    if group_names is None:
        group_names = [f'Group {i}' for i in range(n_groups)]

    comparisons = []
    raw_pvals = []

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            result = compare_two(groups[i], groups[j], test=test)
            comparisons.append({
                'pair': (group_names[i], group_names[j]),
                'p_value': result['p_value'],
                'effect_size': result['effect_size'],
                'mean_diff': round(result['mean1'] - result['mean2'], 4),
            })
            raw_pvals.append(result['p_value'])

    # Apply correction
    if correction == 'none' or len(raw_pvals) == 0:
        adjusted = raw_pvals[:]
    else:
        adjusted = correct_pvalues(raw_pvals, method=correction)

    n_sig = 0
    for idx, comp in enumerate(comparisons):
        comp['p_adjusted'] = round(adjusted[idx], 6)
        comp['significant'] = adjusted[idx] < 0.05
        if comp['significant']:
            n_sig += 1

    sig_pairs = [f"{c['pair'][0]} vs {c['pair'][1]}" for c in comparisons
                 if c['significant']]
    summary = (f"{n_sig}/{len(comparisons)} pairs significant "
               f"({correction} correction)")
    if sig_pairs:
        summary += ": " + ", ".join(sig_pairs[:5])
        if len(sig_pairs) > 5:
            summary += f", ... ({len(sig_pairs) - 5} more)"

    return {
        'comparisons': comparisons,
        'n_significant': n_sig,
        'correction_method': correction,
        'summary': summary,
    }


def estimate_required_n(positives, total, target_ci_width=0.05,
                        confidence=0.95):
    """Estimate required sample size for a proportion measurement.

    Uses the Wilson score confidence interval to compute how many
    observations are needed for the CI half-width to be at most
    ``target_ci_width``. Call after each batch of FOVs to decide
    whether to acquire more.

    Typical use: parasitemia (infected / total RBCs), mitotic index,
    apoptotic fraction, or any binary classification rate.

    Args:
        positives: Number of positive observations so far.
        total: Total observations so far.
        target_ci_width: Target CI half-width (e.g. 0.05 for ±5%).
        confidence: Confidence level (default 0.95).

    Returns:
        dict with:
            proportion: Current observed proportion.
            ci_low: Wilson CI lower bound.
            ci_high: Wilson CI upper bound.
            ci_width: Current CI half-width.
            required_n: Estimated total N for target_ci_width.
            sufficient: Whether current N meets target.
    """
    if total < 1:
        return {
            'proportion': 0.0,
            'ci_low': 0.0,
            'ci_high': 0.0,
            'ci_width': 1.0,
            'required_n': 100,
            'sufficient': False,
        }

    p = positives / total
    z = float(sp_stats.norm.ppf((1 + confidence) / 2))

    # Wilson score interval
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    ci_low = max(0.0, center - spread)
    ci_high = min(1.0, center + spread)
    ci_width = (ci_high - ci_low) / 2

    # Estimate required N: from normal approximation, CI half-width ≈ z*sqrt(p(1-p)/n)
    # Solve for n: n = (z / target_ci_width)^2 * p * (1-p)
    p_est = max(p, 0.01)  # avoid p=0 giving required_n=0
    p_est = min(p_est, 0.99)
    required_n = int(np.ceil((z / target_ci_width)**2 * p_est * (1 - p_est)))
    required_n = max(required_n, 30)  # minimum reasonable sample

    return {
        'proportion': round(p, 4),
        'ci_low': round(ci_low, 4),
        'ci_high': round(ci_high, 4),
        'ci_width': round(ci_width, 4),
        'required_n': required_n,
        'sufficient': total >= required_n,
    }


def correct_pvalues(pvalues, method='holm'):
    """Apply multiple testing correction to a list of p-values.

    Args:
        pvalues: List or 1D array of raw p-values.
        method: Correction method:
            'bonferroni' — Multiply by number of tests (most conservative).
            'holm' — Step-down Holm-Bonferroni (good general-purpose).
            'fdr' — Benjamini-Hochberg FDR control (less conservative).

    Returns:
        list of adjusted p-values (same length as input, capped at 1.0).
    """
    pvals = np.asarray(pvalues, dtype=np.float64)
    n = len(pvals)

    if n == 0:
        return []

    if method == 'bonferroni':
        adjusted = np.minimum(pvals * n, 1.0)
        return [round(float(p), 6) for p in adjusted]

    elif method == 'holm':
        # Step-down: sort ascending, multiply by (n - rank), enforce monotonicity
        order = np.argsort(pvals)
        adjusted = np.zeros(n)
        for rank_idx, orig_idx in enumerate(order):
            adjusted[orig_idx] = pvals[orig_idx] * (n - rank_idx)
        # Enforce monotonicity (step-down): max of current and previous
        sorted_adj = adjusted[order]
        for i in range(1, n):
            sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
        adjusted[order] = sorted_adj
        adjusted = np.minimum(adjusted, 1.0)
        return [round(float(p), 6) for p in adjusted]

    elif method == 'fdr':
        # Benjamini-Hochberg: sort ascending, p * n/rank, enforce monotonicity
        order = np.argsort(pvals)
        adjusted = np.zeros(n)
        for rank_idx, orig_idx in enumerate(order):
            rank = rank_idx + 1
            adjusted[orig_idx] = pvals[orig_idx] * n / rank
        # Enforce monotonicity (step-up): min of current and next
        sorted_adj = adjusted[order]
        for i in range(n - 2, -1, -1):
            sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
        adjusted[order] = sorted_adj
        adjusted = np.minimum(adjusted, 1.0)
        return [round(float(p), 6) for p in adjusted]

    else:
        raise ValueError(
            f"Unknown correction method: {method}. "
            "Use 'bonferroni', 'holm', or 'fdr'."
        )
