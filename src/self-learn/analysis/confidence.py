"""Measurement confidence and uncertainty quantification.

Provides tools to assess measurement reliability, determine when
measurements have sufficient precision, and quantify uncertainty
in microscopy quantification tasks.

Functions:
    measurement_cv          -- Coefficient of variation from replicate measurements
    recount_variability     -- Measure counting variability across methods/parameters
    sufficient_samples      -- Check if enough samples for target precision
    propagate_uncertainty   -- Error propagation for derived quantities
    measurement_summary     -- One-call summary of measurement reliability
"""

import numpy as np


def measurement_cv(values):
    """Compute coefficient of variation for replicate measurements.

    CV < 5% = excellent reproducibility
    CV 5-10% = good
    CV 10-20% = acceptable
    CV > 20% = poor, consider more replicates

    Args:
        values: array-like of replicate measurements.

    Returns:
        dict with:
            cv: float, coefficient of variation (0-1).
            cv_percent: float, CV as percentage.
            mean: float.
            std: float.
            n: int, number of replicates.
            quality: str, 'excellent'/'good'/'acceptable'/'poor'.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 2:
        return {
            "cv": 0.0,
            "cv_percent": 0.0,
            "mean": float(values[0]) if n == 1 else 0.0,
            "std": 0.0,
            "n": n,
            "quality": "insufficient",
        }

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

    cv_pct = cv * 100
    if cv_pct < 5:
        quality = "excellent"
    elif cv_pct < 10:
        quality = "good"
    elif cv_pct < 20:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "cv": round(cv, 4),
        "cv_percent": round(cv_pct, 2),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "n": n,
        "quality": quality,
    }


def recount_variability(counts, method_names=None):
    """Assess counting variability across methods or parameter choices.

    Useful when you've counted cells with different thresholds,
    detection methods, or magnifications and want to know how
    consistent the results are.

    Args:
        counts: list of int/float, counts from different methods.
        method_names: list of str or None.

    Returns:
        dict with:
            consensus: float, median count.
            range: float, max - min.
            cv: float, coefficient of variation.
            spread: float, range / median (relative spread).
            methods: list of dicts with method, count, deviation_from_median.
            agreement: str, 'strong'/'moderate'/'weak'.
    """
    counts = np.asarray(counts, dtype=float)
    n = len(counts)

    if method_names is None:
        method_names = [f"method_{i}" for i in range(n)]

    median_count = float(np.median(counts))
    count_range = float(counts.max() - counts.min())
    mean_count = float(counts.mean())
    std_count = float(counts.std(ddof=1)) if n > 1 else 0.0
    cv = std_count / abs(mean_count) if abs(mean_count) > 1e-10 else 0.0
    spread = count_range / abs(median_count) if abs(median_count) > 1e-10 else 0.0

    methods = []
    for _i, (name, count) in enumerate(zip(method_names, counts, strict=False)):
        dev = float(count - median_count)
        methods.append(
            {
                "method": name,
                "count": float(count),
                "deviation_from_median": round(dev, 2),
            }
        )

    if spread < 0.1:
        agreement = "strong"
    elif spread < 0.3:
        agreement = "moderate"
    else:
        agreement = "weak"

    return {
        "consensus": round(median_count, 2),
        "range": round(count_range, 2),
        "cv": round(cv, 4),
        "spread": round(spread, 4),
        "methods": methods,
        "agreement": agreement,
    }


def sufficient_samples(values, target_precision=0.05, confidence=0.95):
    """Check if enough samples for target precision.

    Uses the standard error to determine if the current sample
    size provides sufficient precision for the mean estimate.

    Args:
        values: array-like of measurements.
        target_precision: float, desired relative precision (0.05 = 5%).
        confidence: float, confidence level (default 0.95).

    Returns:
        dict with:
            sufficient: bool, True if precision target is met.
            current_precision: float, current relative precision.
            n_current: int.
            n_needed: int, estimated samples needed for target precision.
            se: float, standard error.
            ci_lower: float, lower bound of confidence interval.
            ci_upper: float, upper bound of confidence interval.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 2:
        return {
            "sufficient": False,
            "current_precision": float("inf"),
            "n_current": n,
            "n_needed": 100,
            "se": 0.0,
            "ci_lower": float(values[0]) if n == 1 else 0.0,
            "ci_upper": float(values[0]) if n == 1 else 0.0,
        }

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    se = std_val / np.sqrt(n)

    # z-score for confidence level
    from scipy import stats

    z = stats.norm.ppf((1 + confidence) / 2)

    current_precision = z * se / abs(mean_val) if abs(mean_val) > 1e-10 else float("inf")

    # Estimate needed n: n = (z * std / (precision * mean))^2
    if abs(mean_val) > 1e-10 and target_precision > 0:
        n_needed = int(np.ceil((z * std_val / (target_precision * abs(mean_val))) ** 2))
        n_needed = max(n_needed, 2)
    else:
        n_needed = 100

    ci_lower = mean_val - z * se
    ci_upper = mean_val + z * se

    return {
        "sufficient": current_precision <= target_precision,
        "current_precision": round(current_precision, 4),
        "n_current": n,
        "n_needed": n_needed,
        "se": round(se, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def propagate_uncertainty(func, values, uncertainties):
    """Error propagation for derived quantities.

    Uses first-order Taylor expansion (linear propagation) to
    estimate uncertainty in a derived quantity from input uncertainties.

    Args:
        func: callable, function of N variables.
        values: list/array of input values.
        uncertainties: list/array of input uncertainties (std dev).

    Returns:
        dict with:
            result: float, func(*values).
            uncertainty: float, propagated uncertainty.
            relative_uncertainty: float, uncertainty / |result|.
            contributions: list of float, fractional contribution
                of each input to total uncertainty.
    """
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    result = float(func(*values))

    # Numerical partial derivatives
    partials = np.zeros(len(values))
    for i in range(len(values)):
        h = max(abs(values[i]) * 1e-6, 1e-10)
        vals_plus = values.copy()
        vals_plus[i] += h
        vals_minus = values.copy()
        vals_minus[i] -= h
        partials[i] = (func(*vals_plus) - func(*vals_minus)) / (2 * h)

    # Propagated uncertainty (assuming independent)
    variance = np.sum((partials * uncertainties) ** 2)
    uncertainty = float(np.sqrt(variance))

    rel_unc = uncertainty / abs(result) if abs(result) > 1e-10 else 0.0

    # Contributions
    contributions = []
    for i in range(len(values)):
        contrib = (partials[i] * uncertainties[i]) ** 2 / variance if variance > 0 else 0.0
        contributions.append(round(float(contrib), 4))

    return {
        "result": round(result, 6),
        "uncertainty": round(uncertainty, 6),
        "relative_uncertainty": round(rel_unc, 4),
        "contributions": contributions,
    }


def measurement_summary(values, label="measurement", target_cv=10.0):
    """One-call summary of measurement reliability.

    Args:
        values: array-like of measurements.
        label: str, name of the measurement.
        target_cv: float, acceptable CV in percent.

    Returns:
        dict with:
            label: str.
            n: int.
            mean: float.
            std: float.
            se: float.
            cv_percent: float.
            ci_95: tuple (lower, upper).
            quality: str.
            recommendation: str, what to do next.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        return {
            "label": label,
            "n": 0,
            "mean": 0,
            "std": 0,
            "se": 0,
            "cv_percent": 0,
            "ci_95": (0, 0),
            "quality": "no data",
            "recommendation": "Acquire measurements.",
        }

    cv_result = measurement_cv(values)
    samp_result = sufficient_samples(values)

    if n < 3:
        rec = f"Only {n} measurements. Need at least 3 for reliability."
    elif cv_result["cv_percent"] > target_cv:
        rec = (
            f"CV={cv_result['cv_percent']:.1f}% exceeds target {target_cv}%. "
            f"Need ~{samp_result['n_needed']} samples or reduce measurement noise."
        )
    elif not samp_result["sufficient"]:
        rec = (
            f"Precision {samp_result['current_precision']*100:.1f}% not yet at 5% target. "
            f"Need ~{samp_result['n_needed']} total samples."
        )
    else:
        rec = f'Measurement is reliable (CV={cv_result["cv_percent"]:.1f}%).'

    return {
        "label": label,
        "n": n,
        "mean": cv_result["mean"],
        "std": cv_result["std"],
        "se": samp_result["se"],
        "cv_percent": cv_result["cv_percent"],
        "ci_95": (samp_result["ci_lower"], samp_result["ci_upper"]),
        "quality": cv_result["quality"],
        "recommendation": rec,
    }
