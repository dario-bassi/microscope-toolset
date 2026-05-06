"""Particle/cell size distribution analysis.

Analyzes population size distributions: histogram statistics,
percentile measurements, modality detection, and distribution
fitting.

Functions:
    size_stats          -- Basic statistics for a size distribution
    size_percentiles    -- Standard percentiles (D10, D50, D90, etc.)
    fit_distribution    -- Fit normal/lognormal to size data
    detect_subpopulations -- Find modes in multimodal distributions
    size_filter         -- Filter objects by size criteria
"""

import numpy as np
from scipy import stats


def size_stats(sizes, unit="px"):
    """Compute basic statistics for a size distribution.

    Args:
        sizes: array-like, measured sizes (area, diameter, etc.).
        unit: str, unit label for reporting.

    Returns:
        dict with:
            n: int, number of measurements.
            mean: float.
            median: float.
            std: float.
            cv: float, coefficient of variation.
            min: float.
            max: float.
            range: float.
            iqr: float, interquartile range.
            skewness: float.
            unit: str.
    """
    sizes = np.asarray(sizes, dtype=float)
    if len(sizes) == 0:
        return {
            "n": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "cv": 0,
            "min": 0,
            "max": 0,
            "range": 0,
            "iqr": 0,
            "skewness": 0,
            "unit": unit,
        }

    q25 = float(np.percentile(sizes, 25))
    q75 = float(np.percentile(sizes, 75))
    mean_val = float(sizes.mean())

    return {
        "n": len(sizes),
        "mean": round(mean_val, 4),
        "median": round(float(np.median(sizes)), 4),
        "std": round(float(sizes.std()), 4),
        "cv": round(float(sizes.std() / mean_val), 4) if mean_val > 0 else 0,
        "min": round(float(sizes.min()), 4),
        "max": round(float(sizes.max()), 4),
        "range": round(float(sizes.max() - sizes.min()), 4),
        "iqr": round(q75 - q25, 4),
        "skewness": round(float(stats.skew(sizes)), 4) if sizes.std() > 0 else 0.0,
        "unit": unit,
    }


def size_percentiles(sizes, percentiles=None):
    """Compute standard percentiles for size distribution.

    Args:
        sizes: array-like, measured sizes.
        percentiles: list of float, percentiles to compute.
            Default: [10, 25, 50, 75, 90, 95, 99].

    Returns:
        dict with:
            D{p}: float for each percentile p.
            span: float, (D90 - D10) / D50 — distribution width.
    """
    sizes = np.asarray(sizes, dtype=float)
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]

    if len(sizes) == 0:
        result = {f"D{p}": 0.0 for p in percentiles}
        result["span"] = 0.0
        return result

    result = {}
    for p in percentiles:
        result[f"D{p}"] = round(float(np.percentile(sizes, p)), 4)

    # Span = (D90 - D10) / D50
    d10 = result.get("D10", np.percentile(sizes, 10))
    d50 = result.get("D50", np.percentile(sizes, 50))
    d90 = result.get("D90", np.percentile(sizes, 90))
    result["span"] = round((d90 - d10) / d50, 4) if d50 > 0 else 0.0

    return result


def fit_distribution(sizes, model="lognormal"):
    """Fit a parametric distribution to size data.

    Args:
        sizes: array-like, measured sizes (must be positive).
        model: str, distribution model:
            'normal' — Gaussian distribution.
            'lognormal' — Log-normal distribution.

    Returns:
        dict with:
            model: str.
            params: dict of fitted parameters.
            ks_statistic: float, Kolmogorov-Smirnov test statistic.
            p_value: float, KS test p-value.
            aic: float, Akaike Information Criterion.
    """
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes > 0]

    if len(sizes) < 3:
        return {
            "model": model,
            "params": {},
            "ks_statistic": 1.0,
            "p_value": 0.0,
            "aic": np.inf,
        }

    if model == "normal":
        mu, sigma = float(sizes.mean()), float(sizes.std())
        ks_stat, p_val = stats.kstest(sizes, "norm", args=(mu, sigma))
        # AIC = 2k - 2ln(L)
        log_l = np.sum(stats.norm.logpdf(sizes, mu, sigma))
        aic = 4 - 2 * log_l
        params = {"mu": round(mu, 4), "sigma": round(sigma, 4)}

    elif model == "lognormal":
        log_sizes = np.log(sizes)
        mu_log = float(log_sizes.mean())
        sigma_log = float(log_sizes.std())
        # scipy lognorm: shape=sigma, scale=exp(mu)
        shape = sigma_log
        scale = np.exp(mu_log)
        ks_stat, p_val = stats.kstest(sizes, "lognorm", args=(shape, 0, scale))
        log_l = np.sum(stats.lognorm.logpdf(sizes, shape, 0, scale))
        aic = 4 - 2 * log_l
        params = {
            "mu_log": round(mu_log, 4),
            "sigma_log": round(sigma_log, 4),
            "geometric_mean": round(float(np.exp(mu_log)), 4),
        }

    else:
        raise ValueError(f"Unknown model: {model}")

    return {
        "model": model,
        "params": params,
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 4),
        "aic": round(float(aic), 2),
    }


def detect_subpopulations(sizes, n_bins=50, min_separation=0.2):
    """Detect subpopulations (modes) in a size distribution.

    Uses kernel density estimation to find peaks.

    Args:
        sizes: array-like, measured sizes.
        n_bins: int, number of bins for histogram.
        min_separation: float, minimum relative separation between
            peaks (fraction of range) to count as distinct.

    Returns:
        dict with:
            n_modes: int, number of detected modes.
            modes: list of float, mode values.
            mode_counts: list of int, approximate count per mode.
            is_multimodal: bool.
    """
    sizes = np.asarray(sizes, dtype=float)

    if len(sizes) < 5:
        return {
            "n_modes": 1 if len(sizes) > 0 else 0,
            "modes": [float(np.median(sizes))] if len(sizes) > 0 else [],
            "mode_counts": [len(sizes)] if len(sizes) > 0 else [],
            "is_multimodal": False,
        }

    # Histogram-based peak detection
    counts, edges = np.histogram(sizes, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth histogram
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(counts.astype(float), sigma=2)

    # Find peaks: points higher than both neighbors
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] > smoothed.max() * 0.1:  # >10% of max
                peaks.append(i)

    if not peaks:
        peaks = [np.argmax(smoothed)]

    modes = [round(float(centers[p]), 4) for p in peaks]
    mode_counts = [int(smoothed[p]) for p in peaks]

    # Check if modes are sufficiently separated
    data_range = sizes.max() - sizes.min()
    if data_range > 0 and len(modes) > 1:
        filtered_modes = [modes[0]]
        filtered_counts = [mode_counts[0]]
        for m, c in zip(modes[1:], mode_counts[1:], strict=False):
            if abs(m - filtered_modes[-1]) / data_range > min_separation:
                filtered_modes.append(m)
                filtered_counts.append(c)
        modes = filtered_modes
        mode_counts = filtered_counts

    return {
        "n_modes": len(modes),
        "modes": modes,
        "mode_counts": mode_counts,
        "is_multimodal": len(modes) > 1,
    }


def size_filter(sizes, labels=None, min_size=None, max_size=None, percentile_range=None):
    """Filter objects by size criteria.

    Args:
        sizes: array-like, measured sizes.
        labels: array-like or None, object labels corresponding to sizes.
        min_size: float or None, minimum size threshold.
        max_size: float or None, maximum size threshold.
        percentile_range: tuple (low, high) or None, keep objects
            within this percentile range.

    Returns:
        dict with:
            mask: 1D bool array, True for kept objects.
            n_kept: int.
            n_removed: int.
            kept_sizes: array of sizes that passed filter.
            kept_labels: array of labels that passed (if provided).
    """
    sizes = np.asarray(sizes, dtype=float)
    mask = np.ones(len(sizes), dtype=bool)

    if min_size is not None:
        mask &= sizes >= min_size
    if max_size is not None:
        mask &= sizes <= max_size
    if percentile_range is not None:
        lo, hi = percentile_range
        plo = np.percentile(sizes, lo)
        phi = np.percentile(sizes, hi)
        mask &= (sizes >= plo) & (sizes <= phi)

    result = {
        "mask": mask,
        "n_kept": int(mask.sum()),
        "n_removed": int((~mask).sum()),
        "kept_sizes": sizes[mask],
    }

    if labels is not None:
        labels = np.asarray(labels)
        result["kept_labels"] = labels[mask]

    return result
