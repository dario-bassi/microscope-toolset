"""Cell confluency measurement.

Quantifies cell coverage as fraction of field occupied by cells.
Useful for growth curves, cytotoxicity, and monolayer integrity assessment.

Functions:
    measure_confluency   -- Compute confluency from a single image
    confluency_timecourse -- Track confluency over multiple frames
    growth_curve          -- Fit growth model to confluency data
    doubling_time         -- Estimate population doubling time

Real microscope note:
    Binary thresholding assumes clean brightfield foreground/background
    separation. Real cells have halos, shadows, and uneven illumination
    that violate this assumption. For better results on live images:
    - Add morphological preprocessing (erosion/dilation) to clean edges
    - Use adaptive thresholding with larger block sizes
    - Validate threshold direction on sample images first
    - Consider illumination correction (flat-field) if available
"""

import numpy as np
from scipy import ndimage, optimize
from skimage import filters, morphology


def measure_confluency(image, method="otsu", threshold=None, min_cell_area=10, return_mask=False):
    """Measure cell confluency (fraction of FOV covered by cells).

    Args:
        image: 2D array, brightfield or fluorescence image.
        method: str, thresholding method:
            'otsu' — Otsu automatic threshold.
            'adaptive' — local adaptive threshold.
            'manual' — use provided threshold value.
        threshold: float or None, manual threshold (for method='manual').
        min_cell_area: int, minimum object size in pixels.
        return_mask: bool, if True also return the binary mask.

    Returns:
        dict with:
            confluency: float, fraction covered (0-1).
            confluency_percent: float, percentage covered.
            cell_area_px: int, total cell area in pixels.
            total_area_px: int, total image area.
            n_objects: int, number of cell clusters.
            mask: 2D bool array (only if return_mask=True).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    # Determine if cells are bright or dark
    # Use edge intensity vs center intensity
    h, w = image.shape
    edge_vals = np.concatenate(
        [
            image[0, :],
            image[-1, :],
            image[:, 0],
            image[:, -1],
        ]
    )
    center = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    cells_are_bright = np.median(center) > np.median(edge_vals)

    # Threshold
    smoothed = ndimage.gaussian_filter(image, sigma=2)

    if method == "otsu":
        thresh = filters.threshold_otsu(smoothed)
    elif method == "adaptive":
        block_size = max(11, min(h, w) // 8 | 1)
        thresh_img = filters.threshold_local(smoothed, block_size)
        mask = smoothed > thresh_img if cells_are_bright else smoothed < thresh_img
    elif method == "manual":
        if threshold is None:
            raise ValueError("Must provide threshold for method='manual'")
        thresh = threshold
    else:
        raise ValueError(f"Unknown method: {method}")

    if method != "adaptive":
        if cells_are_bright:
            mask = smoothed > thresh
        else:
            mask = smoothed < thresh

    # Clean mask
    mask = ndimage.binary_fill_holes(mask)
    if mask.any() and min_cell_area > 1:
        mask = morphology.remove_small_objects(mask, max_size=min_cell_area - 1)

    cell_area = int(mask.sum())
    total_area = mask.size
    confluency = cell_area / total_area

    # Count objects
    labeled = ndimage.label(mask)[0]
    n_objects = labeled.max()

    result = {
        "confluency": round(confluency, 4),
        "confluency_percent": round(confluency * 100, 2),
        "cell_area_px": cell_area,
        "total_area_px": total_area,
        "n_objects": n_objects,
    }
    if return_mask:
        result["mask"] = mask

    return result


def confluency_timecourse(images, method="otsu", min_cell_area=10):
    """Track confluency over multiple frames.

    Args:
        images: list or 3D array (T, H, W) of timelapse frames.
        method: str, thresholding method.
        min_cell_area: int, minimum object size.

    Returns:
        dict with:
            confluencies: list of float, confluency at each frame.
            n_objects: list of int, object count at each frame.
            initial: float, first frame confluency.
            final: float, last frame confluency.
            change: float, final - initial.
            max_confluency: float, peak confluency observed.
    """
    images = np.asarray(images)
    if images.ndim == 4:
        # RGB stack
        images = np.mean(images, axis=3)

    confluencies = []
    n_objects_list = []

    for img in images:
        result = measure_confluency(img, method=method, min_cell_area=min_cell_area)
        confluencies.append(result["confluency"])
        n_objects_list.append(result["n_objects"])

    initial = confluencies[0] if confluencies else 0
    final = confluencies[-1] if confluencies else 0

    return {
        "confluencies": confluencies,
        "n_objects": n_objects_list,
        "initial": round(initial, 4),
        "final": round(final, 4),
        "change": round(final - initial, 4),
        "max_confluency": round(max(confluencies) if confluencies else 0, 4),
    }


def growth_curve(timepoints, confluencies, model="logistic"):
    """Fit growth model to confluency data.

    Args:
        timepoints: array-like, time values.
        confluencies: array-like, confluency fractions (0-1).
        model: str, growth model:
            'logistic' — S-shaped growth: C(t) = K / (1 + exp(-r*(t-t0)))
            'exponential' — C(t) = C0 * exp(r*t)
            'linear' — C(t) = C0 + r*t

    Returns:
        dict with:
            model: str, model name.
            params: dict, fitted parameters.
            r_squared: float, goodness of fit.
            predicted: array, model predictions.
            growth_rate: float, growth rate parameter.
    """
    t = np.asarray(timepoints, dtype=float)
    c = np.asarray(confluencies, dtype=float)

    if len(t) < 3:
        return {
            "model": model,
            "params": {},
            "r_squared": 0.0,
            "predicted": c.copy(),
            "growth_rate": 0.0,
        }

    if model == "logistic":

        def logistic(t, K, r, t0):
            return K / (1 + np.exp(-r * (t - t0)))

        try:
            p0 = [max(c), 0.1, t[len(t) // 2]]
            bounds = ([0, 0, t[0] - 100], [1.5, 10, t[-1] + 100])
            popt, _ = optimize.curve_fit(logistic, t, c, p0=p0, bounds=bounds, maxfev=5000)
            predicted = logistic(t, *popt)
            params = {"K": popt[0], "r": popt[1], "t0": popt[2]}
            rate = popt[1]
        except (RuntimeError, ValueError):
            predicted = np.full_like(c, np.mean(c))
            params = {}
            rate = 0.0

    elif model == "exponential":

        def exp_growth(t, C0, r):
            return C0 * np.exp(r * t)

        try:
            c_pos = np.maximum(c, 1e-6)
            p0 = [c_pos[0], 0.01]
            popt, _ = optimize.curve_fit(exp_growth, t, c_pos, p0=p0, maxfev=5000)
            predicted = exp_growth(t, *popt)
            params = {"C0": popt[0], "r": popt[1]}
            rate = popt[1]
        except (RuntimeError, ValueError):
            predicted = np.full_like(c, np.mean(c))
            params = {}
            rate = 0.0

    elif model == "linear":
        coeffs = np.polyfit(t, c, 1)
        predicted = np.polyval(coeffs, t)
        params = {"slope": coeffs[0], "intercept": coeffs[1]}
        rate = coeffs[0]

    else:
        raise ValueError(f"Unknown model: {model}")

    # R-squared
    ss_res = np.sum((c - predicted) ** 2)
    ss_tot = np.sum((c - np.mean(c)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "model": model,
        "params": {k: round(v, 6) for k, v in params.items()},
        "r_squared": round(r_squared, 4),
        "predicted": predicted,
        "growth_rate": round(float(rate), 6),
    }


def doubling_time(timepoints, values, method="regression"):
    """Estimate population doubling time from growth data.

    Args:
        timepoints: array-like, time values.
        values: array-like, population measure (count, confluency, OD).
        method: str:
            'regression' — log-linear regression on exponential phase.
            'direct' — find time interval where value doubles.

    Returns:
        dict with:
            doubling_time: float, estimated doubling time in same
                units as timepoints. None if can't be determined.
            growth_rate: float, exponential growth rate constant.
            method: str.
            r_squared: float (for regression method).
    """
    t = np.asarray(timepoints, dtype=float)
    v = np.asarray(values, dtype=float)

    if len(t) < 2 or np.all(v <= 0):
        return {
            "doubling_time": None,
            "growth_rate": 0.0,
            "method": method,
            "r_squared": 0.0,
        }

    if method == "regression":
        # Log-linear regression
        pos = v > 0
        if pos.sum() < 2:
            return {
                "doubling_time": None,
                "growth_rate": 0.0,
                "method": method,
                "r_squared": 0.0,
            }
        log_v = np.log(v[pos])
        t_pos = t[pos]
        coeffs = np.polyfit(t_pos, log_v, 1)
        rate = float(coeffs[0])

        # R-squared
        predicted = np.polyval(coeffs, t_pos)
        ss_res = np.sum((log_v - predicted) ** 2)
        ss_tot = np.sum((log_v - np.mean(log_v)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        dt = np.log(2) / rate if rate > 0 else None

        return {
            "doubling_time": round(dt, 4) if dt is not None else None,
            "growth_rate": round(rate, 6),
            "method": method,
            "r_squared": round(r2, 4),
        }

    elif method == "direct":
        # Find first time the value doubles from initial
        v0 = v[0]
        target = v0 * 2
        above = np.where(v >= target)[0]
        if len(above) == 0:
            return {
                "doubling_time": None,
                "growth_rate": 0.0,
                "method": method,
                "r_squared": 0.0,
            }
        idx = above[0]
        if idx > 0:
            # Linear interpolation
            t0, t1 = t[idx - 1], t[idx]
            v0_i, v1_i = v[idx - 1], v[idx]
            if v1_i != v0_i:
                dt = t0 + (target - v0_i) / (v1_i - v0_i) * (t1 - t0) - t[0]
            else:
                dt = t1 - t[0]
        else:
            dt = 0.0

        rate = np.log(2) / dt if dt > 0 else 0.0

        return {
            "doubling_time": round(float(dt), 4) if dt > 0 else None,
            "growth_rate": round(float(rate), 6),
            "method": method,
            "r_squared": 0.0,
        }

    else:
        raise ValueError(f"Unknown method: {method}")
