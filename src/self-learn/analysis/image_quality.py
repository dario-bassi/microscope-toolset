"""Image quality assessment for microscopy.

Evaluates focus, noise, saturation, and overall image quality
to decide whether an image is suitable for analysis.

Functions:
    assess_quality  -- Comprehensive quality assessment
    focus_score     -- Measure image sharpness/focus
    noise_estimate  -- Estimate image noise level
    check_saturation -- Check for over/under-saturated pixels
    dynamic_range   -- Measure useful dynamic range
"""

import numpy as np
from scipy import ndimage


def assess_quality(image, bit_depth=None):
    """Comprehensive image quality assessment.

    Args:
        image: 2D grayscale image.
        bit_depth: Expected bit depth (8, 12, 16). Auto-detected if None.

    Returns:
        dict with:
            overall: 'good', 'acceptable', or 'poor'.
            focus_score: Laplacian variance (higher = sharper).
            noise_level: Estimated noise standard deviation.
            saturation_fraction: Fraction of saturated pixels.
            dynamic_range_fraction: Used fraction of available range.
            snr_estimate: Rough signal-to-noise ratio.
            warnings: List of quality issue strings.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    if bit_depth is None:
        bit_depth = _detect_bit_depth(img)
    max_val = 2**bit_depth - 1

    focus = focus_score(img)
    noise = noise_estimate(img)
    sat = check_saturation(img, max_value=max_val)
    dr = dynamic_range(img, max_value=max_val)

    # SNR estimate
    signal = img.mean()
    snr = signal / max(noise, 1e-10)

    # Assess overall quality
    warnings = []
    if focus < 10:
        warnings.append("very_blurry")
    elif focus < 50:
        warnings.append("slightly_blurry")

    if sat["fraction_saturated"] > 0.01:
        warnings.append(f"saturated ({sat['fraction_saturated']:.1%})")

    if sat["fraction_zero"] > 0.5:
        warnings.append(f"mostly_black ({sat['fraction_zero']:.1%})")

    if noise > 30:
        warnings.append(f"high_noise (std={noise:.1f})")

    if dr < 0.1:
        warnings.append("low_dynamic_range")

    if snr < 3:
        warnings.append(f"low_snr ({snr:.1f})")

    # Overall rating
    if len(warnings) == 0:
        overall = "good"
    elif all("slightly" in w or w.startswith("low_snr") for w in warnings):
        overall = "acceptable"
    else:
        overall = "poor"

    return {
        "overall": overall,
        "focus_score": round(focus, 2),
        "noise_level": round(noise, 3),
        "saturation_fraction": round(sat["fraction_saturated"], 4),
        "dynamic_range_fraction": round(dr, 4),
        "snr_estimate": round(snr, 2),
        "warnings": warnings,
    }


def focus_score(image, method="laplacian"):
    """Measure image sharpness using edge-based metrics.

    Args:
        image: 2D grayscale image.
        method: 'laplacian' (variance of Laplacian),
                'gradient' (mean gradient magnitude),
                'tenengrad' (sum of squared Sobel gradients).

    Returns:
        float: Focus metric (higher = sharper image).
    """
    img = np.asarray(image, dtype=np.float64)

    if method == "laplacian":
        lap = ndimage.laplace(img)
        return float(np.var(lap))

    elif method == "gradient":
        gy = ndimage.sobel(img, axis=0)
        gx = ndimage.sobel(img, axis=1)
        return float(np.mean(np.sqrt(gy**2 + gx**2)))

    elif method == "tenengrad":
        gy = ndimage.sobel(img, axis=0)
        gx = ndimage.sobel(img, axis=1)
        return float(np.mean(gy**2 + gx**2))

    else:
        raise ValueError(
            f"Unknown method: {method}. " f"Use 'laplacian', 'gradient', or 'tenengrad'."
        )


def noise_estimate(image, method="mad"):
    """Estimate image noise level.

    Args:
        image: 2D grayscale image.
        method: 'mad' (median absolute deviation of high-freq),
                'std' (standard deviation of smoothed residual).

    Returns:
        float: Estimated noise standard deviation.
    """
    img = np.asarray(image, dtype=np.float64)

    if method == "mad":
        # Use Laplacian to isolate high-frequency noise
        lap = ndimage.laplace(img)
        mad = float(np.median(np.abs(lap - np.median(lap))))
        # MAD to std conversion for normal distribution
        return mad * 1.4826 / np.sqrt(20)  # Laplacian has ~20x variance

    elif method == "std":
        # Residual after smoothing
        smoothed = ndimage.gaussian_filter(img, sigma=2)
        residual = img - smoothed
        return float(np.std(residual))

    else:
        raise ValueError(f"Unknown method: {method}.")


def check_saturation(image, max_value=None, low_threshold=0, high_fraction=0.995):
    """Check for over/under-saturated pixels.

    Args:
        image: 2D image.
        max_value: Maximum possible value. Auto-detected if None.
        low_threshold: Values at or below this are considered black.
        high_fraction: Fraction of max_value considered saturated.

    Returns:
        dict with:
            fraction_saturated: Fraction of pixels at max value.
            fraction_zero: Fraction of pixels at minimum/zero.
            n_saturated: Count of saturated pixels.
            n_zero: Count of zero/dark pixels.
    """
    img = np.asarray(image, dtype=np.float64)

    if max_value is None:
        max_value = _detect_max(img)

    high_thresh = max_value * high_fraction
    n_total = img.size

    n_saturated = int(np.sum(img >= high_thresh))
    n_zero = int(np.sum(img <= low_threshold))

    return {
        "fraction_saturated": n_saturated / max(n_total, 1),
        "fraction_zero": n_zero / max(n_total, 1),
        "n_saturated": n_saturated,
        "n_zero": n_zero,
    }


def dynamic_range(image, max_value=None, pmin=1, pmax=99):
    """Measure the useful dynamic range of an image.

    Args:
        image: 2D image.
        max_value: Maximum possible value.
        pmin, pmax: Percentiles for range estimation.

    Returns:
        float: Fraction of available range used (0-1).
    """
    img = np.asarray(image, dtype=np.float64)

    if max_value is None:
        max_value = _detect_max(img)

    if max_value <= 0:
        return 0.0

    lo = float(np.percentile(img, pmin))
    hi = float(np.percentile(img, pmax))

    return (hi - lo) / max_value


def _detect_bit_depth(img):
    """Auto-detect image bit depth from data range."""
    mx = img.max()
    if mx <= 1.0 and img.dtype in (np.float32, np.float64):
        return 8  # assume normalized float
    elif mx <= 255:
        return 8
    elif mx <= 4095:
        return 12
    else:
        return 16


def _detect_max(img):
    """Auto-detect maximum possible value."""
    bd = _detect_bit_depth(img)
    return float(2**bd - 1)
