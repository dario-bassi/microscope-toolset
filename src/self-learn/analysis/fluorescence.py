"""Quantitative fluorescence analysis.

Background subtraction, illumination correction, ROI intensity
measurement, and photobleaching correction for fluorescence images.

Functions:
    subtract_background  -- Remove background using rolling ball or percentile
    correct_illumination -- Flat-field illumination correction
    measure_roi          -- Measure intensity within masked ROI
    bleach_correct       -- Correct photobleaching in a time series
    normalize_intensity  -- Min-max or percentile normalization
"""

import numpy as np
from scipy import ndimage


def subtract_background(image, method='rolling_ball', radius=50,
                        percentile=5):
    """Subtract background from a fluorescence image.

    Args:
        image: 2D grayscale image.
        method: 'rolling_ball' (morphological opening),
                'percentile' (use low percentile as flat background),
                'median' (large-radius median filter).
        radius: Ball radius for rolling_ball, or filter size for median.
        percentile: Percentile value for percentile method.

    Returns:
        dict with:
            corrected: Background-subtracted image (clipped to >= 0).
            background: Estimated background image.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    if method == 'rolling_ball':
        # Morphological opening approximates rolling ball
        struct = ndimage.generate_binary_structure(2, 1)
        bg = ndimage.grey_opening(img, size=(radius, radius))
        corrected = img - bg
        corrected = np.clip(corrected, 0, None)

    elif method == 'percentile':
        bg_val = float(np.percentile(img, percentile))
        bg = np.full_like(img, bg_val)
        corrected = img - bg_val
        corrected = np.clip(corrected, 0, None)

    elif method == 'median':
        bg = ndimage.median_filter(img, size=radius)
        corrected = img - bg
        corrected = np.clip(corrected, 0, None)

    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Use 'rolling_ball', 'percentile', or 'median'.")

    return {
        'corrected': corrected,
        'background': bg,
    }


def correct_illumination(image, flat_field=None, dark_field=None,
                         sigma=None):
    """Apply flat-field illumination correction.

    Corrects for uneven illumination using:
        corrected = (image - dark) / (flat - dark) * mean(flat - dark)

    If no flat_field provided, estimates it by heavy Gaussian blur.

    Args:
        image: 2D grayscale image.
        flat_field: 2D flat-field image. If None, estimated from image.
        dark_field: 2D dark-field image. If None, uses 0.
        sigma: Gaussian sigma for flat-field estimation (default: 1/4 of
            image width). Only used when flat_field is None.

    Returns:
        dict with:
            corrected: Illumination-corrected image.
            flat_field: The flat-field used.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("Image must be 2D")

    dark = np.zeros_like(img)
    if dark_field is not None:
        dark = np.asarray(dark_field, dtype=np.float64)

    if flat_field is None:
        if sigma is None:
            sigma = max(img.shape) // 4
        flat = ndimage.gaussian_filter(img, sigma=sigma)
    else:
        flat = np.asarray(flat_field, dtype=np.float64)

    denominator = flat - dark
    mean_denom = denominator.mean()
    denominator[denominator < 1e-10] = 1e-10

    corrected = (img - dark) / denominator * max(mean_denom, 1e-10)
    corrected = np.clip(corrected, 0, None)

    return {
        'corrected': corrected,
        'flat_field': flat,
    }


def measure_roi(image, mask, background_mask=None):
    """Measure fluorescence intensity within a masked ROI.

    Args:
        image: 2D grayscale image.
        mask: Boolean 2D mask defining the ROI.
        background_mask: Optional boolean mask for background estimation.
            If None, uses pixels outside the ROI mask.

    Returns:
        dict with:
            mean: Mean intensity in ROI.
            median: Median intensity in ROI.
            std: Standard deviation in ROI.
            total: Integrated (total) intensity in ROI.
            area: Number of pixels in ROI.
            background_mean: Mean background intensity.
            signal_to_background: ROI mean / background mean.
            corrected_mean: ROI mean minus background mean.
    """
    img = np.asarray(image, dtype=np.float64)
    roi = np.asarray(mask, dtype=bool)

    if img.shape != roi.shape:
        raise ValueError("Image and mask must have same shape")

    roi_pixels = img[roi]
    n = len(roi_pixels)

    if n == 0:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'total': 0.0, 'area': 0,
            'background_mean': 0.0,
            'signal_to_background': 0.0,
            'corrected_mean': 0.0,
        }

    roi_mean = float(np.mean(roi_pixels))
    roi_median = float(np.median(roi_pixels))
    roi_std = float(np.std(roi_pixels))
    roi_total = float(np.sum(roi_pixels))

    # Background
    if background_mask is not None:
        bg_pixels = img[np.asarray(background_mask, dtype=bool)]
    else:
        bg_pixels = img[~roi]

    if len(bg_pixels) > 0:
        bg_mean = float(np.mean(bg_pixels))
    else:
        bg_mean = 0.0

    sb_ratio = roi_mean / max(bg_mean, 1e-10)
    corrected = roi_mean - bg_mean

    return {
        'mean': round(roi_mean, 3),
        'median': round(roi_median, 3),
        'std': round(roi_std, 3),
        'total': round(roi_total, 3),
        'area': n,
        'background_mean': round(bg_mean, 3),
        'signal_to_background': round(sb_ratio, 3),
        'corrected_mean': round(corrected, 3),
    }


def bleach_correct(images, method='exponential'):
    """Correct photobleaching in a fluorescence time series.

    Args:
        images: List of 2D arrays or 3D array (T, H, W).
        method: 'exponential' (fit and divide out decay),
                'ratio' (normalize each frame to first frame mean),
                'histogram' (match histogram of each frame to first).

    Returns:
        dict with:
            corrected: List of corrected 2D arrays.
            bleach_curve: 1D array of mean intensity per frame (before correction).
            correction_factors: 1D array of multiplicative correction factors.
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        stack = [images[i].astype(np.float64) for i in range(images.shape[0])]
    else:
        stack = [np.asarray(img, dtype=np.float64) for img in images]

    n = len(stack)
    if n == 0:
        return {'corrected': [], 'bleach_curve': np.array([]),
                'correction_factors': np.array([])}

    means = np.array([img.mean() for img in stack])
    ref_mean = means[0] if means[0] > 0 else 1.0

    if method == 'ratio':
        factors = ref_mean / np.maximum(means, 1e-10)

    elif method == 'exponential':
        # Fit exponential: I(t) = A * exp(-t/tau)
        t = np.arange(n, dtype=np.float64)
        valid = means > 0
        if valid.sum() >= 2:
            log_means = np.log(np.maximum(means[valid], 1e-10))
            coeffs = np.polyfit(t[valid], log_means, 1)
            fitted = np.exp(coeffs[0] * t + coeffs[1])
            factors = fitted[0] / np.maximum(fitted, 1e-10)
        else:
            factors = np.ones(n)

    elif method == 'histogram':
        # Simple ratio-based (histogram matching is expensive)
        factors = ref_mean / np.maximum(means, 1e-10)

    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Use 'exponential', 'ratio', or 'histogram'.")

    corrected = [stack[i] * factors[i] for i in range(n)]

    return {
        'corrected': corrected,
        'bleach_curve': means,
        'correction_factors': factors,
    }


def normalize_intensity(image, method='minmax', pmin=1, pmax=99):
    """Normalize image intensity to [0, 1] range.

    Args:
        image: 2D grayscale image.
        method: 'minmax' (absolute min/max), 'percentile' (pmin/pmax).
        pmin: Lower percentile for percentile method.
        pmax: Upper percentile for percentile method.

    Returns:
        2D float64 array in [0, 1].
    """
    img = np.asarray(image, dtype=np.float64)

    if method == 'minmax':
        lo, hi = img.min(), img.max()
    elif method == 'percentile':
        lo = float(np.percentile(img, pmin))
        hi = float(np.percentile(img, pmax))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax' or 'percentile'.")

    if hi > lo:
        result = (img - lo) / (hi - lo)
    else:
        result = np.zeros_like(img)

    return np.clip(result, 0.0, 1.0)
