"""Colocalization analysis for multi-channel fluorescence microscopy.

Quantifies spatial overlap between two fluorescence channels.
Standard metrics for assessing protein co-expression, organelle overlap,
or marker specificity.

Functions:
    pearson_r           -- Pearson correlation coefficient (whole-image)
    manders_coefficients -- Manders M1/M2 overlap coefficients
    costes_threshold    -- Automatic threshold via Costes regression
    intensity_scatter   -- Generate scatter plot data for two channels
    colocalization_map  -- Per-pixel colocalization mask
"""

import numpy as np


def pearson_r(ch1, ch2, mask=None):
    """Pearson correlation coefficient between two channels.

    Measures linear correlation of intensities. Range: -1 to +1.
    +1 = perfect colocalization, 0 = no correlation, -1 = anti-correlated.

    Args:
        ch1, ch2: 2D images (same shape).
        mask: Optional binary mask to restrict analysis to ROI.

    Returns:
        dict with:
            r: Pearson correlation coefficient.
            n_pixels: Number of pixels used.
    """
    a = np.asarray(ch1, dtype=np.float64)
    b = np.asarray(ch2, dtype=np.float64)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a = a[m]
        b = b[m]
    else:
        a = a.ravel()
        b = b.ravel()

    n = len(a)
    if n < 2:
        return {"r": 0.0, "n_pixels": n}

    a_mean = a.mean()
    b_mean = b.mean()
    a_dev = a - a_mean
    b_dev = b - b_mean

    num = np.sum(a_dev * b_dev)
    denom = np.sqrt(np.sum(a_dev**2) * np.sum(b_dev**2))

    if denom < 1e-10:
        return {"r": 0.0, "n_pixels": n}

    r = float(num / denom)
    return {"r": r, "n_pixels": n}


def manders_coefficients(ch1, ch2, threshold1=0, threshold2=0, mask=None):
    """Manders overlap coefficients M1 and M2.

    M1: fraction of ch1 that overlaps with ch2 (above threshold2).
    M2: fraction of ch2 that overlaps with ch1 (above threshold1).

    Args:
        ch1, ch2: 2D images (same shape).
        threshold1: Threshold for ch1 signal (pixels below are background).
        threshold2: Threshold for ch2 signal.
        mask: Optional binary mask.

    Returns:
        dict with:
            M1: Fraction of ch1 overlapping ch2.
            M2: Fraction of ch2 overlapping ch1.
            overlap_coefficient: Overlap coefficient (OC).
    """
    a = np.asarray(ch1, dtype=np.float64)
    b = np.asarray(ch2, dtype=np.float64)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a = a[m]
        b = b[m]
    else:
        a = a.ravel()
        b = b.ravel()

    # Manders M1: fraction of ch1 intensity in regions where ch2 > threshold
    ch1_signal = a > threshold1
    ch2_signal = b > threshold2

    sum_a = float(a[ch1_signal].sum())
    sum_b = float(b[ch2_signal].sum())

    if sum_a > 0:
        M1 = float(a[ch1_signal & ch2_signal].sum()) / sum_a
    else:
        M1 = 0.0

    if sum_b > 0:
        M2 = float(b[ch2_signal & ch1_signal].sum()) / sum_b
    else:
        M2 = 0.0

    # Overlap coefficient
    denom_oc = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom_oc > 0:
        OC = float(np.sum(a * b) / denom_oc)
    else:
        OC = 0.0

    return {"M1": M1, "M2": M2, "overlap_coefficient": OC}


def costes_threshold(ch1, ch2, mask=None):
    """Automatic threshold selection via Costes regression method.

    Finds the threshold pair where the Pearson r between below-threshold
    pixels drops to zero — the point where only background remains.

    Args:
        ch1, ch2: 2D images.
        mask: Optional binary mask.

    Returns:
        dict with:
            threshold1: Auto-threshold for ch1.
            threshold2: Auto-threshold for ch2.
            r_total: Pearson r of full images.
            slope: Regression slope (ch2 = slope * ch1 + intercept).
            intercept: Regression intercept.
    """
    a = np.asarray(ch1, dtype=np.float64)
    b = np.asarray(ch2, dtype=np.float64)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        av = a[m]
        bv = b[m]
    else:
        av = a.ravel()
        bv = b.ravel()

    n = len(av)
    if n < 10:
        return {
            "threshold1": 0.0,
            "threshold2": 0.0,
            "r_total": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
        }

    # Total Pearson r
    r_total = pearson_r(ch1, ch2, mask)["r"]

    # Linear regression: ch2 = slope * ch1 + intercept
    a_mean = av.mean()
    b_mean = bv.mean()
    ss_aa = np.sum((av - a_mean) ** 2)
    ss_ab = np.sum((av - a_mean) * (bv - b_mean))

    if ss_aa < 1e-10:
        slope = 0.0
    else:
        slope = float(ss_ab / ss_aa)
    intercept = float(b_mean - slope * a_mean)

    # Sweep thresholds from max down to find where r of below-threshold → 0
    max_a = float(av.max())
    best_t1 = 0.0
    best_t2 = 0.0

    n_steps = min(50, int(max_a))
    if n_steps < 5:
        n_steps = 5

    for i in range(n_steps, -1, -1):
        t1 = max_a * i / n_steps
        t2 = slope * t1 + intercept

        below = (av < t1) | (bv < t2)
        if below.sum() < 10:
            continue

        a_below = av[below]
        b_below = bv[below]

        a_m = a_below.mean()
        b_m = b_below.mean()
        num = np.sum((a_below - a_m) * (b_below - b_m))
        den = np.sqrt(np.sum((a_below - a_m) ** 2) * np.sum((b_below - b_m) ** 2))
        if den < 1e-10:
            r_below = 0.0
        else:
            r_below = num / den

        if r_below <= 0:
            best_t1 = float(t1)
            best_t2 = float(t2)
            break

    return {
        "threshold1": best_t1,
        "threshold2": best_t2,
        "r_total": r_total,
        "slope": slope,
        "intercept": intercept,
    }


def intensity_scatter(ch1, ch2, mask=None, n_sample=5000):
    """Generate scatter plot data for two channels.

    Subsamples for efficiency if image is large.

    Args:
        ch1, ch2: 2D images.
        mask: Optional binary mask.
        n_sample: Maximum number of points to return.

    Returns:
        dict with:
            x: ch1 intensities (1D array).
            y: ch2 intensities (1D array).
            n_total: Total pixel count before sampling.
    """
    a = np.asarray(ch1, dtype=np.float64)
    b = np.asarray(ch2, dtype=np.float64)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a = a[m]
        b = b[m]
    else:
        a = a.ravel()
        b = b.ravel()

    n_total = len(a)
    if n_total > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_sample, replace=False)
        a = a[idx]
        b = b[idx]

    return {"x": a, "y": b, "n_total": n_total}


def colocalization_map(ch1, ch2, threshold1=0, threshold2=0):
    """Per-pixel colocalization mask.

    Creates a classification of each pixel as:
    - Both channels (colocalized)
    - ch1 only
    - ch2 only
    - Neither (background)

    Args:
        ch1, ch2: 2D images.
        threshold1, threshold2: Background thresholds.

    Returns:
        dict with:
            map: 2D int array (0=bg, 1=ch1_only, 2=ch2_only, 3=both).
            fraction_coloc: Fraction of signal pixels that colocalize.
            n_ch1_only: Pixels with ch1 signal only.
            n_ch2_only: Pixels with ch2 signal only.
            n_both: Pixels with both channels.
            n_neither: Background pixels.
    """
    a = np.asarray(ch1, dtype=np.float64)
    b = np.asarray(ch2, dtype=np.float64)

    sig1 = a > threshold1
    sig2 = b > threshold2

    result = np.zeros(a.shape, dtype=np.int32)
    result[sig1 & ~sig2] = 1  # ch1 only
    result[~sig1 & sig2] = 2  # ch2 only
    result[sig1 & sig2] = 3  # both

    n_ch1 = int((result == 1).sum())
    n_ch2 = int((result == 2).sum())
    n_both = int((result == 3).sum())
    n_bg = int((result == 0).sum())
    n_signal = n_ch1 + n_ch2 + n_both

    frac = n_both / n_signal if n_signal > 0 else 0.0

    return {
        "map": result,
        "fraction_coloc": float(frac),
        "n_ch1_only": n_ch1,
        "n_ch2_only": n_ch2,
        "n_both": n_both,
        "n_neither": n_bg,
    }
