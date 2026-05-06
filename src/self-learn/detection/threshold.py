"""Adaptive threshold selection for microscopy images.

Provides smart threshold selection based on image characteristics,
with quality-checked detection retry when initial parameters fail.

Key functions:
    auto_threshold   -- Analyze image and select best threshold value
    threshold_sweep  -- Try multiple thresholds, evaluate each
    detect_with_retry -- Run detection, validate, retry if poor quality
"""

import numpy as np
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle


def auto_threshold(image, method="auto", background_fraction=0.5):
    """Select an appropriate threshold for a microscopy image.

    Analyzes the image histogram to determine whether Otsu, triangle,
    or percentile-based thresholding is most appropriate.

    Args:
        image: 2D grayscale image (uint8 or float).
        method: 'auto' (analyze and choose), 'otsu', 'triangle', 'li',
            'percentile', or 'background'.
        background_fraction: For 'background' method, fraction of pixels
            assumed to be background. Used to set threshold at
            mean(bg) + 3*std(bg).

    Returns:
        dict with:
            threshold: The selected threshold value.
            method: Name of the method used.
            image_stats: Dict of image statistics.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.size == 0:
        return {"threshold": 0.0, "method": "empty", "image_stats": {}}

    stats = {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": float(img.min()),
        "max": float(img.max()),
        "p5": float(np.percentile(img, 5)),
        "p95": float(np.percentile(img, 95)),
        "dynamic_range": float(img.max() - img.min()),
    }

    if method == "otsu":
        thresh = float(threshold_otsu(img))
        return {"threshold": thresh, "method": "otsu", "image_stats": stats}

    elif method == "triangle":
        thresh = float(threshold_triangle(img))
        return {"threshold": thresh, "method": "triangle", "image_stats": stats}

    elif method == "li":
        thresh = float(threshold_li(img))
        return {"threshold": thresh, "method": "li", "image_stats": stats}

    elif method == "percentile":
        # Objects are bright outliers above background
        thresh = float(np.percentile(img, 95))
        return {"threshold": thresh, "method": "percentile_95", "image_stats": stats}

    elif method == "background":
        # Estimate background from the majority of pixels
        sorted_vals = np.sort(img.ravel())
        n_bg = int(len(sorted_vals) * background_fraction)
        bg = sorted_vals[:n_bg]
        thresh = float(bg.mean() + 3 * bg.std())
        stats["bg_mean"] = float(bg.mean())
        stats["bg_std"] = float(bg.std())
        return {"threshold": thresh, "method": "background_3sigma", "image_stats": stats}

    # Auto mode: analyze histogram to pick best method
    if stats["dynamic_range"] < 10:
        # Very low contrast — use percentile
        thresh = float(np.percentile(img, 97))
        return {"threshold": thresh, "method": "percentile_97_low_contrast", "image_stats": stats}

    # Check if bimodal (objects vs background)
    try:
        otsu_val = float(threshold_otsu(img))
        # Check balance: how many pixels above vs below
        above = float((img > otsu_val).sum()) / img.size
        if 0.05 < above < 0.7:
            # Good bimodal split
            return {"threshold": otsu_val, "method": "otsu", "image_stats": stats}
    except ValueError:
        pass

    # Fallback: background estimation
    sorted_vals = np.sort(img.ravel())
    n_bg = int(len(sorted_vals) * background_fraction)
    bg = sorted_vals[:n_bg]
    thresh = float(bg.mean() + 3 * bg.std())
    stats["bg_mean"] = float(bg.mean())
    stats["bg_std"] = float(bg.std())
    return {"threshold": thresh, "method": "background_3sigma", "image_stats": stats}


def threshold_sweep(image, thresholds, count_fn=None, min_area=5):
    """Try multiple thresholds and return detection counts for each.

    Useful for finding the "elbow" in detection count vs threshold.

    Args:
        image: 2D grayscale image.
        thresholds: List of threshold values to try.
        count_fn: Optional callable(binary_mask) → int.
            If None, counts connected components above min_area.
        min_area: Minimum object area for default counting.

    Returns:
        dict with:
            thresholds: List of thresholds tried.
            counts: List of object counts at each threshold.
            areas_median: List of median object areas.
            best_threshold: Threshold at the elbow (steepest drop).
    """
    from skimage.measure import label, regionprops

    img = np.asarray(image, dtype=np.float64)
    counts = []
    areas_median = []

    for t in thresholds:
        mask = img > t
        if count_fn is not None:
            counts.append(count_fn(mask))
            areas_median.append(0)
        else:
            labeled = label(mask)
            props = [p for p in regionprops(labeled) if p.area >= min_area]
            counts.append(len(props))
            if props:
                areas_median.append(float(np.median([p.area for p in props])))
            else:
                areas_median.append(0)

    counts = np.array(counts)
    thresholds_arr = np.array(thresholds)

    # Find elbow: steepest negative slope in count vs threshold
    if len(counts) > 2:
        diffs = np.diff(counts)
        # Elbow is where count drops most sharply
        elbow_idx = int(np.argmin(diffs))
        best_threshold = float(thresholds_arr[elbow_idx + 1])
    else:
        best_threshold = float(thresholds_arr[0]) if len(thresholds_arr) > 0 else 0

    return {
        "thresholds": list(thresholds),
        "counts": counts.tolist(),
        "areas_median": areas_median,
        "best_threshold": best_threshold,
    }


def detect_with_retry(image, detect_fn, quality_fn=None, max_retries=3, threshold_range=None):
    """Run detection with automatic retry on quality failure.

    If the initial detection fails quality checks, adjusts the threshold
    and retries up to ``max_retries`` times.

    Args:
        image: 2D grayscale image.
        detect_fn: callable(image, threshold) → dict with at least
            'count' and optionally 'areas', 'centroids'.
        quality_fn: Optional callable(detection_result) → dict with 'status'.
            If status != 'ok', retry with adjusted threshold.
            If None, accepts any result.
        max_retries: Maximum retry attempts.
        threshold_range: (low, high) range for threshold sweep on retry.
            If None, uses auto_threshold result ± 30%.

    Returns:
        dict with:
            result: Best detection result.
            attempts: Number of attempts made.
            thresholds_tried: List of thresholds tried.
            quality_results: List of quality check results.
    """
    # Initial threshold
    auto = auto_threshold(image)
    initial_thresh = auto["threshold"]

    if threshold_range is None:
        low = initial_thresh * 0.5
        high = initial_thresh * 1.5
    else:
        low, high = threshold_range

    thresholds_to_try = [initial_thresh]
    # Add evenly spaced alternatives
    for i in range(1, max_retries):
        t = low + (high - low) * i / max_retries
        if t not in thresholds_to_try:
            thresholds_to_try.append(t)

    results = []
    quality_results = []

    for t in thresholds_to_try:
        det = detect_fn(image, t)
        results.append(det)

        if quality_fn is not None:
            q = quality_fn(det)
            quality_results.append(q)
            if q.get("status") == "ok":
                return {
                    "result": det,
                    "attempts": len(results),
                    "thresholds_tried": thresholds_to_try[: len(results)],
                    "quality_results": quality_results,
                }
        else:
            return {
                "result": det,
                "attempts": 1,
                "thresholds_tried": [t],
                "quality_results": [],
            }

    # No threshold passed quality — return best (most 'ok'-like)
    if quality_results:
        # Prefer 'ok', then lowest count difference from expected
        best_idx = 0
        for i, q in enumerate(quality_results):
            if q.get("status") == "ok":
                best_idx = i
                break
        return {
            "result": results[best_idx],
            "attempts": len(results),
            "thresholds_tried": thresholds_to_try[: len(results)],
            "quality_results": quality_results,
        }

    return {
        "result": results[0] if results else {},
        "attempts": len(results),
        "thresholds_tried": thresholds_to_try[: len(results)],
        "quality_results": quality_results,
    }
