"""Intensity analysis: SNR computation, radial profiles, classification.

Provides compute_snr() for image quality, radial_profile() for
spatial intensity analysis, classify_intensities() for multi-class
grouping, estimate_background() for background estimation, and
fold_change() for background-corrected ratio computation.
"""

import cv2
import numpy as np


def radial_profile(image, center=None, bin_width=1.0):
    """Compute radial intensity profile from a center point.

    Averages pixel intensities in concentric annular bins around
    the center. Useful for:
    - Wave speed measurement (track wavefront vs radius over time)
    - Zone of inhibition measurement (density vs radius from disk)
    - Spheroid radial intensity profile

    Args:
        image: 2D grayscale array.
        center: (cx, cy) center point. If None, uses image center.
        bin_width: Width of each radial bin in pixels.

    Returns:
        dict with:
            radii: Array of bin center radii.
            mean_intensity: Mean intensity at each radius.
            std_intensity: Std of intensity at each radius.
            counts: Number of pixels in each bin.
            max_radius: Maximum radius from center to corner.
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]

    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    # Distance map from center
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    max_radius = float(dist.max())
    n_bins = int(np.ceil(max_radius / bin_width))

    radii = []
    mean_int = []
    std_int = []
    counts = []

    for i in range(n_bins):
        r_inner = i * bin_width
        r_outer = (i + 1) * bin_width
        mask = (dist >= r_inner) & (dist < r_outer)
        px = img[mask]
        if len(px) > 0:
            radii.append((r_inner + r_outer) / 2.0)
            mean_int.append(float(px.mean()))
            std_int.append(float(px.std()))
            counts.append(int(len(px)))

    return {
        "radii": np.array(radii),
        "mean_intensity": np.array(mean_int),
        "std_intensity": np.array(std_int),
        "counts": np.array(counts),
        "max_radius": max_radius,
    }


def local_density(centroids, image_shape, radius=30):
    """Compute local cell density map from centroid positions.

    For each pixel, counts how many cell centroids fall within the given
    radius. Useful for detecting aggregation centers, high-density regions,
    and spatial heterogeneity in cell distributions.

    Args:
        centroids: List of (cx, cy) tuples.
        image_shape: (height, width) of the output density map.
        radius: Radius for density kernel (pixels).

    Returns:
        dict with:
            density_map: 2D array, value = number of cells within radius.
            max_density: Maximum local density.
            max_location: (cx, cy) of densest point.
            mean_density: Mean local density.
    """
    h, w = image_shape[:2]
    density_map = np.zeros((h, w), dtype=np.float64)

    if not centroids:
        return {
            "density_map": density_map,
            "max_density": 0.0,
            "max_location": (0, 0),
            "mean_density": 0.0,
        }

    # Place Gaussian-like kernels at each centroid
    # Use a fast approach: stamp circles for each centroid
    r2 = radius * radius
    for cx, cy in centroids:
        y0 = max(0, int(cy - radius))
        y1 = min(h, int(cy + radius + 1))
        x0 = max(0, int(cx - radius))
        x1 = min(w, int(cx + radius + 1))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r2
        density_map[y0:y1, x0:x1] += mask

    max_idx = np.unravel_index(density_map.argmax(), density_map.shape)
    max_loc = (int(max_idx[1]), int(max_idx[0]))

    return {
        "density_map": density_map,
        "max_density": float(density_map.max()),
        "max_location": max_loc,
        "mean_density": float(density_map.mean()),
    }


def compute_snr(image, threshold=None, mask=None):
    """Compute signal-to-noise ratio for a fluorescence image.

    Separates signal (bright objects) from background (dim pixels) and
    computes SNR = (mean_signal - mean_background) / std_background.

    Args:
        image: 2D grayscale array.
        threshold: Intensity threshold to separate signal from background.
            If None, uses median + 2 * MAD (robust auto-threshold).
        mask: Optional boolean mask where True = signal pixels.
            If provided, threshold is ignored.

    Returns:
        dict with keys: snr, signal_mean, bg_mean, bg_std, signal_pixels,
            bg_pixels, threshold_used, dynamic_range (signal_mean/bg_std).
    """
    img = np.asarray(image, dtype=np.float64)

    if mask is not None:
        signal_mask = mask.astype(bool)
    else:
        if threshold is None:
            med = np.median(img)
            mad = np.median(np.abs(img - med))
            threshold = med + 2 * max(mad, 1)
        signal_mask = img > threshold

    bg_mask = ~signal_mask
    if not signal_mask.any() or not bg_mask.any():
        return {
            "snr": 0.0,
            "signal_mean": float(img[signal_mask].mean()) if signal_mask.any() else 0.0,
            "bg_mean": float(img[bg_mask].mean()) if bg_mask.any() else 0.0,
            "bg_std": 0.0,
            "signal_pixels": int(signal_mask.sum()),
            "bg_pixels": int(bg_mask.sum()),
            "threshold_used": float(threshold) if threshold is not None else 0.0,
            "dynamic_range": 0.0,
        }

    signal_mean = float(img[signal_mask].mean())
    bg_mean = float(img[bg_mask].mean())
    bg_std = float(img[bg_mask].std())

    snr = (signal_mean - bg_mean) / max(bg_std, 1e-6)

    return {
        "snr": snr,
        "signal_mean": signal_mean,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "signal_pixels": int(signal_mask.sum()),
        "bg_pixels": int(bg_mask.sum()),
        "threshold_used": float(threshold) if threshold is not None else 0.0,
        "dynamic_range": signal_mean / max(bg_std, 1e-6),
    }


def classify_intensities(values, n_classes=3, method="auto", class_sizes=None):
    """Classify intensity values into ordered classes.

    Tries multiple methods and picks the best one:
    1. Gap detection -- finds natural breaks in sorted values
    2. K-means -- unsupervised clustering
    3. Equal rank -- forced equal-size groups (when class_sizes are equal)
    4. Constrained -- uses known class_sizes as prior

    Args:
        values: list or array of intensity values (one per cell).
        n_classes: number of intensity classes.
        method: 'auto', 'gap', 'kmeans', 'rank', or 'constrained'.
        class_sizes: optional list of expected sizes per class (brightest first).
            If provided with method='auto' or 'constrained', used as prior.

    Returns:
        dict with:
            labels: list of class labels (0=brightest, n_classes-1=dimmest)
            counts: list of counts per class
            boundaries: list of boundary values between classes
            method_used: which method was applied
    """
    vals = np.array(values, dtype=float)
    n = len(vals)
    order = np.argsort(-vals)  # descending by intensity

    if method == "auto":
        gap_result = _classify_by_gaps(vals, order, n_classes)
        km_result = _classify_by_kmeans(vals, n_classes)

        if gap_result is not None:
            if gap_result["counts"] == km_result["counts"]:
                return gap_result
            return km_result

        if class_sizes and len(set(class_sizes)) == 1:
            return _classify_by_rank(vals, order, n_classes)

        if class_sizes:
            return _classify_constrained(vals, order, class_sizes)

        return km_result

    elif method == "gap":
        result = _classify_by_gaps(vals, order, n_classes)
        return result or _classify_by_kmeans(vals, n_classes)
    elif method == "kmeans":
        return _classify_by_kmeans(vals, n_classes)
    elif method == "rank":
        return _classify_by_rank(vals, order, n_classes)
    elif method == "constrained":
        return _classify_constrained(vals, order, class_sizes or [n // n_classes] * n_classes)
    else:
        raise ValueError(f"Unknown method: {method}")


def _classify_by_gaps(vals, order, n_classes, min_gap_frac=0.15):
    """Classify by finding natural gaps in sorted values."""
    sorted_vals = vals[order]
    n = len(sorted_vals)
    if n < n_classes:
        return None

    gaps = []
    for i in range(n - 1):
        gap = sorted_vals[i] - sorted_vals[i + 1]
        gaps.append((gap, i))

    gaps.sort(reverse=True)
    val_range = sorted_vals[0] - sorted_vals[-1]
    if val_range == 0:
        return None

    needed = n_classes - 1
    significant_gaps = [(g, idx) for g, idx in gaps[:needed] if g / val_range >= min_gap_frac]

    if len(significant_gaps) < needed:
        return None

    split_indices = sorted([idx for _, idx in significant_gaps])

    result_labels = np.zeros(len(vals), dtype=int)
    for rank, orig_idx in enumerate(order):
        cls = 0
        for split_idx in split_indices:
            if rank > split_idx:
                cls += 1
        result_labels[orig_idx] = cls

    counts = [int((result_labels == c).sum()) for c in range(n_classes)]
    boundaries = []
    for split_idx in split_indices:
        boundaries.append(float((sorted_vals[split_idx] + sorted_vals[split_idx + 1]) / 2))

    return {
        "labels": result_labels.tolist(),
        "counts": counts,
        "boundaries": boundaries,
        "method_used": "gap",
    }


def _classify_by_kmeans(vals, n_classes):
    """Classify using k-means clustering."""
    data = vals.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, km_labels, centers = cv2.kmeans(data, n_classes, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    km_labels = km_labels.flatten()
    centers = centers.flatten()

    center_order = np.argsort(-centers)
    label_map = {int(old): new for new, old in enumerate(center_order)}
    result_labels = [label_map[int(lbl)] for lbl in km_labels]

    counts = [result_labels.count(c) for c in range(n_classes)]
    boundaries = []
    sorted_centers = sorted(centers, reverse=True)
    for i in range(len(sorted_centers) - 1):
        boundaries.append(float((sorted_centers[i] + sorted_centers[i + 1]) / 2))

    return {
        "labels": result_labels,
        "counts": counts,
        "boundaries": boundaries,
        "method_used": "kmeans",
    }


def _classify_by_rank(vals, order, n_classes):
    """Classify by equal rank partitioning."""
    n = len(vals)
    size = n // n_classes
    remainder = n % n_classes

    result_labels = np.zeros(n, dtype=int)
    pos = 0
    for cls in range(n_classes):
        cls_size = size + (1 if cls < remainder else 0)
        for i in range(pos, pos + cls_size):
            result_labels[order[i]] = cls
        pos += cls_size

    counts = [int((result_labels == c).sum()) for c in range(n_classes)]
    sorted_vals = vals[order]
    boundaries = []
    pos = 0
    for cls in range(n_classes - 1):
        cls_size = size + (1 if cls < remainder else 0)
        pos += cls_size
        boundaries.append(float((sorted_vals[pos - 1] + sorted_vals[pos]) / 2))

    return {
        "labels": result_labels.tolist(),
        "counts": counts,
        "boundaries": boundaries,
        "method_used": "rank",
    }


def _classify_constrained(vals, order, class_sizes):
    """Classify using known class sizes as constraint."""
    n = len(vals)
    result_labels = np.zeros(n, dtype=int)
    pos = 0
    for cls, cls_size in enumerate(class_sizes):
        for i in range(pos, min(pos + cls_size, n)):
            result_labels[order[i]] = cls
        pos += cls_size

    n_classes = len(class_sizes)
    counts = [int((result_labels == c).sum()) for c in range(n_classes)]
    sorted_vals = vals[order]
    boundaries = []
    pos = 0
    for cls in range(n_classes - 1):
        pos += class_sizes[cls]
        if pos < n:
            boundaries.append(float((sorted_vals[pos - 1] + sorted_vals[pos]) / 2))

    return {
        "labels": result_labels.tolist(),
        "counts": counts,
        "boundaries": boundaries,
        "method_used": "constrained",
    }


def estimate_background(image, mask=None, method="percentile", percentile=5):
    """Estimate fluorescence background intensity.

    For fluorescence images, background is the intensity level in regions
    without signal. Important for computing corrected fold changes and
    quantitative intensity ratios.

    Args:
        image: 2D array, fluorescence image.
        mask: 2D bool array, optional. True = foreground (signal).
            Background is estimated from ~mask pixels. If None, uses
            the method on the full image.
        method: str, estimation method:
            'percentile' — lower percentile of (non-masked) pixels.
            'mode' — most common intensity (histogram peak).
            'corners' — mean of four corner regions (10% of image).
        percentile: int, percentile to use if method='percentile'.

    Returns:
        float, estimated background intensity.
    """
    image = np.asarray(image, dtype=float)

    if mask is not None:
        bg_pixels = image[~np.asarray(mask)]
    else:
        bg_pixels = image.ravel()

    if len(bg_pixels) == 0:
        return 0.0

    if method == "percentile":
        return float(np.percentile(bg_pixels, percentile))
    elif method == "mode":
        hist, edges = np.histogram(bg_pixels, bins=256)
        peak_bin = np.argmax(hist)
        return float((edges[peak_bin] + edges[peak_bin + 1]) / 2)
    elif method == "corners":
        h, w = image.shape[:2]
        ch, cw = max(1, h // 10), max(1, w // 10)
        corners = np.concatenate(
            [
                image[:ch, :cw].ravel(),
                image[:ch, -cw:].ravel(),
                image[-ch:, :cw].ravel(),
                image[-ch:, -cw:].ravel(),
            ]
        )
        return float(np.mean(corners))
    else:
        raise ValueError(f"Unknown method: {method!r}")


def fold_change(hotspot, baseline, background=None, image=None, bg_method="percentile"):
    """Compute background-corrected fold change.

    In fluorescence microscopy, fold change must be computed on
    background-subtracted values to avoid compression toward 1.0:
        fold = (hotspot - bg) / (baseline - bg)

    Without background subtraction, a true 2x change (e.g., 60 vs 30
    on bg=50) appears as 80/60 = 1.33x in raw pixels.

    Args:
        hotspot: float or array, intensity in the activated/elevated region.
        baseline: float or array, intensity in the normal/control region.
        background: float or None. If None and image is provided, estimates
            from image. If both None, uses 0 (raw ratio).
        image: 2D array, optional. Used to estimate background if
            background is not provided.
        bg_method: str, method for estimate_background() if auto-estimating.

    Returns:
        dict with:
            fold_change: float, background-corrected fold change.
            hotspot_corrected: float, hotspot - background.
            baseline_corrected: float, baseline - background.
            background: float, background value used.
            raw_fold: float, uncorrected fold change (hotspot/baseline).
    """
    hotspot = float(np.mean(hotspot))
    baseline = float(np.mean(baseline))

    if background is None and image is not None:
        background = estimate_background(image, method=bg_method)
    elif background is None:
        background = 0.0
    else:
        background = float(background)

    hot_corr = hotspot - background
    base_corr = baseline - background
    raw_fold = hotspot / baseline if baseline > 0 else 0.0
    corr_fold = hot_corr / base_corr if base_corr > 0 else 0.0

    return {
        "fold_change": round(corr_fold, 4),
        "hotspot_corrected": round(hot_corr, 2),
        "baseline_corrected": round(base_corr, 2),
        "background": round(background, 2),
        "raw_fold": round(raw_fold, 4),
    }


# ---------------------------------------------------------------------------
# Backward-compatible re-exports from split-out modules
# ---------------------------------------------------------------------------
