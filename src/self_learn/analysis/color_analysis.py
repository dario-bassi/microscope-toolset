"""Color analysis for RGB microscopy images.

Functions for extracting color features from RGB (brightfield, Giemsa, Wright stain)
images. Useful for WBC differential, histology stain analysis, and any task where
cell classification depends on color rather than fluorescence intensity.
"""

import numpy as np


def nucleus_color(patch, dark_percentile=10):
    """Extract the mean color of the darkest pixels in an RGB patch.

    In Giemsa/Wright-stained images, nuclei are the darkest region.
    Returns the mean R, G, B of the bottom ``dark_percentile`` pixels
    by total brightness.

    Args:
        patch: 3D array (H, W, 3) — RGB image patch around a cell.
        dark_percentile: Percentile threshold for "dark" pixels (default 10).

    Returns:
        dict with keys 'R', 'G', 'B' (float means of dark pixels),
        'brightness' (mean total brightness of dark pixels).
        Returns None if patch is empty or grayscale.
    """
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim != 3 or patch.shape[2] < 3:
        return None
    if patch.size == 0:
        return None

    brightness = np.sum(patch[:, :, :3], axis=2)
    thresh = np.percentile(brightness, dark_percentile)
    dark_mask = brightness <= thresh

    if not np.any(dark_mask):
        return None

    R = float(np.mean(patch[:, :, 0][dark_mask]))
    G = float(np.mean(patch[:, :, 1][dark_mask]))
    B = float(np.mean(patch[:, :, 2][dark_mask]))
    return {'R': R, 'G': G, 'B': B, 'brightness': R + G + B}


def detect_color_region(image, color_ranges):
    """Create a mask of pixels matching specified RGB ranges.

    Args:
        image: 3D array (H, W, 3) — RGB image.
        color_ranges: dict with optional keys 'R', 'G', 'B', each a (min, max) tuple.
            Example: {'R': (200, 255), 'G': (120, 180), 'B': (0, 140)} for orange.

    Returns:
        2D boolean mask where all specified channel ranges are satisfied.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros(image.shape[:2], dtype=bool)

    mask = np.ones(image.shape[:2], dtype=bool)
    channel_map = {'R': 0, 'G': 1, 'B': 2}
    for ch_name, (lo, hi) in color_ranges.items():
        idx = channel_map.get(ch_name)
        if idx is not None:
            mask &= (image[:, :, idx] >= lo) & (image[:, :, idx] <= hi)
    return mask


def color_profile(image, positions, radius=10):
    """Compute color statistics for circular ROIs around given positions.

    Args:
        image: 3D array (H, W, 3) — RGB image.
        positions: list of (row, col) tuples (pixel coordinates).
        radius: Radius of circular ROI in pixels.

    Returns:
        list of dicts, each with 'R_mean', 'G_mean', 'B_mean',
        'brightness', and 'n_pixels'.
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    results = []

    for row, col in positions:
        dist = np.sqrt((yy - row) ** 2 + (xx - col) ** 2)
        mask = dist <= radius
        n = int(np.sum(mask))
        if n == 0:
            results.append({'R_mean': 0, 'G_mean': 0, 'B_mean': 0,
                            'brightness': 0, 'n_pixels': 0})
            continue

        if image.ndim == 3 and image.shape[2] >= 3:
            R = float(np.mean(image[:, :, 0][mask]))
            G = float(np.mean(image[:, :, 1][mask]))
            B = float(np.mean(image[:, :, 2][mask]))
        else:
            val = float(np.mean(image.ravel()[:n])) if image.ndim == 2 else 0
            R = G = B = val

        results.append({
            'R_mean': R, 'G_mean': G, 'B_mean': B,
            'brightness': R + G + B, 'n_pixels': n,
        })
    return results


def classify_wbc(patch, dark_percentile=10):
    """Classify a white blood cell from its RGB color profile.

    Uses nucleus darkness and cytoplasm color to distinguish:
    - Eosinophil: orange/pink cytoplasm (R>200, 120<G<180, B<140)
    - Lymphocyte: very dark compact nucleus (brightness < 210)
    - Neutrophil: moderate nucleus, no orange cytoplasm (default)
    - Monocyte: large cell with pale/gray nucleus (brightness > 350)
    - Basophil: very dark purple nucleus and dark cytoplasm

    Args:
        patch: 3D array (H, W, 3) — RGB patch centered on the cell.
        dark_percentile: Percentile for nucleus color extraction.

    Returns:
        dict with 'type' (str), 'confidence' (float 0-1),
        'nucleus_color' (dict from nucleus_color()), 'orange_pixels' (int).
    """
    patch = np.asarray(patch, dtype=np.float64)
    nuc = nucleus_color(patch, dark_percentile)
    if nuc is None:
        return {'type': 'unknown', 'confidence': 0.0,
                'nucleus_color': None, 'orange_pixels': 0}

    # Check for orange cytoplasm (eosinophil marker)
    orange = detect_color_region(patch, {
        'R': (200, 255), 'G': (120, 180), 'B': (0, 140)
    })
    orange_count = int(np.sum(orange))

    # Classification rules
    if orange_count > 3:
        return {'type': 'eosinophil', 'confidence': min(orange_count / 10.0, 1.0),
                'nucleus_color': nuc, 'orange_pixels': orange_count}
    elif nuc['brightness'] < 210:
        return {'type': 'lymphocyte', 'confidence': 0.8,
                'nucleus_color': nuc, 'orange_pixels': orange_count}
    elif nuc['brightness'] > 350:
        return {'type': 'monocyte', 'confidence': 0.6,
                'nucleus_color': nuc, 'orange_pixels': orange_count}
    else:
        return {'type': 'neutrophil', 'confidence': 0.7,
                'nucleus_color': nuc, 'orange_pixels': orange_count}


def hsv_mask(image, hue_range, sat_range=(0, 255), val_range=(0, 255)):
    """Create a mask from HSV ranges on an RGB image.

    Useful for detecting specific stain colors (purple nuclei, pink eosin, etc.)
    without explicit channel thresholds.

    Args:
        image: 3D array (H, W, 3) — RGB image (0-255 range).
        hue_range: (min_hue, max_hue) in degrees (0-360). Wraps around if min > max.
        sat_range: (min_sat, max_sat) in 0-255.
        val_range: (min_val, max_val) in 0-255.

    Returns:
        2D boolean mask.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros(image.shape[:2], dtype=bool)

    # Convert RGB to HSV manually (avoid skimage dependency for this)
    R = image[:, :, 0] / 255.0
    G = image[:, :, 1] / 255.0
    B = image[:, :, 2] / 255.0

    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)
    diff = cmax - cmin

    # Hue (0-360)
    hue = np.zeros_like(diff)
    r_mask = (cmax == R) & (diff > 0)
    g_mask = (cmax == G) & (diff > 0)
    b_mask = (cmax == B) & (diff > 0)
    hue[r_mask] = (60 * ((G[r_mask] - B[r_mask]) / diff[r_mask]) + 360) % 360
    hue[g_mask] = (60 * ((B[g_mask] - R[g_mask]) / diff[g_mask]) + 120) % 360
    hue[b_mask] = (60 * ((R[b_mask] - G[b_mask]) / diff[b_mask]) + 240) % 360

    # Saturation (0-255)
    with np.errstate(invalid='ignore', divide='ignore'):
        sat = np.where(cmax > 0, (diff / cmax) * 255, 0)

    # Value (0-255)
    val = cmax * 255

    # Apply ranges
    h_lo, h_hi = hue_range
    if h_lo <= h_hi:
        h_mask = (hue >= h_lo) & (hue <= h_hi)
    else:
        h_mask = (hue >= h_lo) | (hue <= h_hi)

    s_mask = (sat >= sat_range[0]) & (sat <= sat_range[1])
    v_mask = (val >= val_range[0]) & (val <= val_range[1])

    return h_mask & s_mask & v_mask
