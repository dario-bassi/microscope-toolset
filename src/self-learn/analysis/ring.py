"""Ring / annular structure analysis.

Measure inner and outer boundaries of ring-shaped structures
(organoids, ZOI, spheroid shells) using radial profile and
angular ray-casting.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def find_ring_center(mask):
    """Find the center of a ring-shaped binary mask.

    Args:
        mask: 2D binary array where True = ring pixels.

    Returns:
        (cx, cy) center coordinates.
    """
    cy, cx = ndimage.center_of_mass(mask)
    return float(cx), float(cy)


def ring_radii(image, center=None, threshold=None, n_angles=360):
    """Measure inner and outer radii of a ring structure by ray-casting.

    Casts rays from the center at evenly spaced angles and finds where
    the signal first rises above and last drops below the threshold.

    Args:
        image: 2D array (e.g. membrane channel showing a ring).
        center: (cx, cy). If None, detected from thresholded image.
        threshold: Intensity threshold for ring pixels.
            If None, uses 3 * background_std.
        n_angles: Number of rays to cast.

    Returns:
        dict with:
            inner_radii: Array of inner radius per angle (n_angles,).
            outer_radii: Array of outer radius per angle (n_angles,).
            angles: Array of angles in radians.
            inner_mean: Mean inner radius.
            outer_mean: Mean outer radius.
            inner_std: Std of inner radius.
            outer_std: Std of outer radius.
    """
    img = np.asarray(image, dtype=np.float64)
    smooth = gaussian_filter(img, sigma=1)
    h, w = smooth.shape

    if threshold is None:
        bg_level = np.percentile(smooth, 50)
        signal_level = np.percentile(smooth, 99)
        threshold = bg_level + 0.2 * (signal_level - bg_level)
        if threshold <= bg_level:
            threshold = bg_level + 1

    if center is None:
        mask = smooth > threshold
        if mask.sum() == 0:
            raise ValueError("No signal above threshold")
        center = find_ring_center(mask)

    cx, cy = center
    max_r = min(cx, cy, w - cx, h - cy) * 0.95
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    inner_radii = []
    outer_radii = []
    valid_angles = []

    for theta in angles:
        rs = np.arange(0, max_r, 0.5)
        ray_x = cx + rs * np.cos(theta)
        ray_y = cy + rs * np.sin(theta)

        valid = (ray_x >= 0) & (ray_x < w) & (ray_y >= 0) & (ray_y < h)
        rs = rs[valid]
        xi = np.clip(ray_x[valid].astype(int), 0, w - 1)
        yi = np.clip(ray_y[valid].astype(int), 0, h - 1)

        ray_vals = smooth[yi, xi]
        above = ray_vals > threshold

        if above.sum() < 3:
            continue

        first = np.where(above)[0][0]
        last = np.where(above)[0][-1]
        inner_radii.append(float(rs[first]))
        outer_radii.append(float(rs[last]))
        valid_angles.append(theta)

    inner_radii = np.array(inner_radii)
    outer_radii = np.array(outer_radii)
    valid_angles = np.array(valid_angles)

    return {
        "inner_radii": inner_radii,
        "outer_radii": outer_radii,
        "angles": valid_angles,
        "inner_mean": float(inner_radii.mean()),
        "outer_mean": float(outer_radii.mean()),
        "inner_std": float(inner_radii.std()),
        "outer_std": float(outer_radii.std()),
    }


def measure_ring(image, center=None, threshold=None, pixel_size=1.0):
    """Measure a ring structure: outer/inner diameters and wall thickness.

    Combines ray-casting boundary detection with filled-mask regionprops
    for robust measurement.

    Args:
        image: 2D array (fluorescence channel showing ring).
        center: (cx, cy). Auto-detected if None.
        threshold: Ring signal threshold. Auto if None.
        pixel_size: Microns per pixel for unit conversion.

    Returns:
        dict with outer_diameter_um, inner_diameter_um, wall_thickness_um,
        center, and raw radii data.
    """
    from skimage.measure import label, regionprops

    radii = ring_radii(image, center=center, threshold=threshold)

    # Also measure via filled mask for equivalent diameter
    smooth = gaussian_filter(np.asarray(image, dtype=np.float64), sigma=1)
    if threshold is None:
        bg_level = np.percentile(smooth, 50)
        signal_level = np.percentile(smooth, 99)
        threshold = bg_level + 0.2 * (signal_level - bg_level)
        if threshold <= bg_level:
            threshold = bg_level + 1

    mask = smooth > threshold
    lbl, n = ndimage.label(mask)
    if n == 0:
        return {
            "outer_diameter_um": 2 * radii["outer_mean"] * pixel_size,
            "inner_diameter_um": 2 * radii["inner_mean"] * pixel_size,
            "wall_thickness_um": (radii["outer_mean"] - radii["inner_mean"]) * pixel_size,
            "center": center,
            "radii": radii,
        }

    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    wall_mask = lbl == (np.argmax(sizes) + 1)
    filled = ndimage.binary_fill_holes(wall_mask)

    fp = regionprops(filled.astype(int))
    outer_d = (
        fp[0].equivalent_diameter_area * pixel_size if fp else 2 * radii["outer_mean"] * pixel_size
    )

    # Lumen = hole in the filled ring
    hole = filled & ~wall_mask
    hl = label(hole)
    hp = regionprops(hl)
    if hp:
        if center is not None:
            cx, cy = center
        else:
            cx, cy = radii.get("center", (smooth.shape[1] / 2, smooth.shape[0] / 2))
        # Pick the center-connected hole
        center_hole = max(
            [h for h in hp if np.sqrt((h.centroid[1] - cx) ** 2 + (h.centroid[0] - cy) ** 2) < 50],
            key=lambda h: h.area,
            default=None,
        )
        if center_hole:
            inner_d = center_hole.equivalent_diameter_area * pixel_size
        else:
            inner_d = 2 * radii["inner_mean"] * pixel_size
    else:
        inner_d = 2 * radii["inner_mean"] * pixel_size

    wall_t = (outer_d - inner_d) / 2

    return {
        "outer_diameter_um": float(outer_d),
        "inner_diameter_um": float(inner_d),
        "wall_thickness_um": float(wall_t),
        "center": center or find_ring_center(mask),
        "radii": radii,
    }


def find_best_z_plane(stack, z_positions, metric="filled_area", signal_threshold=5):
    """Find the Z-plane with the best metric for a ring structure.

    Args:
        stack: (N, H, W) Z-stack array.
        z_positions: 1D array of Z values.
        metric: 'filled_area' (largest cross-section) or
                'ring_contrast' (strongest ring signal).
        signal_threshold: Threshold for detecting the ring.

    Returns:
        dict with best_z, best_idx, scores (array of metric per plane).
    """
    scores = np.zeros(len(z_positions))

    for i, _z in enumerate(z_positions):
        img = stack[i].astype(float)
        smooth = gaussian_filter(img, sigma=1)
        mask = smooth > signal_threshold

        lbl, n = ndimage.label(mask)
        if n == 0:
            continue

        sizes = ndimage.sum(mask, lbl, range(1, n + 1))
        largest = lbl == (np.argmax(sizes) + 1)

        if metric == "filled_area":
            filled = ndimage.binary_fill_holes(largest)
            scores[i] = filled.sum()
        elif metric == "ring_contrast":
            scores[i] = img[largest].mean() if largest.sum() > 0 else 0

    best_idx = int(np.argmax(scores))
    return {
        "best_z": float(z_positions[best_idx]),
        "best_idx": best_idx,
        "scores": scores,
    }
