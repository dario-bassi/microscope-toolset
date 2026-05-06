"""Contour and shape analysis for cell morphology.

Computes shape descriptors from binary masks or contour arrays.
Complements morphometry.py (regionprops-based) with contour-specific
metrics useful for cell type classification.

Functions:
    shape_descriptors     -- Comprehensive shape metrics from a binary mask
    fourier_descriptors   -- Fourier shape descriptors for contour matching
    match_shape           -- Compare two shapes using Fourier descriptors
    contour_from_mask     -- Extract contour from a binary mask
    classify_shape        -- Classify as round/elongated/irregular
"""

import numpy as np
from scipy import ndimage


def shape_descriptors(mask):
    """Compute comprehensive shape descriptors from a binary mask.

    Args:
        mask: 2D boolean array (single object).

    Returns:
        dict with:
            area: Area in pixels.
            perimeter: Boundary length in pixels.
            circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle).
            convexity: Convex hull perimeter / perimeter.
            solidity: Area / convex hull area.
            compactness: sqrt(4*area/pi) / major_axis.
            aspect_ratio: Major axis / minor axis.
            extent: Area / bounding box area.
            roundness: 4*area / (pi * major^2).
            centroid: (row, col) centroid.
            major_axis: Major axis length.
            minor_axis: Minor axis length.
            orientation: Orientation angle in degrees (-90 to 90).
    """
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return _empty_descriptors()

    area = int(m.sum())

    # Perimeter: count boundary pixels
    eroded = ndimage.binary_erosion(m)
    boundary = m & ~eroded
    perimeter = float(boundary.sum())
    if perimeter < 1:
        perimeter = 1.0

    # Circularity
    circularity = 4 * np.pi * area / (perimeter**2)
    circularity = min(circularity, 1.0)

    # Convex hull (simple approach using labeled image moments)
    rows, cols = np.where(m)
    if len(rows) < 3:
        return _minimal_descriptors(area, perimeter, rows, cols)

    # Convex hull area via Shoelace formula on hull points
    hull_area, hull_perimeter = _convex_hull_metrics(rows, cols)

    solidity = area / max(hull_area, 1) if hull_area > 0 else 1.0
    convexity = hull_perimeter / max(perimeter, 1) if perimeter > 0 else 1.0

    # Moments for axes
    cy, cx = rows.mean(), cols.mean()
    mu_20 = np.mean((rows - cy) ** 2)
    mu_02 = np.mean((cols - cx) ** 2)
    mu_11 = np.mean((rows - cy) * (cols - cx))

    # Principal axes from inertia tensor
    theta = 0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    proj_major = (rows - cy) * cos_t + (cols - cx) * sin_t
    proj_minor = -(rows - cy) * sin_t + (cols - cx) * cos_t

    major = float(proj_major.max() - proj_major.min())
    minor = float(proj_minor.max() - proj_minor.min())

    if major < 1:
        major = 1.0
    if minor < 1:
        minor = 1.0

    aspect_ratio = major / minor
    compactness = np.sqrt(4 * area / np.pi) / major
    roundness = 4 * area / (np.pi * major**2)

    # Extent
    r_range = rows.max() - rows.min() + 1
    c_range = cols.max() - cols.min() + 1
    bbox_area = r_range * c_range
    extent = area / max(bbox_area, 1)

    return {
        "area": area,
        "perimeter": round(perimeter, 2),
        "circularity": round(float(circularity), 4),
        "convexity": round(float(min(convexity, 1.5)), 4),
        "solidity": round(float(min(solidity, 1.0)), 4),
        "compactness": round(float(compactness), 4),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "extent": round(float(extent), 4),
        "roundness": round(float(roundness), 4),
        "centroid": (round(float(cy), 2), round(float(cx), 2)),
        "major_axis": round(major, 2),
        "minor_axis": round(minor, 2),
        "orientation": round(float(np.degrees(theta)), 2),
    }


def fourier_descriptors(contour, n_descriptors=16):
    """Compute Fourier shape descriptors from a contour.

    Fourier descriptors capture shape independent of position, scale,
    and rotation when properly normalized.

    Args:
        contour: (N, 2) array of (row, col) boundary points.
        n_descriptors: Number of Fourier coefficients to return.

    Returns:
        dict with:
            descriptors: 1D array of normalized Fourier magnitudes.
            n_points: Number of contour points used.
    """
    pts = np.asarray(contour, dtype=np.float64)
    if len(pts) < 4:
        return {"descriptors": np.zeros(n_descriptors), "n_points": len(pts)}

    # Complex representation
    z = pts[:, 1] + 1j * pts[:, 0]  # col + i*row

    # FFT
    Z = np.fft.fft(z)

    # Normalize: translation invariant (remove DC), scale invariant
    Z[0] = 0  # remove translation
    magnitudes = np.abs(Z)

    # Scale invariant: normalize by first non-zero harmonic
    if magnitudes[1] > 0:
        magnitudes /= magnitudes[1]

    # Take requested number of descriptors (symmetric around DC)
    n = min(n_descriptors, len(magnitudes) // 2)
    desc = magnitudes[1 : n + 1]

    # Pad if needed
    if len(desc) < n_descriptors:
        desc = np.concatenate([desc, np.zeros(n_descriptors - len(desc))])

    return {
        "descriptors": desc[:n_descriptors],
        "n_points": len(pts),
    }


def match_shape(contour1, contour2, n_descriptors=16):
    """Compare two shapes using Fourier descriptor distance.

    Args:
        contour1: (N, 2) array of boundary points.
        contour2: (M, 2) array of boundary points.
        n_descriptors: Number of Fourier coefficients to compare.

    Returns:
        dict with:
            distance: L2 distance between descriptor vectors (0 = identical).
            similarity: 1 / (1 + distance) (1.0 = identical).
    """
    fd1 = fourier_descriptors(contour1, n_descriptors)
    fd2 = fourier_descriptors(contour2, n_descriptors)

    d = float(np.sqrt(np.sum((fd1["descriptors"] - fd2["descriptors"]) ** 2)))

    return {
        "distance": round(d, 4),
        "similarity": round(1.0 / (1.0 + d), 4),
    }


def contour_from_mask(mask):
    """Extract ordered boundary points from a binary mask.

    Args:
        mask: 2D boolean array (single object).

    Returns:
        (N, 2) array of (row, col) boundary coordinates, ordered
        counter-clockwise. Empty array if mask has no foreground.
    """
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Find boundary pixels
    eroded = ndimage.binary_erosion(m)
    boundary = m & ~eroded

    rows, cols = np.where(boundary)
    if len(rows) < 3:
        return np.column_stack([rows, cols]).astype(np.float64)

    # Order points by angle from centroid
    cy, cx = rows.mean(), cols.mean()
    angles = np.arctan2(rows - cy, cols - cx)
    order = np.argsort(angles)

    return np.column_stack([rows[order], cols[order]]).astype(np.float64)


def classify_shape(mask, circularity_threshold=0.7, aspect_threshold=2.0):
    """Classify a shape as round, elongated, or irregular.

    Args:
        mask: 2D boolean array.
        circularity_threshold: Above this = round.
        aspect_threshold: Above this = elongated.

    Returns:
        dict with:
            shape_class: 'round', 'elongated', or 'irregular'.
            circularity: Circularity value.
            aspect_ratio: Aspect ratio value.
            confidence: How clearly it fits the class (0-1).
    """
    desc = shape_descriptors(mask)
    circ = desc["circularity"]
    ar = desc["aspect_ratio"]

    if circ >= circularity_threshold and ar < aspect_threshold:
        shape_class = "round"
        confidence = circ
    elif ar >= aspect_threshold:
        shape_class = "elongated"
        confidence = min(ar / 5.0, 1.0)
    else:
        shape_class = "irregular"
        confidence = 1.0 - circ

    return {
        "shape_class": shape_class,
        "circularity": circ,
        "aspect_ratio": ar,
        "confidence": round(float(confidence), 3),
    }


def _empty_descriptors():
    """Return empty shape descriptors."""
    return {
        "area": 0,
        "perimeter": 0.0,
        "circularity": 0.0,
        "convexity": 0.0,
        "solidity": 0.0,
        "compactness": 0.0,
        "aspect_ratio": 1.0,
        "extent": 0.0,
        "roundness": 0.0,
        "centroid": (0.0, 0.0),
        "major_axis": 0.0,
        "minor_axis": 0.0,
        "orientation": 0.0,
    }


def _minimal_descriptors(area, perimeter, rows, cols):
    """Return descriptors for very small objects."""
    cy = float(rows.mean()) if len(rows) > 0 else 0.0
    cx = float(cols.mean()) if len(cols) > 0 else 0.0
    return {
        "area": area,
        "perimeter": round(perimeter, 2),
        "circularity": 1.0,
        "convexity": 1.0,
        "solidity": 1.0,
        "compactness": 1.0,
        "aspect_ratio": 1.0,
        "extent": 1.0,
        "roundness": 1.0,
        "centroid": (cy, cx),
        "major_axis": 1.0,
        "minor_axis": 1.0,
        "orientation": 0.0,
    }


def _convex_hull_metrics(rows, cols):
    """Compute convex hull area and perimeter using Graham scan."""
    points = np.column_stack([cols, rows])

    # Find convex hull using cross product method
    n = len(points)
    if n < 3:
        return float(n), float(n)

    # Sort by x, then y
    idx = np.lexsort((points[:, 1], points[:, 0]))
    points = points[idx]

    # Build upper and lower hulls
    def cross(pt_o, A, B):
        return (A[0] - pt_o[0]) * (B[1] - pt_o[1]) - (A[1] - pt_o[1]) * (B[0] - pt_o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]

    if len(hull) < 3:
        return float(len(points)), float(len(points))

    # Shoelace formula for area
    hull_arr = np.array(hull)
    x, y = hull_arr[:, 0], hull_arr[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Perimeter
    diffs = np.diff(np.vstack([hull_arr, hull_arr[0:1]]), axis=0)
    perimeter = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

    return float(area), perimeter
