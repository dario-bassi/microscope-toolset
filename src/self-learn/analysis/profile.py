"""Line and radial profile analysis for microscopy images.

Extract and analyze intensity profiles along lines, paths, or
radial directions. Useful for measuring distances, gradients,
and structural features.

Functions:
    line_profile      -- Extract intensity along a straight line
    path_profile      -- Extract intensity along an arbitrary path
    radial_profile    -- Radial intensity profile from a center point
    find_edges        -- Find edge positions in a profile
    measure_width     -- Measure full width at half maximum
"""

import numpy as np
from scipy import ndimage


def line_profile(image, start, end, width=1):
    """Extract intensity profile along a straight line.

    Args:
        image: 2D grayscale image.
        start: (row, col) or (y, x) start point.
        end: (row, col) or (y, x) end point.
        width: Line width in pixels (perpendicular averaging).

    Returns:
        dict with:
            profile: 1D array of intensity values along the line.
            distance: 1D array of distances from start (pixels).
            length: Total line length in pixels.
            mean: Mean intensity along the line.
            max: Maximum intensity.
            min: Minimum intensity.
    """
    img = np.asarray(image, dtype=np.float64)
    y0, x0 = float(start[0]), float(start[1])
    y1, x1 = float(end[0]), float(end[1])

    length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    if length < 1:
        return {
            "profile": np.array([img[int(y0), int(x0)]]),
            "distance": np.array([0.0]),
            "length": 0.0,
            "mean": float(img[int(y0), int(x0)]),
            "max": float(img[int(y0), int(x0)]),
            "min": float(img[int(y0), int(x0)]),
        }

    n_points = int(np.ceil(length))
    t = np.linspace(0, 1, n_points)

    if width <= 1:
        # Simple interpolation along the line
        ys = y0 + t * (y1 - y0)
        xs = x0 + t * (x1 - x0)
        profile = ndimage.map_coordinates(img, [ys, xs], order=1)
    else:
        # Average perpendicular to line direction
        dy, dx = y1 - y0, x1 - x0
        norm = np.sqrt(dy**2 + dx**2)
        ny, nx = -dx / norm, dy / norm  # perpendicular unit vector

        offsets = np.linspace(-(width - 1) / 2, (width - 1) / 2, width)
        profiles = []
        for off in offsets:
            ys = y0 + t * (y1 - y0) + off * ny
            xs = x0 + t * (x1 - x0) + off * nx
            # Clip to image bounds
            ys = np.clip(ys, 0, img.shape[0] - 1)
            xs = np.clip(xs, 0, img.shape[1] - 1)
            profiles.append(ndimage.map_coordinates(img, [ys, xs], order=1))
        profile = np.mean(profiles, axis=0)

    distance = np.linspace(0, length, n_points)

    return {
        "profile": profile,
        "distance": distance,
        "length": round(float(length), 2),
        "mean": round(float(profile.mean()), 3),
        "max": round(float(profile.max()), 3),
        "min": round(float(profile.min()), 3),
    }


def path_profile(image, points, width=1):
    """Extract intensity along an arbitrary path defined by waypoints.

    Args:
        image: 2D grayscale image.
        points: (N, 2) array of (row, col) waypoints.
        width: Line width for averaging.

    Returns:
        dict with:
            profile: 1D array of intensity values.
            distance: 1D array of cumulative distance from start.
            total_length: Total path length.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        if len(pts) == 1:
            img = np.asarray(image, dtype=np.float64)
            r, c = int(pts[0, 0]), int(pts[0, 1])
            r = max(0, min(r, img.shape[0] - 1))
            c = max(0, min(c, img.shape[1] - 1))
            return {
                "profile": np.array([img[r, c]]),
                "distance": np.array([0.0]),
                "total_length": 0.0,
            }
        return {"profile": np.array([]), "distance": np.array([]), "total_length": 0.0}

    all_profiles = []
    all_distances = []
    cum_dist = 0.0

    for i in range(len(pts) - 1):
        seg = line_profile(image, pts[i], pts[i + 1], width=width)
        if i > 0:
            # Skip first point to avoid duplication
            all_profiles.append(seg["profile"][1:])
            all_distances.append(seg["distance"][1:] + cum_dist)
        else:
            all_profiles.append(seg["profile"])
            all_distances.append(seg["distance"])
        cum_dist += seg["length"]

    profile = np.concatenate(all_profiles) if all_profiles else np.array([])
    distance = np.concatenate(all_distances) if all_distances else np.array([])

    return {
        "profile": profile,
        "distance": distance,
        "total_length": round(cum_dist, 2),
    }


def find_edges(profile, threshold=None, method="gradient"):
    """Find edge positions in an intensity profile.

    Args:
        profile: 1D intensity array.
        threshold: Intensity threshold for edge detection.
            If None, uses midpoint between min and max.
        method: 'gradient' (steepest gradient) or 'threshold' (crossing).

    Returns:
        dict with:
            edges: List of edge positions (indices, fractional).
            n_edges: Number of edges found.
            edge_strengths: Gradient magnitude at each edge.
    """
    prof = np.asarray(profile, dtype=np.float64)
    if len(prof) < 3:
        return {"edges": [], "n_edges": 0, "edge_strengths": []}

    if method == "gradient":
        grad = np.gradient(prof)
        abs_grad = np.abs(grad)

        # Find local maxima of gradient magnitude
        edges = []
        strengths = []
        for i in range(1, len(abs_grad) - 1):
            if abs_grad[i] > abs_grad[i - 1] and abs_grad[i] > abs_grad[i + 1]:
                # Must be above noise floor
                noise = np.median(abs_grad)
                if abs_grad[i] > noise * 2:
                    edges.append(float(i))
                    strengths.append(float(abs_grad[i]))

    elif method == "threshold":
        if threshold is None:
            threshold = (prof.min() + prof.max()) / 2

        edges = []
        strengths = []
        for i in range(len(prof) - 1):
            if (prof[i] < threshold and prof[i + 1] >= threshold) or (
                prof[i] >= threshold and prof[i + 1] < threshold
            ):
                # Linear interpolation for sub-pixel position
                frac = (threshold - prof[i]) / (prof[i + 1] - prof[i])
                edges.append(i + frac)
                strengths.append(abs(float(prof[i + 1] - prof[i])))

    else:
        raise ValueError(f"Unknown method: {method}.")

    return {
        "edges": edges,
        "n_edges": len(edges),
        "edge_strengths": strengths,
    }


def measure_width(profile, level=0.5):
    """Measure full width at a fraction of maximum.

    Args:
        profile: 1D intensity array.
        level: Fraction of peak height (0.5 = FWHM).

    Returns:
        dict with:
            width: Width in pixels at the given level.
            peak_position: Index of the peak.
            peak_value: Maximum value.
            left_edge: Left crossing position.
            right_edge: Right crossing position.
    """
    prof = np.asarray(profile, dtype=np.float64)
    if len(prof) < 3:
        return {
            "width": 0.0,
            "peak_position": 0,
            "peak_value": float(prof.max()) if len(prof) > 0 else 0.0,
            "left_edge": 0.0,
            "right_edge": 0.0,
        }

    peak_idx = int(np.argmax(prof))
    peak_val = float(prof[peak_idx])
    bg = float(prof.min())
    threshold = bg + (peak_val - bg) * level

    # Find left edge
    left = 0.0
    for i in range(peak_idx, 0, -1):
        if prof[i - 1] < threshold:
            frac = (threshold - prof[i - 1]) / max(prof[i] - prof[i - 1], 1e-10)
            left = (i - 1) + frac
            break

    # Find right edge
    right = float(len(prof) - 1)
    for i in range(peak_idx, len(prof) - 1):
        if prof[i + 1] < threshold:
            frac = (threshold - prof[i]) / max(prof[i] - prof[i + 1], 1e-10)
            right = i + frac
            break

    width = right - left

    return {
        "width": round(float(width), 3),
        "peak_position": peak_idx,
        "peak_value": round(peak_val, 3),
        "left_edge": round(float(left), 3),
        "right_edge": round(float(right), 3),
    }
