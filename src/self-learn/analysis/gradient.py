"""Radial gradient and edge detection for circular structures.

Provides gradient-based edge finding for zone-of-inhibition (ZOI)
measurement, wavefront tracking, spheroid boundary detection, and
other radially symmetric features. Uses steepest-gradient method
instead of fixed thresholds.

Functions:
    radial_density_profile  -- Density/intensity as function of radius
    steepest_gradient       -- Find steepest gradient point in profile
    half_max_radius         -- Find half-maximum radius from profile
    wavefront_radius        -- Track wavefront expansion over time
    annular_statistics      -- Per-annulus intensity/density statistics
"""

import numpy as np


def radial_density_profile(image, center=None, max_radius=None, n_bins=50, mask=None):
    """Compute radial intensity or density profile from center.

    For binary masks, this gives object density vs radius.
    For intensity images, this gives mean intensity vs radius.

    Args:
        image: 2D array (intensity or binary mask).
        center: (y, x) tuple or None (uses image center).
        max_radius: float or None (uses distance to nearest edge).
        n_bins: int, number of radial bins.
        mask: 2D bool array or None, restrict analysis to this region.

    Returns:
        dict with:
            radii: 1D array, center of each radial bin.
            profile: 1D array, mean value in each bin.
            counts: 1D array, number of pixels in each bin.
            edges: 1D array, bin edges (length n_bins + 1).
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    if center is None:
        center = (h / 2.0, w / 2.0)
    cy, cx = center

    # Distance map
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if max_radius is None:
        max_radius = min(cy, h - cy, cx, w - cx)
    max_radius = float(max_radius)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid = mask & (dist <= max_radius)
    else:
        valid = dist <= max_radius

    edges = np.linspace(0, max_radius, n_bins + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        ring = valid & (dist >= edges[i]) & (dist < edges[i + 1])
        n_px = int(ring.sum())
        counts[i] = n_px
        if n_px > 0:
            profile[i] = float(image[ring].mean())

    return {
        "radii": radii,
        "profile": profile,
        "counts": counts,
        "edges": edges,
    }


def steepest_gradient(radii, profile, direction="decreasing", smooth_window=3):
    """Find the radius with steepest gradient in a radial profile.

    This is the recommended method for ZOI edge detection: the edge
    is where bacterial density changes most rapidly, not where it
    first appears or where it reaches a fixed threshold.

    Args:
        radii: 1D array of radial positions.
        profile: 1D array of values at each radius.
        direction: 'decreasing' (e.g., density drops = ZOI edge) or
                   'increasing' (e.g., density rises at boundary).
        smooth_window: int, smoothing window for gradient computation.

    Returns:
        dict with:
            edge_radius: float, radius of steepest gradient.
            edge_index: int, index in radii array.
            gradient: 1D array, gradient at each radius.
            gradient_magnitude: float, magnitude at edge.
    """
    radii = np.asarray(radii, dtype=float)
    profile = np.asarray(profile, dtype=float)

    if len(radii) < 3:
        return {
            "edge_radius": float(radii[0]) if len(radii) > 0 else 0.0,
            "edge_index": 0,
            "gradient": np.zeros(len(radii)),
            "gradient_magnitude": 0.0,
        }

    # Smooth profile
    if smooth_window > 1 and len(profile) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        # Use 'same' with edge padding to avoid artifacts
        padded = np.pad(profile, smooth_window // 2, mode="edge")
        smoothed_padded = np.convolve(padded, kernel, mode="valid")
        smoothed = smoothed_padded[: len(profile)]
    else:
        smoothed = profile.copy()

    # Compute gradient (central difference for interior points)
    gradient = np.zeros(len(radii))
    for i in range(1, len(radii) - 1):
        dr = radii[i + 1] - radii[i - 1]
        if dr > 0:
            gradient[i] = (smoothed[i + 1] - smoothed[i - 1]) / dr

    # Find steepest gradient in the specified direction
    if direction == "decreasing":
        # Most negative gradient
        idx = int(np.argmin(gradient))
    else:
        # Most positive gradient
        idx = int(np.argmax(gradient))

    return {
        "edge_radius": float(radii[idx]),
        "edge_index": idx,
        "gradient": gradient,
        "gradient_magnitude": float(abs(gradient[idx])),
    }


def half_max_radius(radii, profile, baseline_fraction=0.5):
    """Find the radius where profile drops to a fraction of its range.

    For ZOI: the radius where density reaches 50% of the plateau
    level (measured from minimum to maximum).

    Args:
        radii: 1D array.
        profile: 1D array.
        baseline_fraction: float, fraction of range (0-1).
            0.5 = half-maximum (default).

    Returns:
        dict with:
            radius: float, interpolated half-max radius.
            index: int, nearest index in radii array.
            value_at_radius: float, profile value at this radius.
            profile_range: float, max - min of profile.
    """
    radii = np.asarray(radii, dtype=float)
    profile = np.asarray(profile, dtype=float)

    if len(radii) < 2:
        return {
            "radius": float(radii[0]) if len(radii) > 0 else 0.0,
            "index": 0,
            "value_at_radius": float(profile[0]) if len(profile) > 0 else 0.0,
            "profile_range": 0.0,
        }

    pmin = float(profile.min())
    pmax = float(profile.max())
    prange = pmax - pmin

    if prange < 1e-10:
        return {
            "radius": float(radii[-1]),
            "index": len(radii) - 1,
            "value_at_radius": float(profile[-1]),
            "profile_range": 0.0,
        }

    target = pmin + prange * baseline_fraction

    # Find where profile crosses the target (from high to low)
    # Scan from inner to outer radius
    crossing_idx = -1
    for i in range(len(profile) - 1):
        if (profile[i] >= target and profile[i + 1] < target) or (
            profile[i] <= target and profile[i + 1] > target
        ):
            crossing_idx = i
            break

    if crossing_idx >= 0:
        # Linear interpolation
        f = (target - profile[crossing_idx]) / (profile[crossing_idx + 1] - profile[crossing_idx])
        interp_radius = radii[crossing_idx] + f * (radii[crossing_idx + 1] - radii[crossing_idx])
        return {
            "radius": round(float(interp_radius), 4),
            "index": crossing_idx,
            "value_at_radius": round(float(target), 4),
            "profile_range": round(prange, 4),
        }

    # No crossing found — return the closest point
    idx = int(np.argmin(np.abs(profile - target)))
    return {
        "radius": float(radii[idx]),
        "index": idx,
        "value_at_radius": float(profile[idx]),
        "profile_range": round(prange, 4),
    }


def wavefront_radius(stack, center=None, threshold=None, n_bins=50, max_radius=None):
    """Track wavefront expansion radius over time.

    For each frame, measures the furthest radius where signal
    exceeds a threshold. Useful for tracking calcium waves,
    reaction fronts, or diffusion boundaries.

    Args:
        stack: 3D array (T, H, W).
        center: (y, x) tuple or None.
        threshold: float or None (auto: mean + 2*std of frame 0).
        n_bins: int, radial bins.
        max_radius: float or None.

    Returns:
        dict with:
            front_radii: 1D array, wavefront radius per frame.
            velocities: 1D array, radial velocity per frame (px/frame).
            mean_velocity: float, average expansion velocity.
            profiles: 2D array (T, n_bins), radial profiles.
            radii: 1D array, bin centers.
    """
    stack = np.asarray(stack, dtype=float)
    n_frames = stack.shape[0]
    h, w = stack.shape[1:]

    if center is None:
        center = (h / 2.0, w / 2.0)
    cy, cx = center

    if max_radius is None:
        max_radius = min(cy, h - cy, cx, w - cx)

    # Distance map
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    edges = np.linspace(0, max_radius, n_bins + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])

    if threshold is None:
        frame0 = stack[0]
        threshold = float(frame0.mean() + 2 * frame0.std())

    profiles = np.zeros((n_frames, n_bins))
    front_radii = np.zeros(n_frames)

    for t in range(n_frames):
        frame = stack[t]
        for i in range(n_bins):
            ring = (dist >= edges[i]) & (dist < edges[i + 1])
            if ring.sum() > 0:
                profiles[t, i] = float(frame[ring].mean())

        # Wavefront: furthest bin above threshold
        above = profiles[t] > threshold
        if above.any():
            last_above = np.where(above)[0][-1]
            front_radii[t] = float(radii[last_above])

    # Compute velocities
    velocities = np.zeros(n_frames)
    velocities[1:] = np.diff(front_radii)

    mean_vel = float(front_radii[-1] - front_radii[0]) / max(n_frames - 1, 1)

    return {
        "front_radii": front_radii,
        "velocities": velocities,
        "mean_velocity": round(mean_vel, 4),
        "profiles": profiles,
        "radii": radii,
    }


def annular_statistics(image, center=None, inner_radius=0, outer_radius=None, n_rings=5, mask=None):
    """Compute statistics in concentric annular regions.

    Returns per-ring statistics useful for characterizing radial
    structure in spheroids, colonies, or inhibition zones.

    Args:
        image: 2D array.
        center: (y, x) or None.
        inner_radius: float, innermost radius.
        outer_radius: float or None.
        n_rings: int, number of annular regions.
        mask: 2D bool array or None.

    Returns:
        dict with:
            rings: list of dicts with per-ring stats (mean, std, median,
                area_px, inner_r, outer_r).
            radii: 1D array, midpoint of each ring.
            means: 1D array, mean intensity per ring.
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape

    if center is None:
        center = (h / 2.0, w / 2.0)
    cy, cx = center

    if outer_radius is None:
        outer_radius = min(cy, h - cy, cx, w - cx)

    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    edges = np.linspace(inner_radius, outer_radius, n_rings + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])

    rings = []
    means_arr = np.zeros(n_rings)

    for i in range(n_rings):
        ring_mask = (dist >= edges[i]) & (dist < edges[i + 1])
        if mask is not None:
            ring_mask = ring_mask & mask

        n_px = int(ring_mask.sum())
        if n_px > 0:
            values = image[ring_mask]
            ring_mean = float(values.mean())
            ring_std = float(values.std())
            ring_median = float(np.median(values))
        else:
            ring_mean = 0.0
            ring_std = 0.0
            ring_median = 0.0

        means_arr[i] = ring_mean
        rings.append(
            {
                "mean": round(ring_mean, 4),
                "std": round(ring_std, 4),
                "median": round(ring_median, 4),
                "area_px": n_px,
                "inner_r": round(float(edges[i]), 2),
                "outer_r": round(float(edges[i + 1]), 2),
            }
        )

    return {
        "rings": rings,
        "radii": radii,
        "means": means_arr,
    }
