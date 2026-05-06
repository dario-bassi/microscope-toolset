"""Multi-scale microscopy coordinator.

Wraps existing hardware, detection, and analysis functions into
structured multi-scale workflows. Addresses the #1 recurring failure:
forgetting to scout at low magnification before zooming in.

Key functions:
    suggest_magnification -- Recommend objective based on object size
    scale_params          -- Detection parameters scaled for magnification
    overview_first        -- Mandatory 10x overview with cell count/FOV map
    multi_scale_measure   -- Orchestrated multi-scale experiment
    multi_position_events -- Generate MDA events for multi-position acquisition

Typical usage:
    overview = overview_first(core)          # always starts at 10x
    targets = select_rois(overview)          # pick interesting regions
    results = multi_scale_measure(core, targets, zoom_mag=40, detect_fn=...)
"""

import numpy as np

from ..hardware.config import _mag_to_pixel_size, get_config
from ..hardware.core import (
    get_position,
    move_to,
    pixel_to_world,
    set_objective,
    snap,
)

# ---------------------------------------------------------------------------
# Magnification selection
# ---------------------------------------------------------------------------

# Expected object diameters (µm) by sample type
OBJECT_SIZES = {
    "bacteria": (0.5, 3),
    "yeast": (3, 8),
    "mammalian": (10, 30),
    "neuron_soma": (10, 25),
    "tissue": (5, 15),  # nuclei in tissue
    "colony": (50, 500),
    "worm": (50, 1000),
    "spheroid": (100, 500),
}

# Available standard magnifications, ascending
STANDARD_MAGS = [10, 20, 40, 100]


def suggest_magnification(object_size_um, target_pixels=40, available_mags=None):
    """Suggest the best objective magnification for an object size.

    Picks the lowest magnification where the object is at least
    ``target_pixels`` across. This maximizes FOV while keeping objects
    resolvable.

    Args:
        object_size_um: Expected object diameter in micrometers.
        target_pixels: Minimum desired object size in pixels.
            Default 40 (good for detection and measurement).
        available_mags: List of available magnifications.
            Default: [10, 20, 40, 100].

    Returns:
        dict with:
            mag: Recommended magnification.
            object_px: Expected object size in pixels at that mag.
            pixel_size_um: Pixel size at that mag.
            fov_um: FOV width at that mag.
    """
    mags = available_mags or STANDARD_MAGS

    for mag in sorted(mags):
        pixel_size = _mag_to_pixel_size(mag)
        object_px = object_size_um / pixel_size
        if object_px >= target_pixels:
            return {
                "mag": mag,
                "object_px": round(object_px, 1),
                "pixel_size_um": pixel_size,
                "fov_um": 512 * pixel_size,
            }

    # If even highest mag is insufficient, use it anyway
    highest = max(mags)
    pixel_size = _mag_to_pixel_size(highest)
    return {
        "mag": highest,
        "object_px": round(object_size_um / pixel_size, 1),
        "pixel_size_um": pixel_size,
        "fov_um": 512 * pixel_size,
    }


def scale_params(mag, sample_type="mammalian"):
    """Return detection parameters scaled for a magnification.

    Provides sensible defaults for threshold, minimum area, and edge
    margin that account for how objects appear at different magnifications.

    Args:
        mag: Objective magnification (10, 20, 40, 100).
        sample_type: One of OBJECT_SIZES keys. Affects area thresholds.

    Returns:
        dict with:
            threshold_sigma: Suggested threshold in units of background std.
            min_area_px: Minimum connected-component area.
            max_area_px: Maximum connected-component area.
            edge_margin_px: Edge exclusion margin in pixels.
            pixel_size_um: Pixel size in micrometers.
    """
    pixel_size = _mag_to_pixel_size(mag)
    size_range = OBJECT_SIZES.get(sample_type, (10, 30))
    min_diam_um, max_diam_um = size_range

    # Convert to pixel areas (area = pi * (d/2)^2, approximate as d^2 * 0.75)
    min_area_px = max(4, int((min_diam_um / pixel_size) ** 2 * 0.5))
    max_area_px = max(100, int((max_diam_um / pixel_size) ** 2 * 1.5))

    # Edge margin: 3px at 40x, scale proportionally
    edge_margin = max(3, int(3 * (40 / mag)))

    # Threshold: lower at high mag (more contrast), higher at low mag (noise)
    if mag >= 40:
        threshold_sigma = 3.0
    elif mag >= 20:
        threshold_sigma = 2.5
    else:
        threshold_sigma = 2.0

    return {
        "threshold_sigma": threshold_sigma,
        "min_area_px": min_area_px,
        "max_area_px": max_area_px,
        "edge_margin_px": edge_margin,
        "pixel_size_um": pixel_size,
    }


# ---------------------------------------------------------------------------
# Overview workflow
# ---------------------------------------------------------------------------


def overview_first(core, channel=None, detect_fn=None, overview_mag=10):
    """Snap a mandatory 10x overview before any high-mag work.

    This enforces the "always scout at low magnification first" pattern.
    Returns the overview image along with basic statistics.

    Args:
        core: Microscope core.
        channel: Optional channel for the overview snap.
        detect_fn: Optional callable(image) → list of (y, x) centroids.
            If provided, runs detection on the overview.
        overview_mag: Magnification for overview (default 10).

    Returns:
        dict with:
            image: The overview image (np.ndarray).
            mag: Magnification used.
            position: (x, y) stage position.
            fov_um: FOV width in micrometers.
            pixel_size_um: Pixel size.
            mean_intensity: Mean image intensity.
            has_signal: True if image has significant content.
            centroids_world: World-coordinate centroids (if detect_fn).
            cell_count: Number of detected cells (if detect_fn).
    """
    # Switch to overview mag
    set_objective(core, overview_mag)
    cfg = get_config(core)

    # Snap
    img = snap(core, channel=channel)
    pos = get_position(core)
    pixel_size = cfg.pixel_size_um
    fov_um = cfg.fov_width_um

    # Basic stats
    mean_int = float(img.mean())
    img_std = float(img.std())
    has_signal = img_std > 5  # more than flat background

    result = {
        "image": img,
        "mag": overview_mag,
        "position": pos,
        "fov_um": fov_um,
        "pixel_size_um": pixel_size,
        "mean_intensity": mean_int,
        "has_signal": has_signal,
    }

    # Optional detection
    if detect_fn is not None:
        centroids_px = detect_fn(img)
        if centroids_px is not None and len(centroids_px) > 0:
            centroids_px = np.asarray(centroids_px)
            # Convert to world coordinates
            centroids_world = []
            for c in centroids_px:
                y, x = c[0], c[1]
                wx, wy = pixel_to_world(x, y, pos[0], pos[1], config=cfg)
                centroids_world.append([wx, wy])
            centroids_world = np.array(centroids_world)
            result["centroids_world"] = centroids_world
            result["cell_count"] = len(centroids_world)
        else:
            result["centroids_world"] = np.empty((0, 2))
            result["cell_count"] = 0
    else:
        result["centroids_world"] = None
        result["cell_count"] = None

    return result


# ---------------------------------------------------------------------------
# Multi-scale measurement
# ---------------------------------------------------------------------------


def multi_scale_measure(
    core,
    targets,
    zoom_mag=40,
    detect_fn=None,
    measure_fn=None,
    channel=None,
    return_to_overview=True,
):
    """Visit target positions at high magnification and measure.

    Args:
        core: Microscope core.
        targets: List of (x, y) world-coordinate positions to visit.
        zoom_mag: Magnification for detailed measurement.
        detect_fn: Optional callable(image) → list of centroids at zoom.
        measure_fn: Optional callable(image) → dict of measurements.
        channel: Optional channel for zoom acquisition.
        return_to_overview: If True, switch back to 10x after done.

    Returns:
        list of dicts, one per target, each with:
            position: (x, y) world coordinates.
            image: The zoom image.
            mag: Magnification used.
            detections: Centroids from detect_fn (if provided).
            measurements: Results from measure_fn (if provided).
    """
    results = []

    # Switch to zoom mag
    set_objective(core, zoom_mag)
    cfg = get_config(core)
    pixel_size = cfg.pixel_size_um

    for target in targets:
        x, y = float(target[0]), float(target[1])
        move_to(core, x, y)

        img = snap(core, channel=channel)

        entry = {
            "position": (x, y),
            "image": img,
            "mag": zoom_mag,
            "pixel_size_um": pixel_size,
        }

        if detect_fn is not None:
            centroids = detect_fn(img)
            entry["detections"] = centroids if centroids is not None else []
        else:
            entry["detections"] = None

        if measure_fn is not None:
            measurements = measure_fn(img)
            entry["measurements"] = measurements
        else:
            entry["measurements"] = None

        results.append(entry)

    if return_to_overview:
        set_objective(core, 10)

    return results


def validate_object_size(object_size_um, mag, min_pixels=10, ideal_pixels=30):
    """Check if an object is appropriately sized for a magnification.

    Args:
        object_size_um: Object diameter in micrometers.
        mag: Current magnification.
        min_pixels: Minimum acceptable size in pixels.
        ideal_pixels: Ideal size in pixels.

    Returns:
        dict with:
            object_px: Object size in pixels.
            status: 'too_small', 'ok', 'too_large', or 'use_lower_mag'.
            message: Human-readable recommendation.
            suggested_mag: Recommended magnification if current is not ideal.
    """
    pixel_size = _mag_to_pixel_size(mag)
    object_px = object_size_um / pixel_size

    suggestion = suggest_magnification(object_size_um, target_pixels=ideal_pixels)

    if object_px < min_pixels:
        return {
            "object_px": round(object_px, 1),
            "status": "too_small",
            "message": f"Object is {object_px:.0f}px at {mag}x — need higher mag",
            "suggested_mag": suggestion["mag"],
        }
    elif object_px > 300:
        return {
            "object_px": round(object_px, 1),
            "status": "use_lower_mag",
            "message": f"Object is {object_px:.0f}px at {mag}x — could use lower mag",
            "suggested_mag": suggestion["mag"],
        }
    else:
        return {
            "object_px": round(object_px, 1),
            "status": "ok",
            "message": f"Object is {object_px:.0f}px at {mag}x — good resolution",
            "suggested_mag": mag,
        }


def multi_position_events(positions, channels, exposure=50.0, z_pos=None, channel_group=None):
    """Generate MDA events for multi-position, multi-channel acquisition.

    Creates a flat list of MDAEvent objects visiting each position with
    each channel. Use with ``run_events(core, events)`` for structured
    multi-position acquisition instead of manual snap loops.

    Args:
        positions: list of (x, y) world-coordinate tuples.
        channels: list of channel config names (e.g. ['brightfield', 'GFP']).
        exposure: float or list of float, exposure time(s) per channel.
        z_pos: float or None, optional Z position for all events.
        channel_group: str or None, channel group name.

    Returns:
        list of MDAEvent objects, ordered position-first then channel.
    """
    from useq import MDAEvent

    if isinstance(exposure, (int, float)):
        exposures = [float(exposure)] * len(channels)
    else:
        exposures = [float(e) for e in exposure]

    events = []
    for x, y in positions:
        for ch, exp in zip(channels, exposures, strict=False):
            kwargs = {
                "x_pos": float(x),
                "y_pos": float(y),
                "channel": {"config": ch},
                "exposure": exp,
            }
            if channel_group is not None:
                kwargs["channel"] = {"config": ch, "group": channel_group}
            if z_pos is not None:
                kwargs["z_pos"] = float(z_pos)
            events.append(MDAEvent(**kwargs))

    return events
