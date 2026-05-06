"""Intelligent sample scouting and experiment planning.

Provides functions to characterize the microscope setup BEFORE committing
to an analysis strategy. Prevents common errors:
- channel_scout(): snap all channels, identify which have signal
- sample_survey(): find sample extent and suggest FOV positions
- find_sample(): search the world for sample material (coarse grid)
- experiment_protocol(): run multi-condition experiments systematically
"""

import numpy as np
from useq import MDAEvent

from ..hardware.config import get_config
from ..hardware.core import get_position, move_to, run_events, snap


def channel_scout(core, channels=None):
    """Snap each available channel and characterize the signal.

    Call this BEFORE choosing which channel to analyze. Prevents the
    costly error of analyzing the wrong channel (ch421 pattern).

    Args:
        core: Connected microscope core.
        channels: List of channel names to test. If None, discovers all
            available channels from the config group.

    Returns:
        dict mapping channel_name -> {
            'has_signal': bool (max > 3*noise_floor),
            'max': int,
            'mean': float,
            'mean_bright': float (mean of pixels > 5th percentile),
            'snr': float (estimated signal-to-noise),
            'noise_floor': float (median of dim pixels),
        }
    """
    cfg = get_config(core)

    if channels is None:
        if cfg.channel_group:
            channels = list(core.getAvailableConfigs(cfg.channel_group))
        else:
            channels = ["brightfield"]

    # Acquire all channels via MDA events
    events = [MDAEvent(channel={"config": ch}, exposure=50.0) for ch in channels]
    frames = run_events(core, events)

    results = {}
    for (img, _event), ch in zip(frames, channels, strict=False):
        img_f = img.astype(np.float64)

        # Noise floor: median of bottom 50% of pixels
        sorted_vals = np.sort(img_f.ravel())
        noise_floor = float(np.median(sorted_vals[: len(sorted_vals) // 2]))

        # Signal characterization
        mx = int(img.max())
        mn = float(img_f.mean())

        # Mean of bright pixels (above noise floor + small margin)
        threshold = max(noise_floor + 5, np.percentile(img_f, 5))
        bright_mask = img_f > threshold
        mean_bright = float(img_f[bright_mask].mean()) if bright_mask.any() else mn

        # SNR estimate
        signal = mean_bright - noise_floor
        noise = float(np.std(sorted_vals[: len(sorted_vals) // 2])) + 1e-6
        snr = signal / noise

        # Has signal: max is significantly above noise
        has_signal = mx > noise_floor * 3 + 10

        results[ch] = {
            "has_signal": has_signal,
            "max": mx,
            "mean": round(mn, 1),
            "mean_bright": round(mean_bright, 1),
            "snr": round(snr, 1),
            "noise_floor": round(noise_floor, 1),
        }

    return results


def sample_survey(core, channel="brightfield", n_tiles=5, tile_spacing=None):
    """Survey the sample area to find the extent and plan FOV positions.

    Snaps images at the origin and surrounding positions to find where
    the sample is. Returns a bounding box and suggested positions for
    detailed imaging.

    Args:
        core: Connected microscope core.
        channel: Channel to use for scouting (default: brightfield).
        n_tiles: Number of tiles per axis (n_tiles x n_tiles grid).
            Use 1 for single-FOV, 3 for 3x3 grid, 5 for 5x5 grid.
        tile_spacing: Spacing between tiles in world units (um).
            If None, uses FOV size (no overlap).

    Returns:
        dict with:
            'positions': list of (x, y) world positions with sample
            'empty_positions': list of (x, y) positions without sample
            'sample_bbox': (x_min, y_min, x_max, y_max) in world coords
            'center': (x, y) estimated sample center
            'images': dict mapping (x, y) -> image array
    """
    cfg = get_config(core)
    fov = cfg.image_width * cfg.pixel_size_um if cfg.pixel_size_um > 0 else 512

    if tile_spacing is None:
        tile_spacing = fov * 0.9  # 10% overlap

    # Generate grid positions centered on current location
    cx, cy = get_position(core)
    half = (n_tiles - 1) / 2
    grid_positions = []
    for iy in range(n_tiles):
        for ix in range(n_tiles):
            x = cx + (ix - half) * tile_spacing
            y = cy + (iy - half) * tile_spacing
            grid_positions.append((x, y))

    # Acquire all positions via MDA events
    events = [
        MDAEvent(x_pos=float(x), y_pos=float(y), channel={"config": channel}, exposure=50.0)
        for x, y in grid_positions
    ]
    frames = run_events(core, events)

    images = {}
    sample_positions = []
    empty_positions = []

    from scipy.ndimage import uniform_filter

    for (img, _event), (x, y) in zip(frames, grid_positions, strict=False):
        images[(x, y)] = img

        # Detect if sample is present: high variance regions
        img_f = img.astype(np.float64)

        # Check for structure: local variance
        local_mean = uniform_filter(img_f, size=20)
        local_var = uniform_filter((img_f - local_mean) ** 2, size=20)
        has_structure = np.mean(local_var) > 10  # significant texture

        if has_structure:
            sample_positions.append((x, y))
        else:
            empty_positions.append((x, y))

    # Compute bounding box
    if sample_positions:
        xs = [p[0] for p in sample_positions]
        ys = [p[1] for p in sample_positions]
        bbox = (min(xs) - fov / 2, min(ys) - fov / 2, max(xs) + fov / 2, max(ys) + fov / 2)
        center = (np.mean(xs), np.mean(ys))
    else:
        # No sample found — return origin
        bbox = (cx - fov / 2, cy - fov / 2, cx + fov / 2, cy + fov / 2)
        center = (cx, cy)

    # Return to starting position
    move_to(core, cx, cy)

    return {
        "positions": sample_positions,
        "empty_positions": empty_positions,
        "sample_bbox": bbox,
        "center": center,
        "images": images,
    }


def find_sample(
    core,
    channel="brightfield",
    detect_fn=None,
    search_range=2048,
    step=None,
    start=None,
    stop_on_first=True,
):
    """Search the world for sample material with an expanding spiral.

    Useful when the sample location is unknown and you need to search a
    large stage area. Starts from the center of the search range and
    expands outward, checking each position for signal.

    For brightfield: looks for dark objects (min well below background).
    For fluorescence: looks for bright signal above noise floor.
    Accepts a custom detect_fn for specialized detection.

    Args:
        core: Connected microscope core.
        channel: Channel to snap at each position.
        detect_fn: Optional callable(image) -> bool. Returns True if the
            image contains sample. If None, uses built-in heuristic.
        search_range: Total range to search in each axis (world units).
            Searches from start to start + search_range in x and y.
        step: Step size between positions (world units).
            Default: FOV size (no overlap).
        start: (x, y) starting corner of search area. If None, uses
            (0, 0) as the origin.
        stop_on_first: If True, returns immediately when sample is found.
            If False, searches entire grid to find all sample positions.

    Returns:
        dict with:
            'found': bool — whether any sample was detected
            'positions': list of (x, y) positions where sample was found,
                sorted by signal strength (strongest first)
            'best_position': (x, y) strongest signal position, or None
            'positions_checked': number of positions visited
            'signal_strengths': list of float signal values for each
                position in 'positions'
    """
    cfg = get_config(core)
    fov = cfg.image_width * cfg.pixel_size_um if cfg.pixel_size_um > 0 else 512

    if step is None:
        step = fov

    if start is None:
        start = (0.0, 0.0)

    x0, y0 = float(start[0]), float(start[1])

    # Build grid positions in spiral order from center outward
    nx = max(1, int(search_range / step))
    ny = max(1, int(search_range / step))
    cx_idx, cy_idx = nx / 2.0, ny / 2.0

    grid = []
    for ix in range(nx):
        for iy in range(ny):
            x = x0 + (ix + 0.5) * step
            y = y0 + (iy + 0.5) * step
            dist_from_center = ((ix - cx_idx) ** 2 + (iy - cy_idx) ** 2) ** 0.5
            grid.append((x, y, dist_from_center))

    # Sort by distance from center (spiral-like order)
    grid.sort(key=lambda g: g[2])

    found_positions = []
    signal_strengths = []
    positions_checked = 0

    for x, y, _ in grid:
        move_to(core, x, y)
        img = snap(core, channel=channel)
        positions_checked += 1

        if detect_fn is not None:
            has_sample = detect_fn(img)
            strength = 1.0 if has_sample else 0.0
        else:
            has_sample, strength = _default_sample_detect(img, channel)

        if has_sample:
            found_positions.append((x, y))
            signal_strengths.append(strength)
            if stop_on_first:
                break

    # Sort by signal strength (strongest first)
    if found_positions:
        pairs = sorted(zip(signal_strengths, found_positions, strict=False), reverse=True)
        signal_strengths = [s for s, _ in pairs]
        found_positions = [p for _, p in pairs]
        best = found_positions[0]
    else:
        best = None

    return {
        "found": len(found_positions) > 0,
        "positions": found_positions,
        "best_position": best,
        "positions_checked": positions_checked,
        "signal_strengths": signal_strengths,
    }


def _default_sample_detect(image, channel="brightfield"):
    """Default sample detection heuristic.

    For brightfield: sample is dark against bright background.
    Signal strength = how much darker the min region is vs background.

    For fluorescence: sample has bright features above noise.
    Signal strength = max / noise_floor ratio.

    Returns:
        (has_sample: bool, strength: float)
    """
    img = image.astype(np.float64)
    is_bf = "bright" in channel.lower() or channel.lower() in ("bf", "brightfield")

    if is_bf:
        bg = float(np.percentile(img, 90))
        low = float(np.percentile(img, 2))
        contrast = bg - low
        # Significant dark object: contrast > 30 intensity units
        return bool(contrast > 30), contrast
    else:
        noise_floor = float(np.median(img))
        mx = float(img.max())
        # Signal: max is >3x noise floor + margin
        ratio = mx / (noise_floor + 1e-6)
        return bool(ratio > 3.0 and mx > 20), ratio


def experiment_protocol(
    core, conditions, measure_fn, settle_time=1.0, n_repeats=1, baseline_condition=None
):
    """Run a multi-condition experiment systematically.

    Sets each condition (device state), waits for settling, runs the
    measurement function, and collects results. Generalizes the pattern
    used in dose-response, temperature sweep, and drug washout experiments.

    Args:
        core: Connected microscope core.
        conditions: List of dicts, each with:
            'label': str — human-readable label
            'device': str — device name (e.g., 'Anesthesia', 'Temperature')
            'state': int — state index to set
            Optional 'settle_time': float — override per-condition settle time
        measure_fn: Callable(core) -> dict — measurement function.
            Should return a dict of measured values.
        settle_time: Default time (seconds) to wait after setting condition.
        n_repeats: Number of repeat measurements per condition.
        baseline_condition: If given, index of the baseline condition for
            normalization (e.g., 0 for first condition).

    Returns:
        list of dicts, each with:
            'label': condition label
            'condition': the original condition dict
            'measurements': list of measurement results (from measure_fn)
            'mean': dict of mean values across repeats
    """
    import time

    results = []

    for cond in conditions:
        # Set device state
        device = cond["device"]
        state = cond["state"]
        label = cond.get("label", f"{device}={state}")
        wait = cond.get("settle_time", settle_time)

        core.setState(device, state)
        time.sleep(wait)

        # Run measurements
        measurements = []
        for _ in range(n_repeats):
            meas = measure_fn(core)
            measurements.append(meas)

        # Compute means across repeats
        mean_values = {}
        if measurements:
            for key in measurements[0]:
                vals = [m[key] for m in measurements if isinstance(m.get(key), (int, float))]
                if vals:
                    mean_values[key] = float(np.mean(vals))

        results.append(
            {
                "label": label,
                "condition": cond,
                "measurements": measurements,
                "mean": mean_values,
            }
        )

    # Normalize to baseline if requested
    if baseline_condition is not None and 0 <= baseline_condition < len(results):
        baseline = results[baseline_condition]["mean"]
        for r in results:
            r["normalized"] = {}
            for key, val in r["mean"].items():
                if key in baseline and baseline[key] != 0:
                    r["normalized"][key] = val / baseline[key]

    return results
