"""Batch processing for multi-position/multi-tile analysis.

Acquires images at multiple positions and runs analysis on each,
aggregating results. Reduces boilerplate for common multi-FOV workflows.

Functions:
    tile_and_analyze    -- Acquire grid of tiles and run analysis on each
    multi_position_measure -- Measure at pre-defined positions
    multichannel_scan   -- Multi-position, multi-channel MDA scan + analyze
    aggregate_results   -- Combine per-tile measurements
    measure_nuclear_expression -- Per-tile nuclear segmentation and intensity
    identify_hotspot    -- Find outlier tile(s) with elevated expression
"""

import numpy as np
from skimage.measure import label, regionprops
from useq import MDASequence


def tile_and_analyze(core, analyze_fn, grid=(2, 2), channel=None,
                     group=None, overlap=0.1, center=None):
    """Acquire a grid of tiles and analyze each.

    Args:
        core: CMMCorePlus instance.
        analyze_fn: callable(image) → dict, analysis to run per tile.
        grid: (rows, cols) grid dimensions.
        channel: str, channel config name.
        group: str or None, config group name. Auto-discovered if None.
        overlap: float, fractional overlap between tiles (0-0.5).
        center: (x, y) center of grid in world coords, or None for current.

    Returns:
        dict with:
            per_tile: list of dicts, one per tile with analysis results
                plus 'position' (x, y) and 'tile_index' (row, col).
            aggregate: dict, combined statistics from aggregate_results().
            n_tiles: int.
    """
    from src.core.hardware.core import run_events
    from src.core.hardware.config import resolve_brightfield_channel, resolve_channel_group

    group = resolve_channel_group(core, group)
    channel = resolve_brightfield_channel(core, channel) or 'brightfield'

    # Calculate tile positions
    pixel_size = core.getPixelSizeUm()
    if pixel_size <= 0:
        pixel_size = 1.0
    fov_size = core.getImageWidth() * pixel_size
    step = fov_size * (1.0 - overlap)

    rows, cols = grid
    if center is None:
        cx = core.getXPosition()
        cy = core.getYPosition()
    else:
        cx, cy = center

    positions = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - (cols - 1) / 2) * step
            y = cy + (r - (rows - 1) / 2) * step
            positions.append({'x': x, 'y': y})

    # Acquire tiles via MDA
    tiles = []
    seq = MDASequence(
        channels=[{'config': channel, 'group': group}],
        stage_positions=positions,
    )

    def on_frame(img, event):
        tiles.append(img.copy())

    run_events(core, list(seq), on_frame=on_frame)

    # Analyze each tile
    per_tile = []
    for i, (tile, pos) in enumerate(zip(tiles, positions)):
        result = analyze_fn(tile)
        result['position'] = (pos['x'], pos['y'])
        result['tile_index'] = (i // cols, i % cols)
        per_tile.append(result)

    # Aggregate
    agg = aggregate_results(per_tile)

    return {
        'per_tile': per_tile,
        'aggregate': agg,
        'n_tiles': len(per_tile),
    }


def multi_position_measure(core, positions, measure_fn, channel=None,
                           group=None):
    """Measure at specific pre-defined positions.

    Args:
        core: CMMCorePlus instance.
        positions: list of (x, y) tuples or dicts with 'x', 'y'.
        measure_fn: callable(image) → dict.
        channel: str, channel config name.
        group: str or None, config group name. Auto-discovered if None.

    Returns:
        dict with:
            measurements: list of dicts, one per position.
            aggregate: dict from aggregate_results().
            n_positions: int.
    """
    from src.core.hardware.core import run_events
    from src.core.hardware.config import resolve_brightfield_channel, resolve_channel_group

    group = resolve_channel_group(core, group)
    channel = resolve_brightfield_channel(core, channel) or 'brightfield'

    # Normalize positions
    pos_list = []
    for p in positions:
        if isinstance(p, (tuple, list)):
            pos_list.append({'x': p[0], 'y': p[1]})
        else:
            pos_list.append(p)

    # Acquire
    tiles = []
    seq = MDASequence(
        channels=[{'config': channel, 'group': group}],
        stage_positions=pos_list,
    )

    def on_frame(img, event):
        tiles.append(img.copy())

    run_events(core, list(seq), on_frame=on_frame)

    # Measure
    measurements = []
    for tile, pos in zip(tiles, pos_list):
        result = measure_fn(tile)
        result['position'] = (pos['x'], pos['y'])
        measurements.append(result)

    return {
        'measurements': measurements,
        'aggregate': aggregate_results(measurements),
        'n_positions': len(measurements),
    }


def multichannel_scan(core, positions, channels, analyze_fn=None,
                      group=None, exposure=50.0):
    """Multi-position, multi-channel MDA scan with optional per-position analysis.

    Runs a single MDASequence visiting each (x, y) position and snapping
    every channel in order. Groups frames by position and calls
    ``analyze_fn(ch_dict)`` where ``ch_dict`` is ``{channel_name: image}``
    for each position.

    Replaces the manual ``for pos in positions: move_to(...); for ch in
    channels: snap(...)`` loop that bypasses MDA. Per the session startup
    guidance, MDA is preferred over snap-loops for multi-channel acquisitions.

    Args:
        core: CMMCorePlus instance (or proxy).
        positions: List of (x, y) world-coordinate tuples.
        channels: List of channel config names (e.g. ['DAPI', 'bodipy-channel']).
        analyze_fn: Optional ``callable({name: image}) -> dict``. If given,
            called once per position; the result is stored in ``per_tile``.
        group: Optional channel group. Auto-discovered if None.
        exposure: Exposure (ms) applied to every channel.

    Returns:
        dict with:
            per_tile: list of per-position dicts. Each contains
                ``{'position': (x, y), 'channels': {name: image, ...}}``
                plus keys returned by ``analyze_fn`` (if supplied).
            n_tiles: int.
            aggregate: dict from aggregate_results over analyze_fn outputs
                (empty if analyze_fn is None).

    Example::

        r = multichannel_scan(
            core,
            positions=[(0, 0), (256, 0), (0, 256)],
            channels=['DAPI', 'bodipy-channel'],
            analyze_fn=lambda imgs: {
                'n_cells': count_nuclei(imgs['DAPI']),
                'droplets': count_droplets(imgs['bodipy-channel']),
            },
        )
    """
    from src.core.hardware.core import run_events
    from src.core.hardware.config import resolve_channel_group

    group = resolve_channel_group(core, group)

    pos_list = []
    for p in positions:
        if isinstance(p, (tuple, list)):
            pos_list.append({'x': p[0], 'y': p[1]})
        else:
            pos_list.append(p)

    ch_specs = []
    for ch in channels:
        spec = {'config': ch, 'exposure': exposure}
        if group is not None:
            spec['group'] = group
        ch_specs.append(spec)

    seq = MDASequence(channels=ch_specs, stage_positions=pos_list)

    # Bucket frames by position index
    buckets = [{} for _ in pos_list]

    def on_frame(img, event):
        p_idx = event.index.get('p', 0)
        c_idx = event.index.get('c', 0)
        ch_name = channels[c_idx]
        buckets[p_idx][ch_name] = img.copy()

    run_events(core, list(seq), on_frame=on_frame)

    per_tile = []
    for i, (pos, ch_imgs) in enumerate(zip(pos_list, buckets)):
        entry = {
            'position': (pos['x'], pos['y']),
            'tile_index': i,
            'channels': ch_imgs,
        }
        if analyze_fn is not None:
            entry.update(analyze_fn(ch_imgs))
        per_tile.append(entry)

    agg = aggregate_results(per_tile) if analyze_fn is not None else {}

    return {
        'per_tile': per_tile,
        'n_tiles': len(per_tile),
        'aggregate': agg,
    }


def aggregate_results(results):
    """Combine per-tile/per-position measurements.

    Collects all numeric values from result dicts and computes
    summary statistics (mean, std, median, total, n).

    Args:
        results: list of dicts from per-tile analysis.

    Returns:
        dict with per-key statistics:
            {key}_mean, {key}_std, {key}_median, {key}_total, {key}_n
        for each numeric key found in results.
    """
    if not results:
        return {}

    # Collect all numeric keys
    numeric_keys = set()
    for r in results:
        for k, v in r.items():
            if k in ('position', 'tile_index'):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_keys.add(k)

    agg = {}
    for key in sorted(numeric_keys):
        values = []
        for r in results:
            if key in r:
                v = r[key]
                if isinstance(v, (int, float, np.integer, np.floating)):
                    values.append(float(v))

        if values:
            arr = np.array(values)
            agg[f'{key}_mean'] = round(float(arr.mean()), 4)
            agg[f'{key}_std'] = round(float(arr.std()), 4)
            agg[f'{key}_median'] = round(float(np.median(arr)), 4)
            agg[f'{key}_total'] = round(float(arr.sum()), 4)
            agg[f'{key}_n'] = len(values)

    return agg


def measure_nuclear_expression(image, threshold=30, min_area=20):
    """Measure nuclear expression statistics in a fluorescence image.

    Segments bright nuclei via thresholding and measures per-cell intensity.
    Designed as an analyze_fn for tile_and_analyze().

    Args:
        image: 2D fluorescence image (bright nuclei on dark background).
        threshold: Intensity threshold for nuclear segmentation.
        min_area: Minimum nucleus area in pixels.

    Returns:
        dict with:
            n_cells: number of detected nuclei.
            nuclear_mean: mean intensity of all nuclei.
            nuclear_max: max per-cell mean intensity.
            nuclear_min: min per-cell mean intensity.
            nuclear_fraction: fraction of image occupied by nuclei.
            image_mean: overall image mean intensity.
    """
    img = np.asarray(image, dtype=np.float64)
    binary = img > threshold
    labeled = label(binary)
    props = regionprops(labeled, intensity_image=img)
    cells = [p for p in props if p.area >= min_area]

    if not cells:
        return {
            'n_cells': 0,
            'nuclear_mean': 0.0,
            'nuclear_max': 0.0,
            'nuclear_min': 0.0,
            'nuclear_fraction': 0.0,
            'image_mean': float(img.mean()),
        }

    intensities = [float(p.intensity_mean) for p in cells]
    return {
        'n_cells': len(cells),
        'nuclear_mean': float(np.mean(intensities)),
        'nuclear_max': float(np.max(intensities)),
        'nuclear_min': float(np.min(intensities)),
        'nuclear_fraction': float(binary.sum() / img.size),
        'image_mean': float(img.mean()),
    }


def identify_hotspot(per_tile, key='nuclear_mean', z_threshold=1.5):
    """Identify tile(s) with elevated expression relative to the population.

    Uses a z-score approach: tiles with expression > z_threshold standard
    deviations above the median are flagged as hotspots.

    Args:
        per_tile: list of dicts from tile_and_analyze(), each containing
            the measurement key and 'position'.
        key: measurement key to use for comparison (default 'nuclear_mean').
        z_threshold: number of MAD-scaled deviations above median to flag
            as hotspot (default 1.5).

    Returns:
        dict with:
            hotspot_indices: list of indices of hotspot tiles.
            hotspot_positions: list of (x, y) positions of hotspot tiles.
            hotspot_centroid: (x, y) intensity-weighted centroid of hotspot,
                or None if no hotspot found.
            hotspot_mean: mean expression in hotspot tiles.
            baseline_mean: mean expression in non-hotspot tiles.
            fold_change: hotspot_mean / baseline_mean.
            all_values: list of expression values per tile.
    """
    values = [float(t.get(key, 0)) for t in per_tile]
    arr = np.array(values)

    if len(arr) < 2 or arr.std() == 0:
        return {
            'hotspot_indices': [],
            'hotspot_positions': [],
            'hotspot_mean': float(arr.mean()) if len(arr) > 0 else 0.0,
            'baseline_mean': float(arr.mean()) if len(arr) > 0 else 0.0,
            'fold_change': 1.0,
            'all_values': values,
        }

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad == 0:
        mad = float(arr.std())

    hotspot_idx = []
    for i, v in enumerate(values):
        if mad > 0 and (v - median) / mad > z_threshold:
            hotspot_idx.append(i)

    # Fallback: if no hotspot found by z-score, pick the maximum
    if not hotspot_idx and len(values) >= 2:
        max_idx = int(np.argmax(arr))
        if arr[max_idx] > median * 1.1:
            hotspot_idx = [max_idx]

    hotspot_positions = [per_tile[i].get('position', (0, 0)) for i in hotspot_idx]
    hotspot_vals = [values[i] for i in hotspot_idx]
    baseline_vals = [v for i, v in enumerate(values) if i not in hotspot_idx]

    hotspot_mean = float(np.mean(hotspot_vals)) if hotspot_vals else 0.0
    baseline_mean = float(np.mean(baseline_vals)) if baseline_vals else hotspot_mean
    fold_change = hotspot_mean / baseline_mean if baseline_mean > 0 else 0.0

    # Intensity-weighted centroid for precise hotspot localization
    weighted_centroid = None
    if hotspot_idx:
        positions = []
        weights = []
        for i in hotspot_idx:
            pos = per_tile[i].get('position', None)
            if pos is not None:
                positions.append(pos)
                weights.append(values[i])
        if positions:
            pos_arr = np.array(positions)
            w_arr = np.array(weights)
            w_sum = w_arr.sum()
            if w_sum > 0:
                weighted_centroid = tuple(float(v) for v in (w_arr @ pos_arr) / w_sum)

    return {
        'hotspot_indices': hotspot_idx,
        'hotspot_positions': hotspot_positions,
        'hotspot_centroid': weighted_centroid,
        'hotspot_mean': hotspot_mean,
        'baseline_mean': baseline_mean,
        'fold_change': fold_change,
        'all_values': values,
    }
