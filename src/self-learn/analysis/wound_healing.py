"""Wound healing / scratch assay and laser ablation analysis.

Quantifies wound closure from timelapse images of cell monolayers
with a scratch/wound gap, and also laser ablation wound healing.

Functions (scratch assay):
    detect_wound       -- Find wound region in a single frame
    measure_wound_gap  -- Measure wound width at multiple positions (chord-width)
    measure_wound_gap_by_nearest_nucleus -- Distance from FOV centre to nearest
                          real tissue nucleus (debris-robust leading-edge proxy)
    measure_wound_gap_longest_zero_run -- Column-wise longest-zero-run metric
                          (mid-band of rows, 50% threshold; preferred when the
                          metric must compare migration *rates* across conditions
                          rather than absolute gap widths)
    wound_closure_rate -- Compute closure rate from timelapse
    migration_speed    -- Estimate cell migration speed from wound edges
    analyze_scratch_assay -- Full pipeline from timelapse stack

Functions (laser ablation wound healing):
    segment_cells_bf        -- detect cells in brightfield/fluorescence
    define_wound_region     -- create binary mask for wound region
    count_cells_in_region   -- count cells inside region mask
    track_wound_repopulation -- measure per-frame cell count in wound
    analyze_wound_healing   -- full analysis with clearance + healing
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label
from skimage import filters, morphology
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


# ===========================================================================
# Scratch assay functions (traditional wound healing / migration)
# ===========================================================================

def detect_wound(image, orientation='horizontal', min_gap_width=10):
    """Detect wound/scratch region in a monolayer image.

    The wound appears as a bright (cell-free) stripe in a darker
    (cell-covered) monolayer, or vice versa. Works for both
    brightfield and fluorescence.

    Args:
        image: 2D array, single frame of wound healing assay.
        orientation: 'horizontal' or 'vertical', direction of the
            scratch. 'horizontal' means the gap runs left-right.
        min_gap_width: int, minimum wound width in pixels.

    Returns:
        dict with:
            wound_mask: 2D bool array, True in wound region.
            wound_fraction: float, fraction of image that is wound.
            gap_profile: 1D array, wound width along the scratch.
            mean_gap: float, mean wound width in pixels.
            edges: (2D array, 2D array), leading edge positions.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    smoothed = ndimage.gaussian_filter(image, sigma=3)

    if orientation == 'horizontal':
        profile = smoothed.mean(axis=1)
        axis = 0
    else:
        profile = smoothed.mean(axis=0)
        axis = 1

    thresh = filters.threshold_otsu(smoothed)
    edge_mean = np.mean([profile[:10].mean(), profile[-10:].mean()])
    center_mean = profile[len(profile) // 3: 2 * len(profile) // 3].mean()

    if center_mean > edge_mean:
        wound_mask = smoothed > thresh
    else:
        wound_mask = smoothed < thresh

    wound_mask = ndimage.binary_fill_holes(wound_mask)
    wound_mask = morphology.remove_small_objects(
        wound_mask, max_size=min_gap_width ** 2)

    if orientation == 'horizontal':
        gap_profile = wound_mask.sum(axis=0).astype(float)
    else:
        gap_profile = wound_mask.sum(axis=1).astype(float)

    mean_gap = float(gap_profile.mean())
    wound_fraction = wound_mask.sum() / wound_mask.size

    if orientation == 'horizontal':
        top_edge = np.full(image.shape[1], np.nan)
        bottom_edge = np.full(image.shape[1], np.nan)
        for col in range(image.shape[1]):
            wound_rows = np.where(wound_mask[:, col])[0]
            if len(wound_rows) >= min_gap_width:
                top_edge[col] = wound_rows[0]
                bottom_edge[col] = wound_rows[-1]
        edges = (top_edge, bottom_edge)
    else:
        left_edge = np.full(image.shape[0], np.nan)
        right_edge = np.full(image.shape[0], np.nan)
        for row in range(image.shape[0]):
            wound_cols = np.where(wound_mask[row, :])[0]
            if len(wound_cols) >= min_gap_width:
                left_edge[row] = wound_cols[0]
                right_edge[row] = wound_cols[-1]
        edges = (left_edge, right_edge)

    return {
        'wound_mask': wound_mask,
        'wound_fraction': round(float(wound_fraction), 4),
        'gap_profile': gap_profile,
        'mean_gap': round(mean_gap, 2),
        'edges': edges,
    }


def measure_wound_gap_by_nearest_nucleus(image, min_nucleus_area=200):
    """Gap metric that's robust to in-wound debris and front fragmentation.

    Computes the distance (px) from the FOV centre to the nearest real
    tissue-nucleus centroid after Otsu + area filter. As the wound closes,
    the nearest nucleus gets closer to the centre — slope of this metric
    over time is a clean leading-edge closure rate.

    Preferred over ``measure_wound_gap`` when:
      - wound has small debris blobs flickering in/out of the mask,
      - the gap geometry is irregular (not a straight stripe),
      - you want a metric that responds to leading-edge advancement
        rather than total chord-width (the latter is dominated by the
        wider part of the wound, not where cells are actually moving).

    Args:
        image: 2D array (nuclear channel, e.g. DAPI).
        min_nucleus_area: int, reject blobs smaller than this (px) — this
            is what filters wound debris out of the metric. Real tissue
            nuclei are >200 px in typical 10x / 1 µm-px acquisitions.

    Returns:
        float: distance from (H/2, W/2) to the nearest surviving nucleus
               centroid, in pixels. Returns 0.0 if no nuclei pass filter.

    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    thresh = filters.threshold_otsu(img)
    mask = morphology.remove_small_objects(img > thresh,
                                            max_size=min_nucleus_area - 1)
    labeled = ndimage.label(mask)[0]
    props = regionprops(labeled)
    if not props:
        return 0.0
    H, W = img.shape
    cy, cx = H / 2, W / 2
    dists = [np.hypot(p.centroid[0] - cy, p.centroid[1] - cx) for p in props]
    return float(min(dists))


def measure_wound_gap_longest_zero_run(image, mid_band_rows=80,
                                         low_fraction=0.50):
    """Wound-width metric via column-wise longest sub-threshold run.

    On a mid-band of rows (where a vertical wound is clearest), compute
    column-mean intensities → threshold at
    ``low + low_fraction * (high - low)`` → return the length of the
    longest contiguous below-threshold run. This length is the wound
    width in pixels along the mid-band.

    Compared to ``measure_wound_gap`` (which averages gap *profile*),
    this metric is more robust to:
      - Edge-protrusion asymmetry (the longest run tracks whichever
        side gapped longest, not an average that washes out shifts).
      - Wound-interior debris (brief bright spikes don't split the
        run as long as they sit above the 50% threshold).

    Useful when the nearest-nucleus metric gets confounded by
    drug-induced edge protrusion (the leading edge can advance further
    than the population-mean migration rate).

    Args:
        image: 2D or 3D array. If 3D, averaged to grayscale.
        mid_band_rows: int, number of rows centred on image centre
            to average. Scale with wound length.
        low_fraction: float in [0, 1], threshold fraction between low
            and high column-mean. 0.5 = mid-point; 0.3 = closer to
            lawn level (smaller wound width).

    Returns:
        int, longest sub-threshold column run (= wound width in px).
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    H, W = img.shape
    r0 = max(0, H // 2 - mid_band_rows // 2)
    r1 = min(H, r0 + mid_band_rows)
    prof = img[r0:r1].mean(axis=0)
    lo = float(np.percentile(prof, 5))
    hi = float(np.percentile(prof, 95))
    if hi - lo < 5:
        return 0
    thresh = lo + low_fraction * (hi - lo)
    below = prof < thresh
    if not below.any():
        return 0
    best = 0
    run = 0
    for b in below:
        if b:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return int(best)


def measure_wound_gap(image, orientation='horizontal', n_positions=10,
                      min_gap_width=5):
    """Measure wound width at multiple positions along the scratch.

    Args:
        image: 2D array.
        orientation: 'horizontal' or 'vertical'.
        n_positions: int, number of measurement positions.
        min_gap_width: int, minimum gap to consider valid.

    Returns:
        dict with:
            widths: list of float.
            positions: list of int.
            mean_width: float.
            std_width: float.
    """
    result = detect_wound(image, orientation, min_gap_width)
    gap = result['gap_profile']

    length = len(gap)
    margin = max(1, length // 20)
    positions = np.linspace(margin, length - margin - 1, n_positions, dtype=int)
    widths = [float(gap[p]) for p in positions]

    return {
        'widths': widths,
        'positions': positions.tolist(),
        'mean_width': round(float(np.mean(widths)), 2),
        'std_width': round(float(np.std(widths)), 2),
    }


def wound_closure_rate(gap_widths, timepoints, pixel_size_um=1.0):
    """Compute wound closure rate from gap width measurements.

    Args:
        gap_widths: array-like, wound width at each timepoint (pixels).
        timepoints: array-like, time values.
        pixel_size_um: float, pixel size in micrometers.

    Returns:
        dict with:
            initial_gap_um, final_gap_um, closure_um,
            closure_fraction, rate_um_per_time, time_to_close, is_closed.
    """
    widths = np.asarray(gap_widths, dtype=float) * pixel_size_um
    times = np.asarray(timepoints, dtype=float)

    initial = float(widths[0])
    final = float(widths[-1])
    closure = initial - final
    frac = closure / initial if initial > 0 else 0.0

    dt = times[-1] - times[0]
    rate = closure / dt if dt > 0 else 0.0

    time_to_close = None
    if rate > 0 and final > 0:
        time_to_close = float(times[0] + initial / rate)

    return {
        'initial_gap_um': round(initial, 2),
        'final_gap_um': round(final, 2),
        'closure_um': round(closure, 2),
        'closure_fraction': round(frac, 4),
        'rate_um_per_time': round(rate, 4),
        'time_to_close': round(time_to_close, 2) if time_to_close else None,
        'is_closed': final < 0.1 * initial,
    }


def migration_speed(edge_positions, timepoints, pixel_size_um=1.0):
    """Estimate cell migration speed from wound edge positions.

    Args:
        edge_positions: array-like, mean edge position at each timepoint (pixels).
        timepoints: array-like, time values.
        pixel_size_um: float.

    Returns:
        dict with speed_um_per_time, displacement_um, speeds.
    """
    pos = np.asarray(edge_positions, dtype=float) * pixel_size_um
    times = np.asarray(timepoints, dtype=float)

    if len(pos) < 2:
        return {'speed_um_per_time': 0.0, 'displacement_um': 0.0, 'speeds': []}

    dp = np.diff(pos)
    dt = np.diff(times)
    valid = dt > 0
    speeds = np.abs(dp[valid] / dt[valid])

    total_disp = float(np.abs(pos[-1] - pos[0]))
    total_time = times[-1] - times[0]
    mean_speed = total_disp / total_time if total_time > 0 else 0

    return {
        'speed_um_per_time': round(mean_speed, 4),
        'displacement_um': round(total_disp, 2),
        'speeds': [round(s, 4) for s in speeds.tolist()],
    }


def analyze_scratch_assay(stack, timepoints=None, orientation='horizontal',
                          pixel_size_um=1.0):
    """Full pipeline for scratch assay analysis.

    Args:
        stack: 3D array (T, H, W) of timelapse frames.
        timepoints: array-like of time values, or None (uses frame indices).
        orientation: 'horizontal' or 'vertical'.
        pixel_size_um: float.

    Returns:
        dict with per_frame, gap_widths, closure, migration.
    """
    stack = np.asarray(stack, dtype=float)
    n_frames = stack.shape[0]

    if timepoints is None:
        timepoints = np.arange(n_frames, dtype=float)
    else:
        timepoints = np.asarray(timepoints, dtype=float)

    per_frame = []
    gap_widths = []
    edge1_positions = []
    edge2_positions = []

    for i in range(n_frames):
        result = detect_wound(stack[i], orientation)
        per_frame.append({
            'mean_gap': result['mean_gap'],
            'wound_fraction': result['wound_fraction'],
        })
        gap_widths.append(result['mean_gap'])

        e1, e2 = result['edges']
        e1_valid = e1[~np.isnan(e1)]
        e2_valid = e2[~np.isnan(e2)]
        edge1_positions.append(float(np.mean(e1_valid)) if len(e1_valid) > 0 else np.nan)
        edge2_positions.append(float(np.mean(e2_valid)) if len(e2_valid) > 0 else np.nan)

    closure = wound_closure_rate(gap_widths, timepoints, pixel_size_um)

    e1 = np.array(edge1_positions)
    e2 = np.array(edge2_positions)
    valid1 = ~np.isnan(e1)
    valid2 = ~np.isnan(e2)

    mig1 = migration_speed(e1[valid1], timepoints[valid1], pixel_size_um) \
        if valid1.sum() >= 2 else {'speed_um_per_time': 0, 'displacement_um': 0, 'speeds': []}
    mig2 = migration_speed(e2[valid2], timepoints[valid2], pixel_size_um) \
        if valid2.sum() >= 2 else {'speed_um_per_time': 0, 'displacement_um': 0, 'speeds': []}

    return {
        'per_frame': per_frame,
        'gap_widths': [round(g, 2) for g in gap_widths],
        'closure': closure,
        'migration': {
            'edge1': mig1,
            'edge2': mig2,
            'mean_speed': round(
                (mig1['speed_um_per_time'] + mig2['speed_um_per_time']) / 2, 4),
        },
    }


# ===========================================================================
# Laser ablation wound healing functions
# ===========================================================================

def segment_cells_bf(
    img,
    channel='fluorescence',
    min_area=20,
    max_area=2000,
    threshold=None,
):
    """Segment cells from fluorescence or brightfield image.

    For fluorescence: threshold bright regions (cells are bright).
    For brightfield: invert and threshold (cells are dark with halos).

    Args:
        img: 2D numpy array.
        channel: 'fluorescence' (bright cells) or 'brightfield' (dark cells).
        min_area: int, minimum cell area in pixels.
        max_area: int, maximum cell area in pixels.
        threshold: float or None. If None, uses Otsu.

    Returns:
        dict with positions (N,2), areas, n_cells, threshold, labeled.
    """
    img_f = np.asarray(img, dtype=float)

    if channel == 'brightfield':
        img_f = img_f.max() - img_f

    if threshold is None:
        try:
            thr = threshold_otsu(img_f)
            bg = np.median(img_f)
            thr = max(thr, bg + 1.0)
        except Exception:
            bg = np.median(img_f)
            std_val = float(np.std(img_f))
            thr = bg + 2.0 * std_val
    else:
        thr = float(threshold)

    binary = img_f > thr
    labeled, _ = label(binary)
    props = regionprops(labeled)

    cells = [p for p in props if min_area <= p.area <= max_area]
    if cells:
        positions = np.array([[p.centroid[0], p.centroid[1]] for p in cells])
        areas = [p.area for p in cells]
    else:
        positions = np.zeros((0, 2))
        areas = []

    return {
        'positions': positions,
        'areas': areas,
        'n_cells': len(cells),
        'threshold': float(thr),
        'labeled': labeled,
    }


def define_wound_region(fov_size=512, region='right_half', custom_mask=None):
    """Create binary mask defining the wound region.

    Args:
        fov_size: int, size of field of view in pixels.
        region: str, preset region name:
            'right_half', 'left_half', 'top_half', 'bottom_half'.
        custom_mask: 2D boolean array, overrides region if provided.

    Returns:
        2D boolean array (True = wound region).
    """
    if custom_mask is not None:
        return np.asarray(custom_mask, dtype=bool)

    mask = np.zeros((fov_size, fov_size), dtype=bool)
    half = fov_size // 2

    if region == 'right_half':
        mask[:, half:] = True
    elif region == 'left_half':
        mask[:, :half] = True
    elif region == 'top_half':
        mask[:half, :] = True
    elif region == 'bottom_half':
        mask[half:, :] = True
    else:
        raise ValueError(
            f"Unknown region: {region}. Use right_half/left_half/top_half/bottom_half.")

    return mask


def count_cells_in_region(positions, region_mask):
    """Count cells whose centroid falls inside the region mask.

    Args:
        positions: (N, 2) array of (row, col) centroids.
        region_mask: 2D boolean array.

    Returns:
        tuple (n_in_region, n_outside).
    """
    if len(positions) == 0:
        return 0, 0

    rows = positions[:, 0].astype(int).clip(0, region_mask.shape[0] - 1)
    cols = positions[:, 1].astype(int).clip(0, region_mask.shape[1] - 1)
    in_region = region_mask[rows, cols]

    return int(np.sum(in_region)), int(np.sum(~in_region))


def track_wound_repopulation(
    frames,
    wound_mask,
    channel='fluorescence',
    min_area=20,
    max_area=2000,
    threshold=None,
):
    """Track cell count in wound region across a timelapse.

    Args:
        frames: list of 2D numpy arrays (time series).
        wound_mask: 2D boolean array (True = wound region).
        channel: 'fluorescence' or 'brightfield'.
        min_area: int, minimum cell area.
        max_area: int, maximum cell area.
        threshold: float or None. Locked after first frame for consistency.

    Returns:
        dict with n_in_wound, n_total, frac_in_wound,
                 healing_detected, healing_onset_frame, threshold.
    """
    n_in_wound = []
    n_total = []
    locked_threshold = threshold

    for i, frame in enumerate(frames):
        result = segment_cells_bf(
            frame,
            channel=channel,
            min_area=min_area,
            max_area=max_area,
            threshold=locked_threshold,
        )
        if locked_threshold is None and i == 0:
            locked_threshold = result['threshold']

        n_in, n_out = count_cells_in_region(result['positions'], wound_mask)
        n_in_wound.append(n_in)
        n_total.append(result['n_cells'])

    frac_in_wound = [
        n_in / max(n_tot, 1) for n_in, n_tot in zip(n_in_wound, n_total)
    ]

    # Healing: count in wound increases above post-ablation minimum
    healing_onset_frame = None
    healing_detected = False

    if len(n_in_wound) >= 3:
        min_count = min(n_in_wound[:3])
        for i in range(3, len(n_in_wound)):
            if n_in_wound[i] > min_count + 1:
                healing_detected = True
                healing_onset_frame = i
                break

    return {
        'n_in_wound': n_in_wound,
        'n_total': n_total,
        'frac_in_wound': frac_in_wound,
        'healing_detected': healing_detected,
        'healing_onset_frame': healing_onset_frame,
        'threshold': locked_threshold,
    }


def analyze_wound_healing(
    pre_ablation_img,
    ablation_frames,
    healing_frames,
    wound_region='right_half',
    channel='fluorescence',
    fov_size=512,
    min_area=20,
    max_area=2000,
    custom_wound_mask=None,
):
    """Full laser ablation wound healing analysis pipeline.

    Steps:
        1. Measure initial cell count + distribution.
        2. Track ablation progress (cell death in wound region).
        3. Track healing (cell migration into wound region).

    Args:
        pre_ablation_img: 2D array, image before ablation.
        ablation_frames: list of 2D arrays during ablation (SLM on).
        healing_frames: list of 2D arrays after ablation (SLM off).
        wound_region: str, wound region name (see define_wound_region).
        channel: 'fluorescence' or 'brightfield'.
        fov_size: int, FOV size.
        min_area: int, minimum cell area.
        max_area: int, maximum cell area.
        custom_wound_mask: 2D boolean array or None.

    Returns:
        dict with wound_mask, wound_area_frac, initial, ablation, healing.
    """
    wound_mask = define_wound_region(
        fov_size=fov_size,
        region=wound_region,
        custom_mask=custom_wound_mask,
    )
    wound_area_frac = float(wound_mask.mean())

    # Initial state
    init_result = segment_cells_bf(
        pre_ablation_img,
        channel=channel,
        min_area=min_area,
        max_area=max_area,
    )
    threshold = init_result['threshold']
    n_in_wound_init, _ = count_cells_in_region(
        init_result['positions'], wound_mask
    )
    n_initial = init_result['n_cells']
    n_initial_wound = n_in_wound_init

    # Ablation progress
    if ablation_frames:
        last_abl = segment_cells_bf(
            ablation_frames[-1],
            channel=channel,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
        )
        n_after_abl_wound, _ = count_cells_in_region(
            last_abl['positions'], wound_mask
        )
        n_ablated = max(0, n_initial_wound - n_after_abl_wound)
        clearance_efficiency = n_ablated / max(n_initial_wound, 1)
    else:
        n_after_abl_wound = n_initial_wound
        n_ablated = 0
        clearance_efficiency = 0.0

    # Healing phase
    if healing_frames:
        healing_result = track_wound_repopulation(
            healing_frames,
            wound_mask,
            channel=channel,
            min_area=min_area,
            max_area=max_area,
            threshold=threshold,
        )
        max_healed = max(healing_result['n_in_wound']) if healing_result['n_in_wound'] else 0
        repopulation_fraction = max_healed / max(n_initial_wound, 1)
    else:
        healing_result = {
            'healing_detected': False,
            'healing_onset_frame': None,
            'n_in_wound': [], 'n_total': [], 'frac_in_wound': [],
        }
        repopulation_fraction = 0.0

    return {
        'wound_mask': wound_mask,
        'wound_area_frac': round(wound_area_frac, 4),
        'initial': {
            'n_initial': n_initial,
            'n_initial_wound': n_initial_wound,
            'frac_wound': round(n_initial_wound / max(n_initial, 1), 4),
            'threshold': threshold,
        },
        'ablation': {
            'n_post_ablation_wound': n_after_abl_wound,
            'n_ablated': n_ablated,
            'clearance_efficiency': round(float(clearance_efficiency), 4),
        },
        'healing': {
            'healing_detected': healing_result['healing_detected'],
            'healing_onset_frame': healing_result.get('healing_onset_frame'),
            'n_in_wound_series': healing_result['n_in_wound'],
            'n_total_series': healing_result['n_total'],
            'repopulation_fraction': round(float(repopulation_fraction), 4),
        },
    }
