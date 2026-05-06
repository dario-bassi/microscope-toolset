"""Bacterial light trap via SLM phototaxis.

Concentrates motile bacteria into a target region by illuminating that
area with the SLM. Bacteria slow down in bright regions (Frangipane 2018
speed reduction model), accumulating over time.

Key physics: Speed factor ~0.15 in illuminated regions (85% slower).
Expected enrichment: 3-4× after 40+ steps.

Coordinate convention (CRITICAL):
    Challenge says "center at (x, y) = (300, 256)" → x=300, y=256.
    In numpy array: row=256, col=300.
    In SLM mask: make_slm_circle(center=(300, 256), ...) = (col=300, row=256).

Functions:
    detect_bacteria_gfp        -- GFP connected-component detection
    detect_bacteria_area_threshold -- robust rod-shaped bacteria detection
    compute_enrichment         -- density ratio inside vs outside circle
    measure_bacteria_intensity -- intensity-based enrichment (primary metric)
    run_bacteria_trap_mda      -- full experiment via MDA + run_events()

Real microscope note:
    This workflow assumes INSTANT bacterial response to SLM illumination.
    Real bacteria have 100–500ms phototaxis reaction lag plus acceleration
    ramp time. On real hardware:
    - Add configurable response_lag parameter (e.g., 200ms) to imaging loop
    - Extend accumulation phase (n_accum) to allow phototaxis saturation
    - Consider adaptive interval stepping: shorter delays early, longer late
    - Validate enrichment on pilot runs before committing to long experiments
"""

import numpy as np
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from useq import MDAEvent

from ..hardware.core import run_events


def detect_bacteria_gfp(img, threshold=None, min_area=3, max_area=50):
    """Detect bacteria as bright GFP spots on dark background.

    Uses connected components after thresholding. Returns positions in
    (row, col) / (y, x) format (centroid from regionprops).

    Args:
        img: 2D numpy array, GFP fluorescence image.
        threshold: float or None. If None, uses Otsu. Falls back to
            mean+3*std if Otsu fails or image is nearly uniform.
        min_area: int, minimum object area in pixels (default 3).
        max_area: int, maximum object area in pixels (default 50).

    Returns:
        tuple of (positions, n_bacteria, threshold_used):
            positions: (N, 2) array of (row, col) centroids.
            n_bacteria: int count.
            threshold_used: float.

    Notes:
        - At 20x: bacteria span 2-3 px → min_area=3, max_area=50.
        - GFP intensity is typically 0-10 at standard exposure.
          threshold>4 works well; Otsu handles variable brightness.
        - Never count raw pixels — always use connected components.
    """
    img_f = np.asarray(img, dtype=float)

    if threshold is None:
        try:
            thr = threshold_otsu(img_f)
            # Otsu can be too low if background dominates; enforce minimum
            bg = np.median(img_f)
            thr = max(thr, bg + 1.0)
        except Exception:
            bg = np.median(img_f)
            std_val = float(np.std(img_f))
            thr = bg + max(3 * std_val, 1.0)
    else:
        thr = float(threshold)

    binary = img_f > thr
    labeled, _ = label(binary)
    props = regionprops(labeled)

    cells = [p for p in props if min_area <= p.area <= max_area]
    if cells:
        positions = np.array([[p.centroid[0], p.centroid[1]] for p in cells])
    else:
        positions = np.zeros((0, 2))

    return positions, len(cells), thr


def detect_bacteria_area_threshold(img, min_area=8, max_area=120, std_multiplier=2.0):
    """Detect bacteria using area-threshold approach (robust for rod-shaped bacteria).

    Rod-shaped bacteria are NOT well-detected by blob_log (designed for round blobs).
    This method uses direct intensity thresholding + connected component area filtering.

    CRITICAL: Use this method for E. coli and other rod-shaped bacteria.
    blob_log detection misses many rod-shaped bacteria due to shape mismatch.

    Args:
        img: 2D numpy array, GFP fluorescence image.
        min_area: int, minimum object area in pixels (default 8).
            Rod-shaped bacteria: ~15-80px at 10x, 3-20px at 20x.
        max_area: int, maximum object area in pixels (default 120).
            Clumps larger than max_area are excluded.
        std_multiplier: float, threshold = background + std_multiplier * std.
            Default 2.0. Higher values are more conservative.

    Returns:
        tuple of (n_bacteria, positions, threshold_used):
            n_bacteria: int count.
            positions: (N, 2) array of (row, col) centroids.
            threshold_used: float threshold value.

    Notes:
        - Background estimated from 10th percentile (dim pixels).
        - Area filter removes noise (< min_area) and clumps (> max_area).
        - At 10x: bacteria ~8-120 px, FOV = 512×512.
        - At 20x: bacteria ~3-30 px (use min_area=3, max_area=50).
    """
    img_f = np.asarray(img, dtype=float)

    # Background estimation
    background = float(np.percentile(img_f, 10))
    sigma = float(img_f.std())
    thr = background + std_multiplier * sigma

    binary = img_f > thr
    labeled, _ = label(binary)
    props = regionprops(labeled)

    cells = [p for p in props if min_area <= p.area <= max_area]
    if cells:
        positions = np.array([[p.centroid[0], p.centroid[1]] for p in cells])
    else:
        positions = np.zeros((0, 2))

    return len(cells), positions, float(thr)


def count_bacteria_in_circle(positions, center_x, center_y, radius):
    """Count bacteria whose centroid falls inside a circle.

    Args:
        positions: (N, 2) array of (row, col) centroids.
        center_x: float, circle center x (column coordinate).
        center_y: float, circle center y (row coordinate).
        radius: float, circle radius in pixels.

    Returns:
        int, number of bacteria inside circle.
    """
    if len(positions) == 0:
        return 0

    rows = positions[:, 0]
    cols = positions[:, 1]
    dists = np.sqrt((rows - center_y) ** 2 + (cols - center_x) ** 2)
    return int(np.sum(dists <= radius))


def compute_enrichment(positions, center_row, center_col, radius, fov_size=512):
    """Compute enrichment ratio: density inside / density outside circle.

    Enrichment > 1.0 means bacteria have accumulated in the target region.
    Expected value ~3-4× after 40+ illumination steps.

    Args:
        positions: (N, 2) array of (row, col) centroids.
        center_row: int/float, circle center row (numpy y-axis).
        center_col: int/float, circle center col (numpy x-axis).
        radius: float, circle radius in pixels.
        fov_size: int, field of view size in pixels (default 512).

    Returns:
        dict with:
            enrichment: float, ratio of inside density to outside density.
            n_inside: int, bacteria inside circle.
            n_outside: int, bacteria outside circle.
            n_total: int, total bacteria.
            frac_inside: float, fraction of bacteria inside circle.
            area_frac: float, fraction of FOV inside circle.
    """
    area_inside = np.pi * radius**2
    area_fov = fov_size**2
    area_frac = area_inside / area_fov

    if len(positions) == 0:
        return {
            "enrichment": 0.0,
            "n_inside": 0,
            "n_outside": 0,
            "n_total": 0,
            "frac_inside": 0.0,
            "area_frac": area_frac,
        }

    rows = positions[:, 0]
    cols = positions[:, 1]
    dists = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)

    n_inside = int(np.sum(dists <= radius))
    n_outside = int(np.sum(dists > radius))
    n_total = n_inside + n_outside

    frac_inside = n_inside / n_total if n_total > 0 else 0.0

    # Enrichment: ratio of density inside vs overall (normalized by area fraction)
    # enrichment=1 is random; enrichment=3 means 3x concentrated inside
    enrichment = (frac_inside / area_frac) if area_frac > 0 else 0.0

    return {
        "enrichment": round(float(enrichment), 3),
        "n_inside": n_inside,
        "n_outside": n_outside,
        "n_total": n_total,
        "frac_inside": round(float(frac_inside), 4),
        "area_frac": round(float(area_frac), 4),
    }


def run_bacteria_trap_mda(
    core,
    target_center_x,
    target_center_y,
    target_radius,
    gfp_channel="nucleus-channel",
    config_group="Fake",
    n_baseline=5,
    n_accum=50,
    n_final=3,
    slm_device="SLM",
    min_area=8,
    max_area=120,
    std_multiplier=2.0,
):
    """MDA-native bacterial light trap experiment.

    Replaces the snap()-loop version with proper MDA + run_events() throughout.
    Each phase uses a generator of MDAEvents with an on_frame callback for
    real-time analysis.

    Protocol:
        Phase 1 (baseline): N_BASELINE MDA frames, SLM off.
        SLM on: circle at (target_center_x, target_center_y).
        Phase 2 (accumulation): N_ACCUM MDA frames with SLM on.
        SLM off.
        Phase 3 (final): N_FINAL MDA frames to confirm enrichment.

    Args:
        core: pymmcore-proxy core instance.
        target_center_x: int, circle center column.
        target_center_y: int, circle center row.
        target_radius: int, circle radius in pixels.
        gfp_channel: str, channel name for GFP.
        config_group: str, config group name.
        n_baseline: int, baseline frames (default 5).
        n_accum: int, accumulation frames (default 50).
        n_final: int, final measurement frames (default 3).
        slm_device: str, SLM device name.
        min_area: int, minimum bacteria area (default 8).
        max_area: int, maximum bacteria area (default 120).
        std_multiplier: float, threshold = bg + mult*std (default 2.0).

    Returns:
        dict with:
            baseline_counts: list[int], per-frame total counts.
            baseline_in: list[int], per-frame in-trap counts.
            baseline_intensity: list[float], per-frame intensity enrichment.
            accum_counts: list, per-frame total during accumulation.
            accum_in: list, per-frame in-trap counts.
            accum_intensity: list[float], per-frame intensity enrichment.
            final_count: int, median final total count.
            final_in: int, median final in-trap count.
            final_intensity_enrichment: float, median final intensity enrichment.
            count_enrichment: float, final (in/total) / area_frac.
            intensity_enrichment: float, primary enrichment metric.
            area_frac: float, target circle area fraction.
            locked_threshold: float, threshold locked after baseline.
    """
    circle_mask_arr = np.zeros((512, 512), dtype=bool)
    yy, xx = np.ogrid[:512, :512]
    circle_mask_arr[:] = (xx - target_center_x) ** 2 + (
        yy - target_center_y
    ) ** 2 <= target_radius**2
    area_frac = float(np.sum(circle_mask_arr)) / (512.0 * 512.0)

    # Phase 1: baseline (SLM off)
    core.setSLMImage(slm_device, np.zeros((512, 512), dtype=np.uint8))
    baseline_counts, baseline_in, baseline_intensity = [], [], []
    locked_threshold = None

    def baseline_gen():
        for _ in range(n_baseline):
            yield MDAEvent(channel={"config": gfp_channel, "group": config_group})

    def on_baseline(img, event):
        g = img[:, :, 0].astype(float) if img.ndim == 3 else img.astype(float)
        nonlocal locked_threshold
        n, pos, thr = detect_bacteria_area_threshold(
            g, min_area=min_area, max_area=max_area, std_multiplier=std_multiplier
        )
        if locked_threshold is None:
            locked_threshold = thr
        n_in = count_bacteria_in_circle(pos, target_center_x, target_center_y, target_radius)
        ie = measure_bacteria_intensity(g, circle_mask_arr)
        baseline_counts.append(n)
        baseline_in.append(n_in)
        baseline_intensity.append(ie["enrichment"])

    run_events(core, baseline_gen(), on_frame=on_baseline)

    # Apply SLM
    slm_mask = np.zeros((512, 512), dtype=np.uint8)
    slm_mask[circle_mask_arr] = 255
    core.setSLMImage(slm_device, slm_mask)
    core.displaySLMImage(slm_device)

    # Phase 2: accumulation (SLM on)
    accum_counts, accum_in, accum_intensity = [], [], []

    def accum_gen():
        for _ in range(n_accum):
            yield MDAEvent(channel={"config": gfp_channel, "group": config_group})

    def on_accum(img, event):
        g = img[:, :, 0].astype(float) if img.ndim == 3 else img.astype(float)
        n, pos, _ = detect_bacteria_area_threshold(
            g, min_area=min_area, max_area=max_area, std_multiplier=std_multiplier
        )
        n_in = count_bacteria_in_circle(pos, target_center_x, target_center_y, target_radius)
        ie = measure_bacteria_intensity(g, circle_mask_arr)
        accum_counts.append(n)
        accum_in.append(n_in)
        accum_intensity.append(ie["enrichment"])

    run_events(core, accum_gen(), on_frame=on_accum)

    # SLM off
    core.setSLMImage(slm_device, np.zeros((512, 512), dtype=np.uint8))

    # Phase 3: final measurement
    final_counts_raw, final_in_raw, final_ie_raw = [], [], []

    def final_gen():
        for _ in range(n_final):
            yield MDAEvent(channel={"config": gfp_channel, "group": config_group})

    def on_final(img, event):
        g = img[:, :, 0].astype(float) if img.ndim == 3 else img.astype(float)
        n, pos, _ = detect_bacteria_area_threshold(
            g, min_area=min_area, max_area=max_area, std_multiplier=std_multiplier
        )
        n_in = count_bacteria_in_circle(pos, target_center_x, target_center_y, target_radius)
        ie = measure_bacteria_intensity(g, circle_mask_arr)
        final_counts_raw.append(n)
        final_in_raw.append(n_in)
        final_ie_raw.append(ie["enrichment"])

    run_events(core, final_gen(), on_frame=on_final)

    final_count = int(np.median(final_counts_raw)) if final_counts_raw else 0
    final_in = int(np.median(final_in_raw)) if final_in_raw else 0
    final_frac = final_in / max(final_count, 1)
    final_int_enrichment = float(np.median(final_ie_raw)) if final_ie_raw else 0.0
    count_enrichment = final_frac / max(area_frac, 1e-6)

    return {
        "baseline_counts": baseline_counts,
        "baseline_in": baseline_in,
        "baseline_intensity": baseline_intensity,
        "accum_counts": accum_counts,
        "accum_in": accum_in,
        "accum_intensity": accum_intensity,
        "final_count": final_count,
        "final_in": final_in,
        "final_intensity_enrichment": round(final_int_enrichment, 3),
        "count_enrichment": round(count_enrichment, 3),
        "intensity_enrichment": round(final_int_enrichment, 3),
        "area_frac": round(area_frac, 4),
        "locked_threshold": float(locked_threshold) if locked_threshold is not None else 0.0,
    }


def measure_bacteria_intensity(img, circle_mask, total_mask=None):
    """Measure bacterial density via summed fluorescence intensity.

    Intensity-based method sidesteps counting entirely. The total GFP
    fluorescence in a region is proportional to the bacterial biomass,
    regardless of individual cell shape or fragmentation during detection.

    This method is robust to:
    - Rod-shaped bacteria (no shape assumption)
    - Clumped bacteria (no separation needed)
    - Fragmentation artifacts from connected-component labeling
    - Overcounting from small area filters

    Args:
        img: 2D numpy array, background-subtracted fluorescence image.
        circle_mask: boolean 2D array marking the target circle.
        total_mask: boolean 2D array for normalization region.
            Defaults to the entire image.

    Returns:
        dict with:
            intensity_in: float, total intensity inside circle.
            intensity_out: float, total intensity outside circle.
            intensity_total: float, total intensity in normalization region.
            frac_in: float, fraction of total intensity inside circle.
            area_frac: float, fraction of area inside circle.
            enrichment: float, frac_in / area_frac.
    """
    img_f = np.asarray(img, dtype=float)

    # Background subtraction: subtract min to make signal-proportional
    background = float(np.percentile(img_f, 5))
    img_bg = np.maximum(img_f - background, 0.0)

    if total_mask is None:
        total_mask = np.ones(img_f.shape, dtype=bool)

    area_total = float(np.sum(total_mask))
    area_in = float(np.sum(circle_mask & total_mask))
    area_frac = area_in / area_total if area_total > 0 else 0.0

    intensity_in = float(np.sum(img_bg[circle_mask & total_mask]))
    intensity_total = float(np.sum(img_bg[total_mask]))
    intensity_out = intensity_total - intensity_in

    frac_in = intensity_in / intensity_total if intensity_total > 0 else 0.0
    enrichment = frac_in / area_frac if area_frac > 0 else 0.0

    return {
        "intensity_in": round(intensity_in, 2),
        "intensity_out": round(intensity_out, 2),
        "intensity_total": round(intensity_total, 2),
        "frac_in": round(frac_in, 4),
        "area_frac": round(area_frac, 4),
        "enrichment": round(enrichment, 3),
    }
