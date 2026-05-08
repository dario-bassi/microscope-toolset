"""Organelle / vesicle puncta detection and per-cell quantification.

Detects bright fluorescent puncta (lysosomes, endosomes, synaptic vesicles,
FISH spots) using LoG filtering, and assigns them to segmented cell regions
to compute per-cell statistics.

Functions:
    detect_puncta_log        -- LoG-based bright spot detection
    detect_elongated_puncta  -- Tophat + eccentricity for FAs / fiber-borne puncta
    segment_cells_from_nuclei -- DAPI nuclei → expanded cell regions
    count_puncta_per_cell    -- Assign puncta to cell regions
    puncta_statistics        -- Summary statistics from per-cell counts
    lysosome_analysis        -- Full pipeline: DAPI + LysoTracker → stats
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import disk, remove_small_objects, white_tophat
from skimage.segmentation import watershed


def detect_puncta_log(image, sigma=1.5, thresh_factor=0.3,
                      min_intensity_factor=2.5, min_distance=3,
                      log_threshold_abs=None):
    """Detect bright puncta using Laplacian of Gaussian filtering.

    LoG highlights blob-like features at scale `sigma`. Works for bright
    spots on dark background (lysosomes, FISH dots, synaptic vesicles).

    Two gating modes (use one or both):
      - **σ-relative** (default): threshold the LoG response at
        ``mean + thresh_factor·std`` and keep peaks above
        ``mean + min_intensity_factor·std`` of the raw image.
      - **Absolute LoG response** (``log_threshold_abs=...``): threshold
        the LoG response by an absolute value; the raw-intensity gate is
        skipped. Use this when bright puncta dominate the image's std,
        which inflates the σ-relative gate into the puncta band itself
        and lets cell-edge gradient hits through.

    Args:
        image: 2D float array.
        sigma: float, LoG scale. Match to puncta radius ≈ sigma * sqrt(2).
            - 2-5 px diameter → sigma ≈ 0.7-1.8; use sigma=1.5
            - 5-10 px diameter → sigma ≈ 1.8-3.5; use sigma=2.5
            - 10-20 px diameter → sigma ≈ 3.5-7; use sigma=5
        thresh_factor: float, σ-relative LoG response threshold =
            mean + factor * std. Ignored when ``log_threshold_abs`` is set.
        min_intensity_factor: float, σ-relative raw-intensity gate.
            Ignored when ``log_threshold_abs`` is set.
        min_distance: int, minimum distance between peaks (pixels).
        log_threshold_abs: float, absolute threshold on the LoG response.
            When set, this single parameter replaces the σ-relative LoG
            gate and the raw-intensity gate. Pre-probe on a representative
            frame to pick the value; for ~3 px puncta at ``sigma=1.5`` a
            value in the low tens (≈ 20–30) is typical.

    Returns:
        peaks: (N, 2) array of (row, col) coordinates.
        log_responses: (N,) array of LoG response at each peak.
    """
    img = np.asarray(image, dtype=float)
    log = -ndimage.gaussian_laplace(img, sigma=sigma)

    if log_threshold_abs is not None:
        peaks = peak_local_max(
            log,
            min_distance=max(1, min_distance),
            threshold_abs=float(log_threshold_abs),
        )
        if len(peaks) == 0:
            return np.zeros((0, 2), dtype=int), np.array([])
        responses = np.array([log[r, c] for r, c in peaks])
        return peaks, responses

    log_thresh = log.mean() + thresh_factor * log.std()

    peaks = peak_local_max(
        log,
        min_distance=max(1, min_distance),
        threshold_abs=log_thresh,
    )

    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=int), np.array([])

    # Filter by raw intensity
    int_thresh = img.mean() + min_intensity_factor * img.std()
    valid = np.array([img[r, c] > int_thresh for r, c in peaks])
    if valid.any():
        peaks = peaks[valid]
        responses = np.array([log[r, c] for r, c in peaks])
    else:
        # Relax filter: keep top quartile by LoG response
        responses_all = np.array([log[r, c] for r, c in peaks])
        thresh_q = np.percentile(responses_all, 75)
        keep = responses_all >= thresh_q
        peaks = peaks[keep]
        responses = responses_all[keep]

    return peaks, responses


def detect_elongated_puncta(image, tophat_radius=5, threshold_sigmas=3.5,
                              min_area=3, max_area=60,
                              min_eccentricity=0.7):
    """Detect elongated bright puncta against a structured bright background.

    LoG alone over-counts when the background itself contains bright
    linear structures (e.g. phalloidin stress fibers, microtubule bundles,
    actin cables) — every pixel along the fiber registers a LoG maximum.
    This variant uses a white-tophat to suppress smooth and line-like
    baselines, then filters connected components by elongation + size.

    Tuned defaults match focal-adhesion detection on fibroblasts at
    0.5 µm/px (FAs ≈ 1-4 µm long, aspect ratio ≥ 3:1, 5-10 per cell).

    Args:
        image: 2D float array (one channel).
        tophat_radius: int, structuring-element radius for white_tophat.
            Should exceed the puncta's longest axis (~2× max length).
            disk(5) ≈ 10 px footprint fits 2-8 px FAs comfortably.
        threshold_sigmas: float, threshold on tophat response =
            mean + sigmas*std. Lower = more sensitive.
        min_area, max_area: int, area (px) bounds on accepted components.
        min_eccentricity: float in [0, 1), skimage regionprops.eccentricity
            cutoff. 0 = round, 1 = line. Use 0.7+ for clearly elongated,
            0.9+ for very linear features.

    Returns:
        list of skimage RegionProperties for accepted puncta, sorted by
        descending area. Each has .centroid, .bbox, .area, .eccentricity,
        .orientation, .mean_intensity, .max_intensity.

    See also:
        detect_puncta_log: for round / spatially isolated bright puncta.
    """
    img = np.asarray(image, dtype=float)
    th = white_tophat(img, disk(tophat_radius))
    mask = th > (th.mean() + threshold_sigmas * th.std())
    # New skimage API: max_size removes objects with ≤ that many pixels.
    # Equivalent to old min_size=N (remove strictly smaller than N) is
    # max_size=N-1.
    mask = remove_small_objects(mask, max_size=max(0, min_area - 1))
    lbl = sk_label(mask)
    props = regionprops(lbl, intensity_image=img)
    accepted = [
        p for p in props
        if min_area <= p.area <= max_area
        and p.eccentricity > min_eccentricity
    ]
    accepted.sort(key=lambda p: -p.area)
    return accepted


def detect_puncta_multiscale(image, sigmas=(1.0, 1.5, 2.0),
                              thresh_factor=0.25, min_distance=3):
    """Detect puncta across multiple LoG scales.

    Useful when puncta size varies (e.g. different stages of lysosome maturation).
    Deduplicates peaks within min_distance of each other.

    Args:
        image: 2D float array.
        sigmas: tuple of LoG scales to try.
        thresh_factor: LoG response threshold factor.
        min_distance: int, suppression radius for deduplication.

    Returns:
        peaks: (N, 2) array of (row, col) coordinates.
    """
    img = np.asarray(image, dtype=float)
    all_peaks = []

    for s in sigmas:
        peaks, _ = detect_puncta_log(image, sigma=s,
                                      thresh_factor=thresh_factor,
                                      min_distance=min_distance)
        all_peaks.extend(peaks.tolist())

    if not all_peaks:
        return np.zeros((0, 2), dtype=int)

    # Deduplicate: suppress peaks within min_distance of a stronger one
    all_peaks = np.array(all_peaks)
    keep = np.ones(len(all_peaks), dtype=bool)
    # Sort by LoG response (use intensity as proxy)
    int_vals = np.array([img[r, c] for r, c in all_peaks])
    order = np.argsort(-int_vals)
    all_peaks = all_peaks[order]

    kept = []
    for i, (r, c) in enumerate(all_peaks):
        if not keep[i]:
            continue
        kept.append((r, c))
        for j in range(i + 1, len(all_peaks)):
            if not keep[j]:
                continue
            rr, cc = all_peaks[j]
            if (r - rr)**2 + (c - cc)**2 <= min_distance**2:
                keep[j] = False

    return np.array(kept, dtype=int) if kept else np.zeros((0, 2), dtype=int)


def segment_cells_from_nuclei(dapi_image, min_nucleus_area=100,
                               expand_px=40, edge_margin=0):
    """Segment cell regions by expanding detected DAPI nuclei.

    Uses watershed from nucleus seeds to partition the image into
    Voronoi-like cell territories. Each nucleus seed grows outward
    up to `expand_px` pixels, stopping where it meets another cell.

    Args:
        dapi_image: 2D float array, DAPI/nuclear channel.
        min_nucleus_area: int, minimum nucleus area to count (pixels).
        expand_px: int, maximum expansion radius beyond nucleus edge.
            At 20x (0.5 µm/px), typical fibroblast/HeLa:
              - compact: expand_px=30-40 px (~15-20 µm beyond nucleus)
              - spread: expand_px=50-60 px (~25-30 µm beyond nucleus)
        edge_margin: int, exclude nuclei with centroid within this many
            pixels of image border (0 = include all nuclei).

    Returns:
        cell_labels: 2D int array, same size as dapi_image.
            0 = background, 1..N = cell IDs.
        n_cells: int, number of detected cells.
        nucleus_props: list of skimage regionprops for nucleus regions.
    """
    img = np.asarray(dapi_image, dtype=float)
    H, W = img.shape

    # Threshold nuclei
    thresh = img.mean() + 1.5 * img.std()
    nuc_mask = img > thresh

    # Label and filter by size
    lbl, _ = ndimage.label(nuc_mask)
    props = regionprops(lbl)
    clean_mask = np.zeros_like(nuc_mask)
    good_props = []
    for p in props:
        if p.area < min_nucleus_area:
            continue
        cy, cx = p.centroid
        if edge_margin > 0:
            if cy <= edge_margin or cy >= H - edge_margin:
                continue
            if cx <= edge_margin or cx >= W - edge_margin:
                continue
        clean_mask |= lbl == p.label
        good_props.append(p)

    nuc_seeds, n_seeds = ndimage.label(clean_mask)
    if n_seeds == 0:
        return np.zeros(img.shape, dtype=int), 0, []

    # Distance map from nucleus edges
    dist_map = ndimage.distance_transform_edt(~clean_mask)
    expand_mask = dist_map <= expand_px

    # Watershed fills from seeds into expansion region
    cell_labels = watershed(dist_map, nuc_seeds, mask=expand_mask)

    return cell_labels, n_seeds, good_props


def count_puncta_per_cell(peaks, cell_labels, n_cells=None):
    """Assign detected puncta to cell regions.

    Args:
        peaks: (N, 2) array of (row, col) puncta coordinates.
        cell_labels: 2D int array (same shape as image). 0=background.
        n_cells: int or None, number of cells. If None, inferred
            from max(cell_labels).

    Returns:
        counts: dict mapping cell_id → puncta_count.
        unassigned: int, puncta not in any cell region.
    """
    if n_cells is None:
        n_cells = int(cell_labels.max())

    counts = {cid: 0 for cid in range(1, n_cells + 1)}
    unassigned = 0

    for row, col in peaks:
        row, col = int(row), int(col)
        if 0 <= row < cell_labels.shape[0] and 0 <= col < cell_labels.shape[1]:
            cid = int(cell_labels[row, col])
            if cid > 0:
                counts[cid] = counts.get(cid, 0) + 1
            else:
                unassigned += 1
        else:
            unassigned += 1

    return counts, unassigned


def puncta_statistics(counts_per_cell):
    """Compute population statistics from per-cell puncta counts.

    Args:
        counts_per_cell: dict or list of per-cell counts.

    Returns:
        dict with: mean, median, std, min, max, total, n_cells, counts_list.
    """
    vals = list(counts_per_cell.values()) if isinstance(counts_per_cell, dict) else list(counts_per_cell)
    if not vals:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0,
            'min': 0, 'max': 0, 'total': 0, 'n_cells': 0,
            'counts_list': [],
        }

    arr = np.array(vals, dtype=float)
    return {
        'mean': round(float(np.mean(arr)), 2),
        'median': round(float(np.median(arr)), 2),
        'std': round(float(np.std(arr)), 2),
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'total': int(np.sum(arr)),
        'n_cells': len(vals),
        'counts_list': sorted(vals),
    }


def lysosome_analysis(lyso_image, dapi_image,
                      puncta_sigma=1.5,
                      puncta_thresh_factor=0.3,
                      expand_px=40,
                      min_nucleus_area=100,
                      edge_margin=0):
    """Full lysosome / acidic organelle analysis pipeline.

    Steps:
    1. Detect nuclei from DAPI → segment cell regions (watershed expansion)
    2. Detect LysoTracker puncta via LoG
    3. Assign puncta to cells → per-cell statistics

    Args:
        lyso_image: 2D float array, LysoTracker or similar channel.
        dapi_image: 2D float array, DAPI/nuclear channel.
        puncta_sigma: LoG scale (see detect_puncta_log).
        puncta_thresh_factor: LoG response threshold factor.
        expand_px: cell expansion radius beyond nucleus.
        min_nucleus_area: minimum nucleus area (pixels).
        edge_margin: exclude edge nuclei within this border margin.

    Returns:
        dict with:
            cell_labels: 2D array, segmented cell map.
            peaks: (N,2) puncta positions.
            counts_per_cell: dict, per-cell counts.
            stats: dict from puncta_statistics().
    """
    # Step 1: cell regions
    cell_labels, n_cells, nuc_props = segment_cells_from_nuclei(
        dapi_image,
        min_nucleus_area=min_nucleus_area,
        expand_px=expand_px,
        edge_margin=edge_margin,
    )

    # Step 2: puncta detection
    peaks, responses = detect_puncta_log(
        lyso_image,
        sigma=puncta_sigma,
        thresh_factor=puncta_thresh_factor,
    )

    # Step 3: assignment
    counts, unassigned = count_puncta_per_cell(peaks, cell_labels, n_cells)
    stats = puncta_statistics(counts)

    return {
        'cell_labels': cell_labels,
        'peaks': peaks,
        'log_responses': responses,
        'counts_per_cell': counts,
        'n_cells': n_cells,
        'n_puncta_total': len(peaks),
        'n_puncta_unassigned': unassigned,
        'stats': stats,
    }
