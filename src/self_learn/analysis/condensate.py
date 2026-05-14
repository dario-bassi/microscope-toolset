"""Protein condensate and aggregate analysis.

Detects and characterizes biomolecular condensates: nuclear foci, stress granules,
P-bodies, and other punctate fluorescent assemblies formed by liquid-liquid phase
separation (LLPS) or aggregation.

Condensates are distinct from lipid droplets:
  - Typically smaller: 0.1–2 µm (1–8 px at 40x)
  - May be nuclear, cytoplasmic, or both
  - Dynamic: form/dissolve in minutes under stress or drug treatment
  - Often concentrate specific proteins (GFP/mCherry fusions)

Common applications:
  - Stress granule formation (heat shock, oxidative stress, arsenite)
  - P-body dynamics (mRNA processing/decay bodies)
  - Nuclear speckle analysis (splicing factor condensation)
  - Paraspeckle detection
  - Aggregation assays (Huntingtin polyQ, α-synuclein, TDP-43)
  - LLPS reporter assays (FUS, hnRNPA1, EWS/FLI1)

Functions:
    detect_foci               -- LoG blob detection + connected-component fallback
    foci_per_cell             -- Assign foci to cells, count and measure per cell
    foci_intensity_profile    -- Radial intensity profile around each focus
    spatial_distribution      -- Nuclear vs cytoplasmic foci ratio
    condensate_dynamics       -- Temporal tracking: formation/dissolution events
    stress_response_index     -- Fraction of cells with ≥N foci (stress metric)
    aggregation_score         -- Per-cell aggregate burden (count × mean intensity)
    foci_statistics           -- Population-level statistics across cells
    condensate_analysis       -- Full pipeline: detect, assign, measure, summarize
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label as ndlabel
from skimage.feature import blob_log, peak_local_max
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects, disk, binary_erosion
from skimage.filters import gaussian as sk_gaussian


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_foci(img, min_sigma=0.8, max_sigma=5.0, n_scales=8, threshold=0.05,
                min_area=3, max_area=500, mask=None):
    """Detect punctate fluorescent foci using LoG blob detection.

    Suitable for any small, bright, round structures: stress granules,
    P-bodies, nuclear speckles, protein aggregates. Uses Laplacian-of-Gaussian
    scale-space blob detection with connected-component refinement.

    Args:
        img: 2D float or uint array. Single fluorescence channel (z-projected or
            single plane). Higher intensity = more signal.
        min_sigma: float. Minimum blob sigma (smallest detectable foci, ~1 px).
        max_sigma: float. Maximum blob sigma (largest detectable foci, ~5 px).
        n_scales: int. Number of sigma scales in LoG pyramid.
        threshold: float. LoG response threshold. Lower = more sensitive;
            0.05 works for typical GFP foci, increase for noisy images.
        min_area: int. Minimum focus area in pixels (removes noise peaks).
        max_area: int. Maximum focus area in pixels (removes large blobs).
        mask: 2D bool array (optional). Restrict detection to True pixels.
            Useful to search only within cells or nucleus.

    Returns:
        list of dict, one per detected focus:
            cy, cx     -- centroid (row, column) in pixels
            sigma      -- estimated radius in pixels (from LoG)
            area       -- pixel area of the focus
            peak_intensity -- maximum pixel intensity
            mean_intensity -- mean intensity within focus region
            total_intensity -- sum of intensities (proxy for content)
    """
    gray = img.astype(float)
    if gray.max() == 0:
        return []

    # Normalize to [0, 1] for LoG
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)

    # Apply mask
    if mask is not None:
        gray_norm = gray_norm * mask.astype(float)

    # LoG blob detection
    blobs = blob_log(gray_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=n_scales, threshold=threshold, overlap=0.5)

    if len(blobs) == 0:
        return []

    foci = []
    H, W = gray.shape
    for blob in blobs:
        cy, cx, sigma = blob
        cy, cx = float(cy), float(cx)
        r = int(np.ceil(sigma * 2.0))  # analysis radius

        # Build circular region mask
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        region_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (sigma * 2.0) ** 2

        patch = gray[y0:y1, x0:x1]
        area = int(region_mask.sum())

        if area < min_area or area > max_area:
            continue
        if mask is not None and not mask[int(cy), int(cx)]:
            continue

        intensities = patch[region_mask]
        foci.append({
            'cy': cy, 'cx': cx, 'sigma': float(sigma),
            'area': area,
            'peak_intensity': float(patch.max()),
            'mean_intensity': float(intensities.mean()),
            'total_intensity': float(intensities.sum()),
        })

    return foci


def detect_foci_threshold(img, background_multiplier=2.5, min_area=3, max_area=500,
                           smooth_sigma=0.5, mask=None):
    """Detect foci by adaptive threshold above local background.

    Alternative to LoG for images with dense or irregular foci patterns.
    Thresholds at background_multiplier × estimated background level.

    Args:
        img: 2D float array. Fluorescence channel.
        background_multiplier: float. Threshold = bg_mean + multiplier × bg_std.
        min_area: int. Minimum connected component area.
        max_area: int. Maximum connected component area.
        smooth_sigma: float. Pre-smoothing sigma (0 = no smoothing).
        mask: 2D bool array (optional). Restrict analysis region.

    Returns:
        list of dict with same fields as detect_foci().
    """
    gray = img.astype(float)
    if gray.max() == 0:
        return []

    if smooth_sigma > 0:
        gray = gaussian_filter(gray, sigma=smooth_sigma)

    # Estimate background from low-intensity pixels
    bg_threshold = np.percentile(gray, 50)
    bg_pixels = gray[gray <= bg_threshold]
    bg_mean = float(bg_pixels.mean()) if len(bg_pixels) > 0 else 0.0
    bg_std = float(bg_pixels.std()) if len(bg_pixels) > 0 else 1.0

    thresh = bg_mean + background_multiplier * bg_std
    binary = gray > thresh
    if mask is not None:
        binary = binary & mask

    if not binary.any():
        return []

    binary = remove_small_objects(binary, max_size=min_area)
    labeled, _ = ndlabel(binary)
    props = regionprops(labeled, intensity_image=img.astype(float))

    foci = []
    for p in props:
        if p.area > max_area:
            continue
        cy, cx = p.centroid
        foci.append({
            'cy': float(cy), 'cx': float(cx),
            'sigma': float(np.sqrt(p.area / np.pi)),
            'area': int(p.area),
            'peak_intensity': float(p.intensity_max),
            'mean_intensity': float(p.intensity_mean),
            'total_intensity': float(p.intensity_mean * p.area),
        })
    return foci


# ── Cell-level analysis ────────────────────────────────────────────────────────

def foci_per_cell(foci, cell_labels):
    """Assign detected foci to cells and compute per-cell metrics.

    Args:
        foci: list of dict from detect_foci(). Each has 'cy', 'cx' keys.
        cell_labels: 2D int array (labeled segmentation). 0 = background,
            1..N = cell IDs. Use segment_nuclei() or watershed-based segmentation.

    Returns:
        dict mapping cell_id → dict with:
            n_foci         -- number of foci in this cell
            foci_list      -- list of focus dicts
            total_intensity -- sum of all foci total_intensity
            mean_foci_area -- mean focus area
            mean_peak_intensity -- mean peak intensity across foci
            foci_density   -- foci per 100 µm² (if pixel_size provided, else per 100 px²)
    """
    result = {}
    H, W = cell_labels.shape

    # Map each focus to its cell
    for focus in foci:
        cy, cx = int(round(focus['cy'])), int(round(focus['cx']))
        if 0 <= cy < H and 0 <= cx < W:
            cell_id = int(cell_labels[cy, cx])
        else:
            cell_id = 0
        if cell_id == 0:
            continue
        if cell_id not in result:
            result[cell_id] = {
                'n_foci': 0, 'foci_list': [],
                'total_intensity': 0.0, 'mean_foci_area': 0.0,
                'mean_peak_intensity': 0.0, 'foci_density': 0.0,
            }
        result[cell_id]['foci_list'].append(focus)
        result[cell_id]['n_foci'] += 1
        result[cell_id]['total_intensity'] += focus['total_intensity']

    # Compute cell areas and aggregate stats
    cell_areas = {}
    for cell_id in np.unique(cell_labels):
        if cell_id > 0:
            cell_areas[cell_id] = int(np.sum(cell_labels == cell_id))

    for cell_id, data in result.items():
        fl = data['foci_list']
        if fl:
            data['mean_foci_area'] = float(np.mean([f['area'] for f in fl]))
            data['mean_peak_intensity'] = float(np.mean([f['peak_intensity'] for f in fl]))
        area_px = cell_areas.get(cell_id, 1)
        data['foci_density'] = 100.0 * data['n_foci'] / area_px

    # Add empty cells
    for cell_id in np.unique(cell_labels):
        if cell_id > 0 and cell_id not in result:
            result[cell_id] = {
                'n_foci': 0, 'foci_list': [],
                'total_intensity': 0.0, 'mean_foci_area': 0.0,
                'mean_peak_intensity': 0.0, 'foci_density': 0.0,
            }

    return result


def spatial_distribution(foci, nuclear_mask, cytoplasm_mask=None):
    """Classify foci as nuclear or cytoplasmic.

    Args:
        foci: list of dict from detect_foci(). Each has 'cy', 'cx' keys.
        nuclear_mask: 2D bool array. True = nucleus region.
        cytoplasm_mask: 2D bool array (optional). True = cytoplasm. If None,
            non-nuclear foci are labeled 'cytoplasmic'.

    Returns:
        dict with:
            nuclear_foci      -- list of foci in nucleus
            cytoplasmic_foci  -- list of foci outside nucleus
            n_nuclear         -- count
            n_cytoplasmic     -- count
            nuclear_fraction  -- fraction of foci that are nuclear
            nuclear_enrichment -- ratio of nuclear vs cytoplasmic density
    """
    H, W = nuclear_mask.shape
    nuclear_area = float(nuclear_mask.sum())
    if cytoplasm_mask is not None:
        cyto_area = float(cytoplasm_mask.sum())
    else:
        cyto_area = float((~nuclear_mask).sum())

    nuclear_foci = []
    cytoplasmic_foci = []

    for focus in foci:
        cy, cx = int(round(focus['cy'])), int(round(focus['cx']))
        if 0 <= cy < H and 0 <= cx < W and nuclear_mask[cy, cx]:
            nuclear_foci.append(focus)
        else:
            cytoplasmic_foci.append(focus)

    n_nuc = len(nuclear_foci)
    n_cyto = len(cytoplasmic_foci)
    total = n_nuc + n_cyto

    nuc_density = n_nuc / nuclear_area if nuclear_area > 0 else 0.0
    cyto_density = n_cyto / cyto_area if cyto_area > 0 else 0.0
    enrichment = nuc_density / cyto_density if cyto_density > 0 else float('inf')

    return {
        'nuclear_foci': nuclear_foci,
        'cytoplasmic_foci': cytoplasmic_foci,
        'n_nuclear': n_nuc,
        'n_cytoplasmic': n_cyto,
        'nuclear_fraction': n_nuc / total if total > 0 else 0.0,
        'nuclear_enrichment': float(enrichment),
    }


# ── Intensity profiling ────────────────────────────────────────────────────────

def foci_intensity_profile(img, foci, max_radius=10):
    """Compute mean radial intensity profile around each focus.

    Useful for assessing focus sharpness, shell structure (ring-like condensates),
    or background subtraction accuracy.

    Args:
        img: 2D float array.
        foci: list of dicts with 'cy', 'cx' keys.
        max_radius: int. Maximum radius for profile in pixels.

    Returns:
        np.ndarray, shape (n_foci, max_radius). Mean intensity at each radial
        distance from the focus center. Row i = focus i, col r = radius r.
        NaN for foci too close to image edge.
    """
    if not foci:
        return np.zeros((0, max_radius))

    gray = img.astype(float)
    H, W = gray.shape
    profiles = np.full((len(foci), max_radius), np.nan)

    for i, focus in enumerate(foci):
        cy, cx = focus['cy'], focus['cx']
        y0, y1 = max(0, int(cy) - max_radius), min(H, int(cy) + max_radius + 1)
        x0, x1 = max(0, int(cx) - max_radius), min(W, int(cx) + max_radius + 1)
        patch = gray[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dists = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        for r in range(max_radius):
            ring = (dists >= r) & (dists < r + 1)
            if ring.any():
                profiles[i, r] = float(patch[ring].mean())

    return profiles


# ── Population-level metrics ───────────────────────────────────────────────────

def stress_response_index(per_cell_data, threshold_foci=3):
    """Fraction of cells with ≥ threshold_foci foci.

    Used to quantify the stress response across a population. A cell is
    considered 'stressed' if it has formed ≥ threshold_foci condensates
    (e.g., stress granules, γH2AX foci under DNA damage).

    Args:
        per_cell_data: dict from foci_per_cell(). Maps cell_id → metrics.
        threshold_foci: int. Minimum foci count to classify cell as stressed.

    Returns:
        dict with:
            stress_index    -- fraction [0..1] of cells with ≥ threshold foci
            n_stressed      -- number of stressed cells
            n_total         -- total number of cells
            threshold_used  -- threshold_foci value used
    """
    n_total = len(per_cell_data)
    if n_total == 0:
        return {'stress_index': 0.0, 'n_stressed': 0, 'n_total': 0,
                'threshold_used': threshold_foci}

    n_stressed = sum(1 for d in per_cell_data.values() if d['n_foci'] >= threshold_foci)
    return {
        'stress_index': n_stressed / n_total,
        'n_stressed': n_stressed,
        'n_total': n_total,
        'threshold_used': threshold_foci,
    }


def aggregation_score(per_cell_data):
    """Per-cell and population aggregate burden score.

    Combines foci count and intensity into a single burden metric.
    Useful for dose-response curves and treatment comparisons.

    Args:
        per_cell_data: dict from foci_per_cell().

    Returns:
        dict with:
            per_cell_scores  -- dict: cell_id → score (n_foci × mean_peak_intensity)
            mean_score       -- population mean score
            median_score     -- population median score
            std_score        -- standard deviation
            max_score        -- maximum score (most aggregated cell)
    """
    scores = {}
    for cell_id, data in per_cell_data.items():
        n = data['n_foci']
        peak = data['mean_peak_intensity'] if n > 0 else 0.0
        scores[cell_id] = float(n * peak)

    vals = list(scores.values())
    return {
        'per_cell_scores': scores,
        'mean_score': float(np.mean(vals)) if vals else 0.0,
        'median_score': float(np.median(vals)) if vals else 0.0,
        'std_score': float(np.std(vals)) if vals else 0.0,
        'max_score': float(max(vals)) if vals else 0.0,
    }


def foci_statistics(foci):
    """Population-level statistics across all detected foci.

    Args:
        foci: list of dicts from detect_foci().

    Returns:
        dict with:
            n_foci          -- total number of foci
            mean_area       -- mean focus area in pixels
            median_area     -- median focus area
            mean_intensity  -- mean of mean_intensity per focus
            mean_peak       -- mean peak_intensity
            total_intensity -- sum of all focus total_intensities
            area_distribution -- sorted list of all areas
    """
    if not foci:
        return {
            'n_foci': 0, 'mean_area': 0.0, 'median_area': 0.0,
            'mean_intensity': 0.0, 'mean_peak': 0.0, 'total_intensity': 0.0,
            'area_distribution': [],
        }

    areas = [f['area'] for f in foci]
    intensities = [f['mean_intensity'] for f in foci]
    peaks = [f['peak_intensity'] for f in foci]
    totals = [f['total_intensity'] for f in foci]

    return {
        'n_foci': len(foci),
        'mean_area': float(np.mean(areas)),
        'median_area': float(np.median(areas)),
        'std_area': float(np.std(areas)),
        'mean_intensity': float(np.mean(intensities)),
        'mean_peak': float(np.mean(peaks)),
        'total_intensity': float(sum(totals)),
        'area_distribution': sorted(areas),
    }


# ── Temporal / dynamic analysis ────────────────────────────────────────────────

def condensate_dynamics(foci_timeseries, pixel_size=1.0, max_distance=5.0):
    """Track foci across time frames: detect formation and dissolution events.

    Matches foci between consecutive frames by proximity (nearest-neighbor).
    Unmatched foci in new frame = formation events; disappearances = dissolution.

    Args:
        foci_timeseries: list of list-of-dicts. One list per frame.
            Each dict has 'cy', 'cx' from detect_foci().
        pixel_size: float. µm per pixel (for distance threshold scaling).
        max_distance: float. Maximum distance in µm to match foci across frames.

    Returns:
        dict with:
            n_formation     -- list, formation events per frame transition
            n_dissolution   -- list, dissolution events per frame transition
            net_change      -- list, net change in foci count per frame
            count_timeseries -- list, n_foci per frame
            formation_rate  -- mean formation events per frame
            dissolution_rate -- mean dissolution events per frame
    """
    max_dist_px = max_distance / pixel_size

    n_frames = len(foci_timeseries)
    count_ts = [len(f) for f in foci_timeseries]
    n_form = []
    n_diss = []

    for t in range(1, n_frames):
        prev = foci_timeseries[t - 1]
        curr = foci_timeseries[t]

        if not prev:
            n_form.append(len(curr))
            n_diss.append(0)
            continue
        if not curr:
            n_form.append(0)
            n_diss.append(len(prev))
            continue

        prev_pts = np.array([[f['cy'], f['cx']] for f in prev])
        curr_pts = np.array([[f['cy'], f['cx']] for f in curr])

        matched_prev = set()
        matched_curr = set()

        for j, cp in enumerate(curr_pts):
            dists = np.sqrt(np.sum((prev_pts - cp) ** 2, axis=1))
            nearest = int(np.argmin(dists))
            if dists[nearest] <= max_dist_px:
                matched_prev.add(nearest)
                matched_curr.add(j)

        formed = len(curr) - len(matched_curr)
        dissolved = len(prev) - len(matched_prev)
        n_form.append(formed)
        n_diss.append(dissolved)

    net_change = [n_form[i] - n_diss[i] for i in range(len(n_form))]

    return {
        'n_formation': n_form,
        'n_dissolution': n_diss,
        'net_change': net_change,
        'count_timeseries': count_ts,
        'formation_rate': float(np.mean(n_form)) if n_form else 0.0,
        'dissolution_rate': float(np.mean(n_diss)) if n_diss else 0.0,
    }


# ── Full pipeline ──────────────────────────────────────────────────────────────

def condensate_analysis(img, cell_labels=None, nuclear_mask=None,
                        min_sigma=0.8, max_sigma=5.0, threshold=0.05,
                        pixel_size=1.0):
    """Complete condensate analysis pipeline.

    Detects foci, assigns to cells (if labels provided), and computes
    spatial distribution, stress index, and aggregate burden score.

    Args:
        img: 2D float array. Fluorescence channel showing condensates.
        cell_labels: 2D int array (optional). Labeled cell segmentation.
        nuclear_mask: 2D bool array (optional). Nuclear regions.
        min_sigma: float. Minimum blob sigma for LoG detection.
        max_sigma: float. Maximum blob sigma for LoG detection.
        threshold: float. LoG threshold.
        pixel_size: float. µm per pixel (for density calculations).

    Returns:
        dict with:
            foci             -- list of detected foci
            stats            -- population statistics (foci_statistics)
            per_cell         -- per-cell data (if cell_labels provided)
            spatial          -- nuclear/cytoplasmic split (if nuclear_mask)
            stress_index     -- fraction stressed cells (if cell_labels)
            aggregation      -- aggregation burden score (if cell_labels)
    """
    foci = detect_foci(img, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    stats = foci_statistics(foci)

    result = {'foci': foci, 'stats': stats, 'pixel_size': pixel_size}

    if cell_labels is not None:
        per_cell = foci_per_cell(foci, cell_labels)
        result['per_cell'] = per_cell
        result['stress_index'] = stress_response_index(per_cell)
        result['aggregation'] = aggregation_score(per_cell)

    if nuclear_mask is not None:
        result['spatial'] = spatial_distribution(foci, nuclear_mask)

    return result
