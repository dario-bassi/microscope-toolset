"""Cell cycle phase analysis from microscopy images.

Classifies cells into cell cycle phases (G1, S, G2, M) based on
nuclear morphology, DNA content (intensity), and shape features.

Functions:
    dna_content_histogram -- DNA content distribution from nuclear intensity
    classify_cycle_phase  -- Assign cells to G1/S/G2/M phases
    mitotic_index         -- Fraction of cells in mitosis
    proliferation_markers -- Count cells positive for Ki-67 or similar markers
    cycle_phase_summary   -- Population statistics by phase
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure


def dna_content_histogram(intensities, n_bins=50):
    """Compute DNA content distribution from nuclear intensities.

    In a population of cycling cells, the DNA content histogram
    shows peaks at 2N (G1) and 4N (G2/M), with S-phase cells in between.

    Args:
        intensities: 1D array of integrated nuclear intensities (sum, not mean).
        n_bins: int, number of histogram bins.

    Returns:
        dict with:
            bin_centers: array, center of each bin.
            counts: array, number of cells per bin.
            frequencies: array, fraction per bin.
            g1_peak: float, estimated 2N peak position.
            g2_peak: float, estimated 4N peak position.
            g1_g2_ratio: float, G2 peak / G1 peak (should be ~2.0).
            cv_g1: float, coefficient of variation of G1 peak.
    """
    intensities = np.asarray(intensities, dtype=float)
    if len(intensities) < 5:
        return {
            'bin_centers': np.array([]),
            'counts': np.array([]),
            'frequencies': np.array([]),
            'g1_peak': 0.0,
            'g2_peak': 0.0,
            'g1_g2_ratio': 0.0,
            'cv_g1': 0.0,
        }

    counts, bin_edges = np.histogram(intensities, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    frequencies = counts / max(counts.sum(), 1)

    # Smooth histogram to find peaks
    if len(counts) > 5:
        smoothed = ndimage.uniform_filter1d(counts.astype(float), 3)
    else:
        smoothed = counts.astype(float)

    # Find G1 peak: largest peak in lower half of range
    mid = len(smoothed) // 2
    g1_idx = int(np.argmax(smoothed[:mid]))
    g1_peak = float(bin_centers[g1_idx])

    # Find G2 peak: largest peak in upper half, or near 2× G1
    # Search around 2× G1 position
    target_g2 = g1_peak * 2
    g2_search_start = max(mid - 5, 0)
    g2_region = smoothed[g2_search_start:]
    if len(g2_region) > 0:
        g2_idx = g2_search_start + int(np.argmax(g2_region))
        g2_peak = float(bin_centers[min(g2_idx, len(bin_centers) - 1)])
    else:
        g2_peak = target_g2

    g1_g2_ratio = g2_peak / g1_peak if g1_peak > 0 else 0.0

    # Estimate G1 CV
    g1_half_width = (bin_edges[1] - bin_edges[0]) * 3
    g1_cells = intensities[
        (intensities > g1_peak - g1_half_width) &
        (intensities < g1_peak + g1_half_width)
    ]
    cv_g1 = float(g1_cells.std() / g1_cells.mean()) if len(g1_cells) > 1 and g1_cells.mean() > 0 else 0.0

    return {
        'bin_centers': bin_centers,
        'counts': counts,
        'frequencies': frequencies,
        'g1_peak': round(g1_peak, 2),
        'g2_peak': round(g2_peak, 2),
        'g1_g2_ratio': round(g1_g2_ratio, 4),
        'cv_g1': round(cv_g1, 4),
    }


def classify_cycle_phase(intensities, areas, circularities=None,
                         g1_peak=None):
    """Classify cells into cell cycle phases.

    Uses DNA content (intensity), nuclear area, and optionally
    circularity to assign each cell to G1, S, G2, or M phase.

    Args:
        intensities: 1D array, integrated nuclear intensity.
        areas: 1D array, nuclear area in pixels.
        circularities: 1D array or None, nuclear circularity (4π*A/P²).
        g1_peak: float or None, known G1 peak intensity. Auto-detected
            if None.

    Returns:
        dict with:
            phases: list of str, phase assignment per cell.
            phase_counts: dict mapping phase -> count.
            g1_threshold: float, upper bound for G1.
            g2_threshold: float, lower bound for G2/M.
    """
    intensities = np.asarray(intensities, dtype=float)
    areas = np.asarray(areas, dtype=float)
    n = len(intensities)

    if n == 0:
        return {
            'phases': [],
            'phase_counts': {'G1': 0, 'S': 0, 'G2': 0, 'M': 0},
            'g1_threshold': 0.0,
            'g2_threshold': 0.0,
        }

    # Auto-detect G1 peak if not provided
    if g1_peak is None:
        hist = dna_content_histogram(intensities)
        g1_peak = hist['g1_peak']

    if g1_peak <= 0:
        g1_peak = float(np.median(intensities))

    # Thresholds based on G1 peak
    g1_upper = g1_peak * 1.3   # upper bound for G1
    g2_lower = g1_peak * 1.7   # lower bound for G2/M
    # S-phase: between g1_upper and g2_lower
    # M-phase: G2-level intensity but low circularity (condensed chromatin)

    phases = []
    for i in range(n):
        inten = intensities[i]
        area = areas[i]

        if inten < g1_upper:
            phases.append('G1')
        elif inten > g2_lower:
            # Check for mitotic features if circularity available
            if circularities is not None and circularities[i] < 0.6:
                phases.append('M')
            elif area > np.median(areas) * 1.5:
                # Large nucleus at G2 level → could be M
                phases.append('M')
            else:
                phases.append('G2')
        else:
            phases.append('S')

    phase_counts = {p: phases.count(p) for p in ['G1', 'S', 'G2', 'M']}

    return {
        'phases': phases,
        'phase_counts': phase_counts,
        'g1_threshold': round(float(g1_upper), 2),
        'g2_threshold': round(float(g2_lower), 2),
    }


def mitotic_index(phases=None, n_mitotic=None, n_total=None):
    """Compute mitotic index (fraction of cells in mitosis).

    Args:
        phases: list of str phase labels (from classify_cycle_phase).
            Counts 'M' phase cells.
        n_mitotic: int, direct count of mitotic cells (alternative to phases).
        n_total: int, total cell count (required if n_mitotic provided).

    Returns:
        dict with:
            mitotic_index: float (0-1).
            n_mitotic: int.
            n_total: int.
            percentage: float (0-100).
    """
    if phases is not None:
        n_m = sum(1 for p in phases if p == 'M')
        n_t = len(phases)
    elif n_mitotic is not None and n_total is not None:
        n_m = n_mitotic
        n_t = n_total
    else:
        raise ValueError("Provide either 'phases' or both 'n_mitotic' and 'n_total'")

    mi = n_m / n_t if n_t > 0 else 0.0

    return {
        'mitotic_index': round(mi, 4),
        'n_mitotic': n_m,
        'n_total': n_t,
        'percentage': round(mi * 100, 2),
    }


def population_index(labels, positive_label):
    """Compute the fraction of a population with a given label.

    Generalises mitotic_index() to arbitrary binary classifications
    such as budding index, viability, or proliferation fraction.

    Args:
        labels: List/array of classification labels (str or int).
        positive_label: The label to count as "positive".

    Returns:
        dict with:
            index: float (0-1).
            n_positive: int.
            n_total: int.
            percentage: float (0-100).
    """
    labels = list(labels)
    n_t = len(labels)
    n_p = sum(1 for l in labels if l == positive_label)
    idx = n_p / n_t if n_t > 0 else 0.0
    return {
        'index': round(idx, 4),
        'n_positive': n_p,
        'n_total': n_t,
        'percentage': round(idx * 100, 2),
    }


def proliferation_markers(intensities, threshold=None, threshold_method='otsu'):
    """Count cells positive for a proliferation marker (e.g., Ki-67).

    Args:
        intensities: 1D array, marker intensity per cell.
        threshold: float or None. If None, auto-detect.
        threshold_method: str, 'otsu', 'median', or 'percentile'.

    Returns:
        dict with:
            n_positive: int.
            n_negative: int.
            n_total: int.
            fraction_positive: float.
            threshold_used: float.
    """
    intensities = np.asarray(intensities, dtype=float)
    n = len(intensities)

    if n == 0:
        return {
            'n_positive': 0, 'n_negative': 0, 'n_total': 0,
            'fraction_positive': 0.0, 'threshold_used': 0.0,
        }

    if threshold is None:
        if threshold_method == 'otsu':
            # Simple Otsu: minimize intra-class variance
            sorted_vals = np.sort(intensities)
            best_t = float(np.median(intensities))
            best_var = float('inf')
            for t in np.linspace(sorted_vals[1], sorted_vals[-2], 50):
                low = intensities[intensities <= t]
                high = intensities[intensities > t]
                if len(low) == 0 or len(high) == 0:
                    continue
                w0 = len(low) / n
                w1 = len(high) / n
                var = w0 * low.var() + w1 * high.var()
                if var < best_var:
                    best_var = var
                    best_t = t
            threshold = best_t
        elif threshold_method == 'median':
            threshold = float(np.median(intensities) * 1.5)
        elif threshold_method == 'percentile':
            threshold = float(np.percentile(intensities, 75))
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

    n_pos = int((intensities > threshold).sum())
    n_neg = n - n_pos

    return {
        'n_positive': n_pos,
        'n_negative': n_neg,
        'n_total': n,
        'fraction_positive': round(n_pos / n, 4) if n > 0 else 0.0,
        'threshold_used': round(float(threshold), 4),
    }


def cycle_phase_summary(phases, intensities=None, areas=None):
    """Population statistics by cell cycle phase.

    Args:
        phases: list of str, phase per cell ('G1', 'S', 'G2', 'M').
        intensities: optional 1D array, DNA content per cell.
        areas: optional 1D array, nuclear area per cell.

    Returns:
        dict mapping phase -> {
            count: int, fraction: float,
            mean_intensity: float (if provided),
            mean_area: float (if provided).
        }
    """
    phases = list(phases)
    n = len(phases)
    if intensities is not None:
        intensities = np.asarray(intensities, dtype=float)
    if areas is not None:
        areas = np.asarray(areas, dtype=float)

    summary = {}
    for phase in ['G1', 'S', 'G2', 'M']:
        indices = [i for i, p in enumerate(phases) if p == phase]
        count = len(indices)
        entry = {
            'count': count,
            'fraction': round(count / n, 4) if n > 0 else 0.0,
        }
        if intensities is not None and count > 0:
            entry['mean_intensity'] = round(float(intensities[indices].mean()), 2)
        if areas is not None and count > 0:
            entry['mean_area'] = round(float(areas[indices].mean()), 2)
        summary[phase] = entry

    return summary


def classify_fucci(nuc_image, gem_image, sigma=2.0,
                   ratio_threshold=2.5, gem_min=15,
                   min_cell_size=25):
    """Classify cells into FUCCI phases (G1/S/G2/M) from dual reporter images.

    FUCCI reporters:
      - Cdt1-mCherry (nuc_image): Bright in G1, dim in S/G2/M
      - Geminin-GFP (gem_image):  Bright in G2/M, dim in G1

    Args:
        nuc_image: 2D numpy array, nucleus-channel image (Cdt1/mCherry).
        gem_image: 2D numpy array, geminin-channel image (Geminin/GFP).
        sigma: float, Gaussian smoothing sigma for segmentation.
        ratio_threshold: float, gem/nuc ratio above which cell is G2/M.
        gem_min: float, minimum peak geminin intensity for G2/M.
        min_cell_size: int, minimum cell area in pixels.

    Returns:
        dict with:
            cells: list of dicts with x, y, phase, nuc_max, gem_max, ratio.
            n_cells: int, total cells detected.
            n_G1: int, G1 cells.
            n_S: int, S-phase cells.
            n_G2M: int, G2/M cells.
            g2m_cells: list of G2/M cell dicts sorted by gem_max descending.
            labeled: 2D array, cell label mask.
    """
    from skimage import filters, measure, morphology as morph
    from skimage.filters import gaussian as sk_gaussian
    from scipy import ndimage

    nuc = np.asarray(nuc_image, dtype=float)
    gem = np.asarray(gem_image, dtype=float)

    # Smooth and segment
    nuc_s = sk_gaussian(nuc, sigma=sigma)
    gem_s = sk_gaussian(gem, sigma=sigma)
    combined = np.maximum(nuc_s, gem_s)

    try:
        thresh = filters.threshold_otsu(combined)
    except Exception:
        thresh = combined.mean() + combined.std()

    mask = combined > thresh
    mask = ndimage.binary_fill_holes(mask)
    mask = morph.remove_small_objects(mask, max_size=min_cell_size - 1)
    labeled, n_cells = ndimage.label(mask)

    props = measure.regionprops(labeled)
    cells = []
    for p in props:
        lab = p.label
        cell_nuc = float(nuc[labeled == lab].max())
        cell_gem = float(gem[labeled == lab].max())
        cy, cx = p.centroid
        ratio = cell_gem / (cell_nuc + 1e-6)

        if ratio > ratio_threshold and cell_gem > gem_min:
            phase = 'G2/M'
        elif cell_nuc > cell_gem * ratio_threshold:
            phase = 'G1'
        else:
            phase = 'S'

        cells.append({
            'x': float(cx), 'y': float(cy),
            'area': int(p.area),
            'nuc_max': cell_nuc,
            'gem_max': cell_gem,
            'ratio': float(ratio),
            'phase': phase,
        })

    g2m_cells = sorted([c for c in cells if c['phase'] == 'G2/M'],
                        key=lambda c: -c['gem_max'])

    return {
        'cells': cells,
        'n_cells': len(cells),
        'n_G1': sum(1 for c in cells if c['phase'] == 'G1'),
        'n_S': sum(1 for c in cells if c['phase'] == 'S'),
        'n_G2M': len(g2m_cells),
        'g2m_cells': g2m_cells,
        'labeled': labeled,
    }


def detect_geminin_division(frames, brightest_frame=None, roi_radius=50,
                             drop_threshold=0.40):
    """Detect cell division from geminin signal drop in timelapse.

    During M->G1 transition, Geminin is rapidly degraded, causing signal drop.

    Args:
        frames: list of 2D numpy arrays (geminin-channel timelapse).
        brightest_frame: int or None, index of frame with brightest spot.
            If None, uses first frame.
        roi_radius: int, radius around brightest pixel for ROI.
        drop_threshold: float, fraction of baseline below which = division.

    Returns:
        dict with:
            intensities: list of float, mean intensity per frame in ROI.
            division_observed: bool.
            division_frame: int or None, 1-indexed frame when division detected.
            roi_center: (row, col) of brightest pixel.
            roi_mask: 2D bool array.
    """
    from scipy.ndimage import gaussian_filter

    if not frames:
        return {
            'intensities': [],
            'division_observed': False,
            'division_frame': None,
            'roi_center': (256, 256),
            'roi_mask': np.zeros((512, 512), dtype=bool),
        }

    ref_idx = brightest_frame if brightest_frame is not None else 0
    ref_frame = np.asarray(frames[ref_idx], dtype=float)

    # Find brightest spot in reference frame
    smooth_ref = gaussian_filter(ref_frame, sigma=3)
    yr, xr = np.unravel_index(smooth_ref.argmax(), smooth_ref.shape)
    roi_center = (int(yr), int(xr))

    # Build ROI mask
    h, w = ref_frame.shape
    yy, xx = np.ogrid[:h, :w]
    roi_mask = (yy - yr)**2 + (xx - xr)**2 <= roi_radius**2

    # Compute per-frame intensities
    intensities = []
    for f in frames:
        g = np.asarray(f, dtype=float)
        val = float(g[roi_mask].mean()) if roi_mask.any() else 0.0
        intensities.append(val)

    baseline = intensities[0] if intensities else 0.0
    threshold = baseline * drop_threshold

    division_frame = None
    for i, intensity in enumerate(intensities[1:], start=2):
        if intensity < threshold:
            division_frame = i
            break

    return {
        'intensities': intensities,
        'division_observed': division_frame is not None,
        'division_frame': division_frame,
        'roi_center': roi_center,
        'roi_mask': roi_mask,
    }


def detect_fucci_g2m(gem_img, threshold=0.65, nuc_img=None, ratio_min=None):
    """Detect G2/M phase cells using FUCCI geminin-GFP reporter.

    G2/M cells are bright in geminin-channel. Uses direct intensity threshold
    on normalized image. Correct threshold separates G2/M (0.80-1.00) from
    S-phase (0.25-0.50) and G1 (0.05) cells.

    Optionally, pass nuc_img for ratio-based confirmation (gem/nuc > ratio_min)
    to improve robustness against debris or autofluorescence.

    Args:
        gem_img: 2D array, geminin-channel image (Geminin-GFP).
        threshold: float, normalized intensity threshold (default 0.65).
            G2/M cells: normalized ~0.80-1.00
            S-phase cells: normalized ~0.25-0.50 (excluded by > 0.65)
            G1 cells: normalized ~0.05 (excluded)
        nuc_img: 2D array or None, nucleus-channel (Cdt1-mCherry).
            If provided, filters regions by gem/nuc intensity ratio.
        ratio_min: float or None, minimum gem/nuc ratio (default None = no filter).
            G2/M cells: ratio ~10-15 (gem bright, nuc dim)
            S cells: ratio ~0.7 (both moderate)
            G1 cells: ratio ~0.05 (nuc bright, gem dim)
            Recommended: ratio_min=3.0 for secondary confirmation.

    Returns:
        dict with:
            n_G2M: int, number of G2/M cells detected
            positions: list of (x, y) pixel positions (centroid of each region)
            labels: labeled array of G2/M regions
            binary: boolean mask of G2/M regions
            intensities: list of max gem intensity per cell
            ratios: list of gem/nuc intensity ratio per cell (empty if nuc_img=None)
    """
    from scipy.ndimage import label, center_of_mass

    gem_norm = gem_img / (gem_img.max() + 1e-6)
    binary = gem_norm > threshold
    labeled, n = label(binary)

    positions = []
    intensities = []
    ratios = []
    kept_labels = []

    if n > 0:
        coms = center_of_mass(binary, labeled, range(1, n + 1))
        for i, (row, col) in enumerate(coms):
            r, c = int(row), int(col)
            win = 4
            r_lo, r_hi = max(0, r-win), min(gem_img.shape[0], r+win+1)
            c_lo, c_hi = max(0, c-win), min(gem_img.shape[1], c+win+1)
            gem_val = float(gem_img[r_lo:r_hi, c_lo:c_hi].max())

            # Optional ratio filter with nucleus channel
            ratio = None
            if nuc_img is not None:
                nuc_val = float(nuc_img[r_lo:r_hi, c_lo:c_hi].max())
                ratio = gem_val / max(nuc_val, 1.0)
                ratios.append(ratio)
                if ratio_min is not None and ratio < ratio_min:
                    continue  # skip cells that fail ratio filter

            positions.append((float(col), float(row)))  # (x, y) = (col, row)
            intensities.append(gem_val)
            kept_labels.append(i + 1)

    # Rebuild labeled array for kept cells only
    if kept_labels and len(kept_labels) < n:
        new_labeled = np.zeros_like(labeled)
        for new_lbl, old_lbl in enumerate(kept_labels, 1):
            new_labeled[labeled == old_lbl] = new_lbl
        labeled = new_labeled
        binary = new_labeled > 0

    return {
        'n_G2M': len(positions),
        'positions': positions,
        'labels': labeled,
        'binary': binary,
        'intensities': intensities,
        'ratios': ratios,
    }


def detect_fucci_division(timelapse, baseline_frames=3, drop_threshold=0.4,
                          sudden_drop_fraction=0.30):
    """Detect cell division from geminin-GFP timelapse.

    Division is signaled by a sharp drop in Geminin signal (degrades M→G1).
    Uses two detection modes:

    Mode 1 (absolute): signal drops below drop_threshold × baseline.
        - Requires strong degradation (e.g., cell leaves FOV or geminin fully degrades)
        - drop_threshold=0.4 means <40% of baseline

    Mode 2 (relative/sudden): consecutive frame drop > sudden_drop_fraction.
        - Detects sudden drops even if absolute level doesn't reach threshold
        - sudden_drop_fraction=0.30 means >30% drop in a single frame
        - This catches cases where both daughter cells remain in FOV
          (signal halves but doesn't reach <40% of baseline)

    Args:
        timelapse: list of 2D arrays, geminin-channel images over time.
        baseline_frames: int, number of initial frames to use for baseline.
        drop_threshold: float, signal must drop below this fraction of baseline
            (mode 1). Default 0.4 (40% of baseline).
        sudden_drop_fraction: float, single-frame drop fraction for mode 2 detection.
            Default 0.30 (30% drop in one frame = division). Set to None to disable.

    Returns:
        dict with:
            division_observed: bool
            division_frame: int or None (1-indexed, first frame with drop)
            division_mode: str or None ('absolute' or 'sudden')
            baseline_mean: float, mean intensity of baseline frames
            frame_means: list of per-frame mean intensities
            drop_fraction: float, total signal drop from baseline to end
    """
    if not timelapse:
        return {
            'division_observed': False,
            'division_frame': None,
            'division_mode': None,
            'baseline_mean': 0.0,
            'frame_means': [],
            'drop_fraction': 0.0,
        }

    frame_means = [f.mean() for f in timelapse]
    n_base = min(baseline_frames, len(frame_means))
    baseline_mean = float(np.mean(frame_means[:n_base]))

    division_frame = None
    division_observed = False
    division_mode = None

    for i, v in enumerate(frame_means):
        if i < n_base - 1:
            continue

        # Mode 1: absolute drop below threshold
        if v < baseline_mean * drop_threshold:
            division_frame = i + 1
            division_observed = True
            division_mode = 'absolute'
            break

        # Mode 2: sudden single-frame drop
        if sudden_drop_fraction is not None and i > 0:
            prev = frame_means[i - 1]
            if prev > 0:
                frame_drop = (prev - v) / prev
                if frame_drop > sudden_drop_fraction:
                    division_frame = i + 1
                    division_observed = True
                    division_mode = 'sudden'
                    break

    last3 = frame_means[-3:] if len(frame_means) >= 3 else frame_means
    final_mean = float(np.mean(last3))
    drop_fraction = 1.0 - final_mean / max(baseline_mean, 1e-6)

    return {
        'division_observed': division_observed,
        'division_frame': division_frame,
        'division_mode': division_mode,
        'baseline_mean': baseline_mean,
        'frame_means': frame_means,
        'drop_fraction': drop_fraction,
    }


def detect_fucci_division_v2(timelapse, baseline_frames=5, drop_threshold=0.40,
                              sudden_drop_fraction=0.25, slope_pvalue=0.05):
    """Enhanced Geminin division detection with three modes + confidence score.

    Improvements over v1:
    - Longer baseline (5 frames) for more robust baseline estimate
    - More sensitive sudden-drop (0.25 vs 0.30)
    - Mode 3: slope detection — sustained negative linear trend over post-baseline
    - Returns confidence_score (0-1): how confident we are in the division call
    - Handles borderline cases that v1 missed (ch538: 44% drop just above 50% threshold)

    Modes:
        1. Absolute: mean of last 3 frames < drop_threshold × baseline
        2. Sudden: any single-frame drop > sudden_drop_fraction of previous frame
        3. Slope: linear regression over post-baseline frames has significant
           negative slope (signal clearly declining toward division)

    Args:
        timelapse: list of 2D arrays, geminin-channel images.
        baseline_frames: int, frames for baseline (default 5, more robust than 3).
        drop_threshold: float, absolute threshold fraction (default 0.40).
        sudden_drop_fraction: float, single-frame drop fraction (default 0.25).
        slope_pvalue: float, p-value threshold for slope significance (default 0.05).

    Returns:
        dict with all v1 keys plus:
            division_mode: str ('absolute', 'sudden', 'slope', or None)
            confidence_score: float 0-1 (1=certain, 0.5=borderline)
    """
    if not timelapse:
        return {
            'division_observed': False, 'division_frame': None,
            'division_mode': None, 'baseline_mean': 0.0,
            'frame_means': [], 'drop_fraction': 0.0, 'confidence_score': 0.0,
        }

    frame_means = [float(f.mean()) for f in timelapse]
    n_base = min(baseline_frames, max(1, len(frame_means) // 2))
    baseline_mean = float(np.mean(frame_means[:n_base]))

    division_frame = None
    division_observed = False
    division_mode = None
    confidence_score = 0.0

    # Mode 1 & 2: frame-by-frame scan (same as v1 but with updated defaults)
    for i, v in enumerate(frame_means):
        if i < n_base - 1:
            continue

        if v < baseline_mean * drop_threshold:
            division_frame = i + 1
            division_observed = True
            division_mode = 'absolute'
            # Confidence proportional to how far below threshold
            ratio = v / max(baseline_mean, 1e-6)
            confidence_score = min(1.0, 1.0 - ratio / drop_threshold)
            break

        if sudden_drop_fraction is not None and i > 0:
            prev = frame_means[i - 1]
            if prev > 0:
                frame_drop = (prev - v) / prev
                if frame_drop > sudden_drop_fraction:
                    division_frame = i + 1
                    division_observed = True
                    division_mode = 'sudden'
                    confidence_score = min(1.0, frame_drop / sudden_drop_fraction * 0.8)
                    break

    # Mode 3: slope detection over post-baseline frames (if not already detected)
    if not division_observed and len(frame_means) > n_base + 3:
        from scipy.stats import linregress
        post_frames = frame_means[n_base:]
        xs = np.arange(len(post_frames), dtype=float)
        slope, intercept, r_value, p_value, _ = linregress(xs, post_frames)
        # Slope is significantly negative AND total drop is meaningful
        total_drop = 1.0 - (intercept + slope * (len(post_frames) - 1)) / max(baseline_mean, 1e-6)
        if slope < 0 and p_value < slope_pvalue and total_drop > 0.20:
            # Estimate where signal would cross threshold (linear extrapolation)
            if baseline_mean > 0:
                frames_to_threshold = (baseline_mean * drop_threshold - intercept) / slope
                est_frame = int(n_base + max(0, frames_to_threshold))
                division_frame = min(est_frame, len(frame_means))
            else:
                division_frame = n_base + 1
            division_observed = True
            division_mode = 'slope'
            confidence_score = min(0.7, (1 - p_value) * total_drop)

    last3 = frame_means[-3:] if len(frame_means) >= 3 else frame_means
    final_mean = float(np.mean(last3))
    drop_fraction = 1.0 - final_mean / max(baseline_mean, 1e-6)

    return {
        'division_observed': division_observed,
        'division_frame': division_frame,
        'division_mode': division_mode,
        'baseline_mean': baseline_mean,
        'frame_means': frame_means,
        'drop_fraction': drop_fraction,
        'confidence_score': float(confidence_score),
    }
