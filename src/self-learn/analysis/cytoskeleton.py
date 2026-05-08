"""Cytoskeleton analysis — actin fiber detection and classification.

Provides functions for analyzing F-actin (phalloidin-stained) stress fibers
in fluorescence microscopy images. Implements the standard pipeline for
fibroblast stress fiber subtype classification:

  1. Enhance fibers: Frangi vesselness filter (tubular ridge detection)
  2. Threshold: create binary fiber mask
  3. Skeletonize: reduce fibers to 1-px skeleton
  4. Label components: each connected skeleton segment = one fiber
  5. Classify: VSF / DSF / PAC / TA based on morphological features

Stress fiber subtypes:
  - VSF (Ventral Stress Fiber): thick, bright, spans full cell, BOTH ends
    anchored at focal adhesions at the cell periphery.
  - DSF (Dorsal Stress Fiber): thinner, connects ONE focal adhesion to a
    transverse arc or cell center — does NOT span full length.
  - PAC (Perinuclear Actin Cap): thick fibers arching OVER the nucleus,
    brightest signal, short span.
  - TA  (Transverse Arc): faint curved fibers running PERPENDICULAR to
    the cell's long axis, between ventral fibers.

Functions:
    enhance_fibers         -- Frangi vesselness filter for fiber enhancement
    segment_fibers         -- Binary fiber mask from enhanced image
    classify_fiber_subtype -- Single-fiber VSF/DSF/PAC/TA classification
    analyze_cell_fibers    -- Full per-cell pipeline: enhance → skel → classify
    fiber_differential     -- Aggregate counts + fractions across many cells
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_erosion, gaussian_filter
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label as sk_label, regionprops
from collections import Counter


def enhance_fibers(img, sigmas=None, mask=None):
    """Apply Frangi vesselness filter to enhance tubular fiber structures.

    The Frangi filter highlights elongated, tubular structures of varying
    widths. Appropriate for actin stress fibers, neurites, blood vessels.

    Args:
        img: 2D float array. If 3D (H, W, C), uses first channel.
        sigmas: iterable of floats, width scales to detect.
            Default range(1, 4) covers thin to medium fibers.
        mask: optional 2D bool array. If provided, zeroes out enhanced
            image outside the mask (e.g. cell boundary).

    Returns:
        2D float array (same shape as input), vesselness response.
        Values are 0-1 normalized, with 1 = strongest fiber response.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img[:, :, 0]

    if sigmas is None:
        sigmas = range(1, 4)

    # Normalize to 0-1 range (Frangi works best with normalized input)
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img)
    img_norm = (img - mn) / (mx - mn)

    if mask is not None:
        img_norm[~mask] = 0

    enhanced = frangi(img_norm, sigmas=sigmas, black_ridges=False)

    if mask is not None:
        enhanced[~mask] = 0

    return enhanced


def segment_fibers(enhanced, threshold_sigma=0.5, mask=None, min_fiber_length=5):
    """Create binary fiber mask from Frangi-enhanced image.

    Args:
        enhanced: 2D float array from enhance_fibers().
        threshold_sigma: float, threshold = mean + sigma * std of the
            non-zero region. Lower = more fibers detected.
        mask: optional 2D bool array (cell boundary mask).
        min_fiber_length: int, minimum skeleton length in pixels.

    Returns:
        fiber_mask: 2D bool array, True where fibers are detected.
        skeleton: 2D bool array, skeletonized fiber mask.
    """
    region = enhanced[mask] if mask is not None else enhanced[enhanced > 0]
    if len(region) == 0 or region.max() == 0:
        empty = np.zeros(enhanced.shape, dtype=bool)
        return empty, empty

    thresh = region.mean() + threshold_sigma * region.std()
    fiber_mask = enhanced > thresh

    if mask is not None:
        fiber_mask &= mask

    if fiber_mask.any():
        fiber_mask = remove_small_objects(fiber_mask, max_size=min_fiber_length)
        fiber_mask = binary_fill_holes(fiber_mask)

    skeleton = skeletonize(fiber_mask) if fiber_mask.any() else np.zeros_like(fiber_mask)

    return fiber_mask, skeleton


def classify_fiber_subtype(skel_comp, cell_prop, nuc_center, img_factin,
                           cell_mask, cell_length_px,
                           pac_nuc_dist=40, pac_length_frac=0.4,
                           ta_angle_deg=50, vsf_length_frac=0.4,
                           n_boundary_endpoints=2):
    """Classify a single fiber skeleton component into VSF/DSF/PAC/TA.

    Decision hierarchy (in priority order):
    1. PAC: fiber centroid close to nucleus (<pac_nuc_dist px), short
    2. TA:  fiber oriented perpendicular to cell major axis (>ta_angle_deg),
            signal fainter than cell average
    3. VSF: long (>vsf_length_frac * cell_length), both ends at cell boundary
    4. DSF: one end at cell boundary (default for medium-length fibers)
    Fallback: classify by length/angle if none of the above match.

    Args:
        skel_comp: 2D bool array, single fiber skeleton component.
        cell_prop: skimage RegionProps for the containing cell.
        nuc_center: (row, col) array or None — nucleus centroid.
        img_factin: 2D float array, F-actin channel (for intensity).
        cell_mask: 2D bool array, the cell boundary mask.
        cell_length_px: float, major axis length of the cell in pixels.
        pac_nuc_dist: float, max nucleus distance for PAC classification.
        pac_length_frac: float, max length fraction for PAC classification.
        ta_angle_deg: float, min perpendicular angle for TA classification.
        vsf_length_frac: float, min length fraction for VSF classification.
        n_boundary_endpoints: int, min endpoints at boundary for VSF.
            Default 2 (both ends anchored).

    Returns:
        subtype: str, one of 'VSF', 'DSF', 'PAC', 'TA'.
        features: dict with classification features for diagnostics.
    """
    fib_labeled = sk_label(skel_comp.astype(np.uint8))
    fib_props = regionprops(fib_labeled)
    if not fib_props:
        return 'unknown', {}

    fp = fib_props[0]
    centroid = np.array(fp.centroid)
    length = fp.axis_major_length
    orientation = fp.orientation  # radians

    # Mean intensity of fiber pixels
    fib_pixels = img_factin[skel_comp > 0]
    mean_int = float(np.mean(fib_pixels)) if len(fib_pixels) > 0 else 0.0
    cell_mean_int = float(img_factin[cell_mask].mean()) if cell_mask.any() else 0.0

    # Angle between fiber and cell major axis
    cell_orient = cell_prop.orientation
    fiber_angle = abs(np.degrees(orientation - cell_orient))
    while fiber_angle > 90:
        fiber_angle = 180 - fiber_angle  # fold to [0, 90]

    # Distance from fiber centroid to nucleus
    nuc_dist = 999.0
    if nuc_center is not None:
        nuc_center = np.asarray(nuc_center)
        if nuc_center.ndim == 1 and len(nuc_center) >= 2:
            nuc_dist = float(np.linalg.norm(centroid - nuc_center[:2]))

    # Endpoint detection: pixels with only 1 neighbor in skel
    coords = np.argwhere(skel_comp)
    if len(coords) < 3:
        return 'unknown', {}

    endpoints = []
    for r, c in coords:
        neighbors = int(skel_comp[max(0, r-1):r+2, max(0, c-1):c+2].sum()) - 1
        if neighbors == 1:
            endpoints.append(np.array([r, c]))

    # Count endpoints near cell boundary (not in eroded mask)
    eroded = binary_erosion(cell_mask, iterations=8)
    n_at_boundary = sum(
        1 for ep in endpoints
        if 0 <= int(ep[0]) < eroded.shape[0] and 0 <= int(ep[1]) < eroded.shape[1]
        and not eroded[int(ep[0]), int(ep[1])]
    )

    length_frac = length / max(1.0, cell_length_px)

    features = {
        'length_px': float(length),
        'length_frac': float(length_frac),
        'fiber_angle_deg': float(fiber_angle),
        'mean_intensity': float(mean_int),
        'cell_mean_intensity': float(cell_mean_int),
        'nuc_dist_px': float(nuc_dist),
        'n_endpoints': len(endpoints),
        'n_at_boundary': n_at_boundary,
    }

    # 1. PAC: near nucleus, bright, short
    if nuc_dist < pac_nuc_dist and length_frac < pac_length_frac:
        return 'PAC', features

    # 2. TA: perpendicular to cell axis, faint
    if fiber_angle > ta_angle_deg and mean_int < cell_mean_int * 1.2:
        return 'TA', features

    # 3. VSF: long, both ends at boundary
    if length_frac >= vsf_length_frac and n_at_boundary >= n_boundary_endpoints:
        return 'VSF', features

    # 4. DSF: one end at boundary
    if n_at_boundary == 1:
        return 'DSF', features

    # Fallback: classify by dominant feature
    if length_frac >= vsf_length_frac:
        return 'VSF', features
    if fiber_angle > 45:
        return 'TA', features
    return 'DSF', features


def analyze_cell_fibers(img_factin, cell_mask, nuc_center=None,
                        sigmas=None, threshold_sigma=0.5, min_fiber_px=5,
                        **classify_kwargs):
    """Run the full fiber analysis pipeline for a single cell.

    Pipeline: enhance → threshold → skeletonize → label → classify each component.

    Args:
        img_factin: 2D or 3D float array, F-actin fluorescence image.
        cell_mask: 2D bool array, True within cell boundary.
        nuc_center: (row, col) array or None, nucleus centroid.
        sigmas: iterable, Frangi filter scales (default range(1, 4)).
        threshold_sigma: float, fiber segmentation threshold (default 0.5).
        min_fiber_px: int, minimum skeleton length to classify (default 5).
        **classify_kwargs: passed to classify_fiber_subtype().

    Returns:
        dict with:
            subtypes:      list of str, one per classified fiber
            counts:        Counter of subtype → count
            dominant:      str, most common subtype (or 'unknown')
            n_fibers:      int, total fibers classified
            fiber_details: list of dicts, one per fiber with features + subtype
            skeleton:      2D bool array
            fiber_mask:    2D bool array
            enhanced:      2D float array (Frangi output)
    """
    img_factin = np.asarray(img_factin, dtype=float)
    if img_factin.ndim == 3:
        img_factin = img_factin[:, :, 0]

    # Get cell properties for orientation reference
    cell_labeled = sk_label(cell_mask.astype(np.uint8))
    cell_props = regionprops(cell_labeled)
    if not cell_props:
        return {'subtypes': [], 'counts': Counter(), 'dominant': 'unknown',
                'n_fibers': 0, 'fiber_details': [], 'skeleton': np.zeros_like(cell_mask),
                'fiber_mask': np.zeros_like(cell_mask), 'enhanced': np.zeros_like(img_factin)}

    cell_prop = cell_props[0]
    cell_length_px = cell_prop.axis_major_length

    # Step 1: Enhance fibers
    enhanced = enhance_fibers(img_factin, sigmas=sigmas, mask=cell_mask)

    # Step 2: Segment
    fiber_mask, skeleton = segment_fibers(enhanced, threshold_sigma=threshold_sigma,
                                          mask=cell_mask, min_fiber_length=min_fiber_px)

    # Step 3: Label individual fiber components
    skel_labeled, n_components = sk_label(skeleton, return_num=True)

    subtypes = []
    fiber_details = []
    for fib_id in range(1, n_components + 1):
        comp = skel_labeled == fib_id
        if comp.sum() < min_fiber_px:
            continue

        subtype, feats = classify_fiber_subtype(
            comp, cell_prop, nuc_center, img_factin, cell_mask, cell_length_px,
            **classify_kwargs
        )
        if subtype != 'unknown':
            subtypes.append(subtype)
            fiber_details.append({'subtype': subtype, **feats})

    counts = Counter(subtypes)
    dominant = counts.most_common(1)[0][0] if counts else 'unknown'

    return {
        'subtypes': subtypes,
        'counts': dict(counts),
        'dominant': dominant,
        'n_fibers': len(subtypes),
        'fiber_details': fiber_details,
        'skeleton': skeleton,
        'fiber_mask': fiber_mask,
        'enhanced': enhanced,
    }


def fiber_differential(cell_results):
    """Aggregate fiber subtype counts across multiple cells.

    Args:
        cell_results: list of dicts from analyze_cell_fibers(), each with
            a 'subtypes' key (list of str).

    Returns:
        dict with:
            VSF_count, DSF_count, PAC_count, TA_count: int
            total_fibers: int
            VSF_fraction, DSF_fraction, PAC_fraction, TA_fraction: float (0-1)
            dominant_type: str
            per_cell: list of {'dominant': str, 'VSF': int, 'DSF': int, 'PAC': int, 'TA': int}
    """
    all_subtypes = []
    per_cell = []
    for cr in cell_results:
        subtypes = cr.get('subtypes', [])
        all_subtypes.extend(subtypes)
        c = Counter(subtypes)
        per_cell.append({
            'dominant': cr.get('dominant', 'unknown'),
            'VSF': c.get('VSF', 0), 'DSF': c.get('DSF', 0),
            'PAC': c.get('PAC', 0), 'TA': c.get('TA', 0),
            'total': len(subtypes),
        })

    total_counts = Counter(all_subtypes)
    n_total = len(all_subtypes)

    return {
        'VSF_count': total_counts.get('VSF', 0),
        'DSF_count': total_counts.get('DSF', 0),
        'PAC_count': total_counts.get('PAC', 0),
        'TA_count': total_counts.get('TA', 0),
        'total_fibers': n_total,
        'VSF_fraction': float(total_counts.get('VSF', 0)) / max(1, n_total),
        'DSF_fraction': float(total_counts.get('DSF', 0)) / max(1, n_total),
        'PAC_fraction': float(total_counts.get('PAC', 0)) / max(1, n_total),
        'TA_fraction': float(total_counts.get('TA', 0)) / max(1, n_total),
        'dominant_type': total_counts.most_common(1)[0][0] if total_counts else 'unknown',
        'per_cell': per_cell,
    }
