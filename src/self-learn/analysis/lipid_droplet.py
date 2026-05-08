"""Lipid droplet analysis — detection, morphology, and metabolic scoring.

Quantifies intracellular lipid droplets from fluorescence images (LipidTOX,
BODIPY, Oil Red O, Nile Red). Lipid droplets are round, bright cytoplasmic
organelles ranging from 0.1 to 10 µm in diameter.

Common applications:
  - Adipogenesis differentiation (3T3-L1 cells)
  - Hepatic steatosis (fatty liver disease models)
  - Drug effects on lipid metabolism
  - Neutral lipid storage in non-adipocytes

Functions:
    detect_lipid_droplets    -- LoG-based detection of round bright organelles
    droplet_morphology       -- Per-droplet size, roundness, intensity
    droplet_count_per_cell   -- Assign droplets to cells and count per cell
    lipid_content_score      -- Area fraction as proxy for lipid loading
    steatosis_index          -- Clinical-style steatosis scoring (0-3)
    adipogenesis_score       -- Differentiation index (control vs treated)
    lipid_droplet_analysis   -- Full per-cell pipeline
    population_lipid         -- Aggregate across cell population
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label as ndlabel
from skimage.filters import gaussian as sk_gaussian
from skimage.feature import blob_log
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects, disk, binary_erosion


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_lipid_droplets(img, cell_mask=None, min_sigma=1.0, max_sigma=8.0,
                          n_scales=6, threshold=0.1, min_area=5, max_area=None,
                          pixel_size_um=None,
                          min_diameter_um=None, max_diameter_um=None,
                          min_area_um2=None):
    """Detect lipid droplets as round bright spots using LoG blob detection.

    Uses skimage blob_log for scale-space detection. Droplets are typically
    0.2-5 µm in diameter, bodipy steatotic droplets up to ~5-10 µm.

    Physical defaults (when ``pixel_size_um`` is provided):
    ``min_diameter_um=0.5``, ``max_diameter_um=5.0``, ``min_area_um2=0.3``.
    Internally converts to pixel sigma via ``sigma ≈ diameter / (2√2 · px)``.

    The raw pixel defaults (``min_sigma=1.0, max_sigma=8.0, min_area=5``)
    only make sense at ~0.25 µm/px (40x on 512 sensor); at 10x/1 µm/px
    they miss small droplets, and at 100x they over-detect.

    Args:
        img: 2D float array, lipid droplet fluorescence channel.
        cell_mask: 2D bool array, restrict detection to cell interior.
            If None, searches entire image.
        min_sigma: float, minimum blob scale in pixels. Ignored if
            ``pixel_size_um`` is given.
        max_sigma: float, maximum blob scale in pixels. Ignored if
            ``pixel_size_um`` is given.
        n_scales: int, number of sigma scales in LoG pyramid.
        threshold: float, LoG response threshold. Lower = more sensitive.
        min_area: int, minimum droplet area in pixels. Ignored if
            ``pixel_size_um`` is given.
        max_area: int or None, maximum droplet area. If None, uses 10×min_area².
        pixel_size_um: Optional float from ``get_config(core).pixel_size_um``.
            **Recommended** — physical-unit defaults adapt to magnification.
        min_diameter_um: Optional float, overrides physical default (0.5 µm).
        max_diameter_um: Optional float, overrides physical default (5.0 µm).
        min_area_um2: Optional float, overrides physical default (0.3 µm²).

    Returns:
        dict with:
            positions:  (N, 2) array of (row, col) centroids
            sizes:      (N,) array of estimated radii in pixels
            intensities: (N,) array of mean intensities per droplet
            n_droplets: int, total detected
            droplet_mask: 2D bool array, True at droplet pixels
            props:      list of RegionProps (for detailed analysis)
    """
    if pixel_size_um is not None:
        min_d_um = 0.5 if min_diameter_um is None else float(min_diameter_um)
        max_d_um = 5.0 if max_diameter_um is None else float(max_diameter_um)
        min_a_um2 = 0.3 if min_area_um2 is None else float(min_area_um2)
        # LoG sigma convention: a blob of diameter d has peak response at
        # sigma ≈ d / (2 * sqrt(2))
        ps = float(pixel_size_um)
        min_sigma = max(0.5, (min_d_um / ps) / (2.0 * np.sqrt(2.0)))
        max_sigma = max(min_sigma + 0.5, (max_d_um / ps) / (2.0 * np.sqrt(2.0)))
        min_area = max(1, int(round(min_a_um2 / (ps * ps))))
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img[:, :, 0]

    # Normalize to [0, 1]
    mn, mx = img.min(), img.max()
    if mx == mn:
        empty = np.zeros(img.shape, dtype=bool)
        return {'positions': np.zeros((0, 2)), 'sizes': np.array([]),
                'intensities': np.array([]), 'n_droplets': 0,
                'droplet_mask': empty, 'props': []}

    img_norm = (img - mn) / (mx - mn)

    if cell_mask is not None:
        img_search = img_norm * cell_mask.astype(float)
    else:
        img_search = img_norm

    # LoG blob detection
    try:
        blobs = blob_log(img_search, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=n_scales, threshold=threshold)
    except Exception:
        blobs = np.zeros((0, 3))

    if len(blobs) == 0:
        empty = np.zeros(img.shape, dtype=bool)
        return {'positions': np.zeros((0, 2)), 'sizes': np.array([]),
                'intensities': np.array([]), 'n_droplets': 0,
                'droplet_mask': empty, 'props': []}

    # Build droplet mask from detected blobs
    droplet_mask = np.zeros(img.shape, dtype=bool)
    valid_positions = []
    valid_sizes = []
    valid_intensities = []

    if max_area is None:
        max_area = max(min_area * 100, 500)

    for row, col, sigma in blobs:
        r = int(round(row))
        c = int(round(col))
        radius = int(np.ceil(sigma * np.sqrt(2)))  # LoG blob radius

        # Draw circular region
        rr, cc = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = rr**2 + cc**2 <= radius**2

        r0 = max(0, r - radius)
        r1 = min(img.shape[0], r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(img.shape[1], c + radius + 1)

        cr0 = radius - (r - r0)
        cr1 = radius + (r1 - r)
        cc0 = radius - (c - c0)
        cc1 = radius + (c1 - c)

        region = circle[cr0:cr1, cc0:cc1]
        patch = droplet_mask[r0:r1, c0:c1]
        patch |= region

    # Label individual droplets from the mask
    if cell_mask is not None:
        droplet_mask &= cell_mask
    droplet_mask = remove_small_objects(droplet_mask, max_size=min_area)

    labeled, _ = sk_label(droplet_mask, return_num=True)
    props = regionprops(labeled, intensity_image=img)

    # Filter by area
    valid_props = [p for p in props if min_area <= p.area <= max_area]

    # Build output arrays
    positions = np.array([p.centroid for p in valid_props]) if valid_props else np.zeros((0, 2))
    sizes = np.array([np.sqrt(p.area / np.pi) for p in valid_props]) if valid_props else np.array([])
    intensities = np.array([p.intensity_mean for p in valid_props]) if valid_props else np.array([])

    # Rebuild mask with only valid droplets
    clean_mask = np.zeros(img.shape, dtype=bool)
    for p in valid_props:
        clean_mask[labeled == p.label] = True

    return {
        'positions': positions,
        'sizes': sizes,
        'intensities': intensities,
        'n_droplets': len(valid_props),
        'droplet_mask': clean_mask,
        'props': valid_props,
    }


def droplet_morphology(props):
    """Extract per-droplet morphological features.

    Args:
        props: list of RegionProps from detect_lipid_droplets()['props'].

    Returns:
        list of dicts, one per droplet with:
            area_px:       int, droplet area in pixels
            radius_px:     float, equivalent circle radius
            circularity:   float, 4π·area/perimeter² (1=perfect circle)
            eccentricity:  float, 0=round, 1=elongated
            mean_intensity: float, mean fluorescence
            max_intensity: float, max fluorescence (core brightness)
            solidity:      float, area/convex_hull_area
    """
    results = []
    for p in props:
        area = float(p.area)
        perim = float(p.perimeter) if p.perimeter > 0 else 1.0
        circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
        results.append({
            'area_px': int(area),
            'radius_px': float(np.sqrt(area / np.pi)),
            'circularity': float(circularity),
            'eccentricity': float(p.eccentricity),
            'mean_intensity': float(p.intensity_mean),
            'max_intensity': float(p.intensity_max),
            'solidity': float(p.solidity),
        })
    return results


# ── Per-cell counting ─────────────────────────────────────────────────────────

def droplet_count_per_cell(droplet_positions, cell_labels):
    """Assign lipid droplets to cells by position.

    Args:
        droplet_positions: (N, 2) array of (row, col) droplet centroids.
        cell_labels: 2D int array, labeled cell regions (background=0).

    Returns:
        dict mapping cell_id → list of droplet indices in that cell.
        Also includes 'unassigned' for droplets not inside any cell.
    """
    assignment = {}
    unassigned = []

    for i, (r, c) in enumerate(droplet_positions):
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < cell_labels.shape[0] and 0 <= ci < cell_labels.shape[1]:
            cell_id = int(cell_labels[ri, ci])
            if cell_id > 0:
                assignment.setdefault(cell_id, []).append(i)
            else:
                unassigned.append(i)
        else:
            unassigned.append(i)

    assignment['unassigned'] = unassigned
    return assignment


# ── Metabolic scoring ─────────────────────────────────────────────────────────

def lipid_content_score(droplet_mask, cell_mask):
    """Estimate lipid loading as fraction of cell area occupied by droplets.

    A pure indicator of neutral lipid accumulation. Adipocytes show 80-90%
    occupancy. Normal fibroblasts show < 5%.

    Args:
        droplet_mask: 2D bool array, True at droplet positions.
        cell_mask: 2D bool array, True within cell boundary.

    Returns:
        dict with:
            lipid_fraction: float [0-1], droplet area / cell area
            droplet_area_px: int, total droplet pixel count
            cell_area_px: int, total cell pixel count
    """
    cell_area = int(cell_mask.sum())
    droplet_area = int((droplet_mask & cell_mask).sum())
    fraction = float(droplet_area) / max(1, cell_area)
    return {
        'lipid_fraction': fraction,
        'droplet_area_px': droplet_area,
        'cell_area_px': cell_area,
    }


def steatosis_index(lipid_fraction, n_droplets, cell_area_px):
    """Score hepatic steatosis severity (0=none to 3=severe).

    Clinical-style grading based on lipid fraction and droplet density.
    Used for fatty liver disease models.

    Grades:
        0: lipid_fraction < 0.05 (normal)
        1: 0.05 ≤ fraction < 0.20 (mild steatosis)
        2: 0.20 ≤ fraction < 0.50 (moderate steatosis)
        3: fraction ≥ 0.50 (severe steatosis / pan-steatosis)

    Args:
        lipid_fraction: float [0-1] from lipid_content_score().
        n_droplets: int, number of droplets per cell.
        cell_area_px: int, cell area in pixels.

    Returns:
        dict with:
            grade: int [0-3], steatosis grade
            label: str, descriptive label
            droplet_density: float, droplets per 100 px² of cell area
    """
    if lipid_fraction < 0.05:
        grade, label = 0, 'normal'
    elif lipid_fraction < 0.20:
        grade, label = 1, 'mild_steatosis'
    elif lipid_fraction < 0.50:
        grade, label = 2, 'moderate_steatosis'
    else:
        grade, label = 3, 'severe_steatosis'

    density = float(n_droplets) / max(1, cell_area_px) * 100.0

    return {
        'grade': grade,
        'label': label,
        'droplet_density': density,
    }


def adipogenesis_score(n_droplets_treated, mean_size_treated,
                       n_droplets_control, mean_size_control):
    """Compute adipogenesis differentiation index.

    Compares treated (differentiated) vs control (undifferentiated) cells.
    Combines fold changes in droplet count and size.

    Args:
        n_droplets_treated: float, mean droplets per cell in treated group.
        mean_size_treated: float, mean droplet radius (px) in treated group.
        n_droplets_control: float, mean droplets per cell in control group.
        mean_size_control: float, mean droplet radius (px) in control group.

    Returns:
        dict with:
            count_fold_change: float, treated/control droplet count
            size_fold_change: float, treated/control mean size
            differentiation_index: float, geometric mean of both fold changes
            differentiated: bool, index > 2.0 suggests differentiation
    """
    eps = 1e-6
    count_fc = float(n_droplets_treated) / max(eps, float(n_droplets_control))
    size_fc = float(mean_size_treated) / max(eps, float(mean_size_control))
    diff_idx = float(np.sqrt(count_fc * size_fc))

    return {
        'count_fold_change': count_fc,
        'size_fold_change': size_fc,
        'differentiation_index': diff_idx,
        'differentiated': diff_idx > 2.0,
    }


# ── Full per-cell pipeline ────────────────────────────────────────────────────

def lipid_droplet_analysis(img, cell_mask, **detection_kwargs):
    """Full lipid droplet analysis pipeline for a single cell.

    Args:
        img: 2D float array, lipid dye fluorescence channel.
        cell_mask: 2D bool array, cell boundary mask.
        **detection_kwargs: passed to detect_lipid_droplets().

    Returns:
        dict with:
            n_droplets: int
            mean_size_px: float, mean droplet radius
            size_distribution: list of radii
            lipid_fraction: float
            steatosis_grade: int [0-3]
            steatosis_label: str
            per_droplet: list of morphology dicts
            droplet_mask: 2D bool array
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = img[:, :, 0]

    result = detect_lipid_droplets(img, cell_mask=cell_mask, **detection_kwargs)
    n = result['n_droplets']
    props = result['props']
    dmask = result['droplet_mask']

    morphology = droplet_morphology(props)
    sizes = [m['radius_px'] for m in morphology]
    mean_size = float(np.mean(sizes)) if sizes else 0.0

    lcs = lipid_content_score(dmask, cell_mask)
    cell_area = lcs['cell_area_px']

    si = steatosis_index(lcs['lipid_fraction'], n, cell_area)

    return {
        'n_droplets': n,
        'mean_size_px': mean_size,
        'size_distribution': sizes,
        'lipid_fraction': lcs['lipid_fraction'],
        'droplet_area_px': lcs['droplet_area_px'],
        'cell_area_px': cell_area,
        'steatosis_grade': si['grade'],
        'steatosis_label': si['label'],
        'per_droplet': morphology,
        'droplet_mask': dmask,
    }


# ── Population-level analysis ─────────────────────────────────────────────────

def population_lipid(cell_results):
    """Aggregate lipid droplet analysis across cell population.

    Args:
        cell_results: list of dicts from lipid_droplet_analysis().

    Returns:
        dict with:
            n_cells: int
            mean_droplets_per_cell: float
            std_droplets_per_cell: float
            mean_lipid_fraction: float
            mean_size_px: float
            steatosis_distribution: dict of grade → count
            dominant_grade: int [0-3]
    """
    if not cell_results:
        return {
            'n_cells': 0, 'mean_droplets_per_cell': 0.0,
            'std_droplets_per_cell': 0.0, 'mean_lipid_fraction': 0.0,
            'mean_size_px': 0.0, 'steatosis_distribution': {},
            'dominant_grade': 0,
        }

    counts = [r['n_droplets'] for r in cell_results]
    fractions = [r['lipid_fraction'] for r in cell_results]
    sizes = [r['mean_size_px'] for r in cell_results]
    grades = [r['steatosis_grade'] for r in cell_results]

    steatosis_dist = {}
    for g in grades:
        steatosis_dist[g] = steatosis_dist.get(g, 0) + 1

    dominant = max(steatosis_dist, key=steatosis_dist.get) if steatosis_dist else 0

    return {
        'n_cells': len(cell_results),
        'mean_droplets_per_cell': float(np.mean(counts)),
        'std_droplets_per_cell': float(np.std(counts)),
        'mean_lipid_fraction': float(np.mean(fractions)),
        'mean_size_px': float(np.mean(sizes)),
        'steatosis_distribution': steatosis_dist,
        'dominant_grade': dominant,
    }
