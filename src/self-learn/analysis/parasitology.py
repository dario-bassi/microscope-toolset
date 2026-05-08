"""Blood smear parasitology: RBC counting, infection detection, stage classification.

Provides reusable analysis for Giemsa-stained blood smears, supporting:
- Total RBC quantification using distance-transform peak detection
- Infected cell identification from membrane/stain channels
- P. falciparum stage classification (ring, trophozoite, schizont, gametocyte)
- Parasitemia calculation with per-FOV breakdown

Typical workflow:
    rbcs = count_rbcs(bf_gray)
    infected = detect_infected(membrane_img)
    stages = classify_stages(infected, nucleus_img)
    result = compute_parasitemia(rbcs, stages)
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max


def count_rbcs(bf_gray, threshold=220, min_distance=20, min_cell_area=50):
    """Count total RBCs in a brightfield grayscale image.

    Uses distance-transform peak detection to count cells even when they
    touch or overlap. Works on Giemsa, Wright, or unstained blood smears.

    Args:
        bf_gray: 2D float/uint8 array — grayscale BF image.
            Background should be bright (>200), cells darker.
        threshold: Intensity below which pixels are considered cell.
        min_distance: Minimum separation between detected peaks (pixels).
            Roughly half the expected cell diameter.
        min_cell_area: Minimum connected component area (noise filter).

    Returns:
        dict with:
            count: int — total RBC count.
            peak_coords: ndarray (N, 2) — (row, col) of each detected peak.
            cell_mask: 2D bool — thresholded + cleaned cell mask.
    """
    bf_gray = np.asarray(bf_gray, dtype=np.float64)

    mask = bf_gray < threshold
    mask = morphology.remove_small_objects(mask, max_size=min_cell_area)
    mask = ndimage.binary_fill_holes(mask)

    dt = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(dt, min_distance=min_distance, labels=mask,
                            exclude_border=False)

    return {
        'count': len(coords),
        'peak_coords': coords,
        'cell_mask': mask,
    }


def detect_infected(membrane_img, threshold=20, min_area=200, max_area=20000):
    """Detect infected RBCs from a membrane/outline channel.

    The membrane channel highlights infected RBC outlines as bright circles.
    This function fills and labels them to identify individual infected cells.

    Args:
        membrane_img: 2D uint8 array — membrane fluorescence channel.
        threshold: Minimum brightness for membrane signal.
        min_area: Minimum filled region area (pixels) to keep.
        max_area: Maximum area (filters out merged/debris regions).

    Returns:
        list of dicts, each with:
            label: int — region label.
            centroid: (row, col) — centroid of the infected RBC.
            area: float — filled area in pixels.
            eccentricity: float — shape eccentricity (0=circle, 1=line).
            bbox: (r0, c0, r1, c1) — bounding box.
            coords: ndarray (N, 2) — pixel coordinates of the region.
    """
    mem = np.asarray(membrane_img, dtype=np.float64)
    mask = mem > threshold
    filled = ndimage.binary_fill_holes(mask)
    labeled = measure.label(filled)
    props = measure.regionprops(labeled)

    results = []
    for p in props:
        if p.area < min_area or p.area > max_area:
            continue
        results.append({
            'label': p.label,
            'centroid': (float(p.centroid[0]), float(p.centroid[1])),
            'area': float(p.area),
            'eccentricity': float(p.eccentricity),
            'bbox': p.bbox,
            'coords': p.coords,
        })
    return results


def classify_stage(nucleus_roi, rbc_mask, rbc_area, rbc_eccentricity,
                   nuc_threshold=30, min_dot_area=3,
                   schizont_min_dots=5, troph_min_mass=300,
                   gamet_min_ecc=0.78, gamet_area_range=(500, 3500)):
    """Classify a single P. falciparum parasite by lifecycle stage.

    Uses nucleus-channel chromatin patterns within the infected RBC:
    - Ring: 1-2 small chromatin dots (most common in peripheral blood)
    - Trophozoite: 1 large dense chromatin mass
    - Schizont: 5+ merozoite nuclei in tight rosette pattern
    - Gametocyte: elongated (banana/crescent) RBC shape

    IMPORTANT: P. falciparum can have multiple ring parasites per RBC
    (~10% multi-infection rate). Multiple rings with 2 dots each can
    look like a schizont (6 scattered dots). The classifier distinguishes
    these by checking spatial arrangement: schizont merozoites form a
    tight, regular rosette; multi-ring dots are scattered.

    Args:
        nucleus_roi: 2D array — nucleus channel cropped to RBC bounding box.
        rbc_mask: 2D bool — mask of the RBC within the same crop.
        rbc_area: float — total RBC area in pixels.
        rbc_eccentricity: float — eccentricity of the RBC shape.
        nuc_threshold: Brightness threshold for chromatin detection.
        min_dot_area: Minimum chromatin dot size (noise filter).
        schizont_min_dots: Minimum separate dots for schizont classification.
        troph_min_mass: Minimum single-mass area for trophozoite.
        gamet_min_ecc: Minimum RBC eccentricity for gametocyte.
        gamet_area_range: (min, max) RBC area for gametocyte candidates.

    Returns:
        str — one of 'ring', 'trophozoite', 'schizont', 'gametocyte'.
    """
    nuc_inside = (nucleus_roi > nuc_threshold) & rbc_mask
    nuc_labeled = measure.label(nuc_inside)
    nuc_dots = [p for p in measure.regionprops(nuc_labeled)
                if p.area >= min_dot_area]
    n_dots = len(nuc_dots)
    largest_dot = max((d.area for d in nuc_dots), default=0)
    nuc_area = int(np.sum(nuc_inside))
    nuc_frac = nuc_area / rbc_area if rbc_area > 0 else 0

    if n_dots < 1:
        return 'ring'

    # 1. Gametocyte: banana/crescent shape distorts RBC → high eccentricity
    ga_lo, ga_hi = gamet_area_range
    if (rbc_eccentricity > gamet_min_ecc and
            ga_lo < rbc_area < ga_hi and n_dots <= 3):
        return 'gametocyte'

    # 2. Trophozoite: one large dense chromatin mass
    if largest_dot >= troph_min_mass and n_dots <= 2 and nuc_frac > 0.12:
        return 'trophozoite'

    # 3. Schizont: many dots in a TIGHT rosette (not scattered multi-ring)
    if n_dots >= schizont_min_dots:
        centroids = np.array([d.centroid for d in nuc_dots])
        dot_center = centroids.mean(axis=0)
        dists = np.sqrt(np.sum((centroids - dot_center)**2, axis=1))
        mean_dist = float(dists.mean())
        rbc_radius = np.sqrt(rbc_area / np.pi)
        spread_ratio = mean_dist / rbc_radius if rbc_radius > 0 else 1.0

        # Check dot size uniformity (schizont merozoites are similar size)
        dot_areas = [d.area for d in nuc_dots]
        area_cv = (float(np.std(dot_areas)) / float(np.mean(dot_areas))
                   if np.mean(dot_areas) > 0 else 0)

        # Tight rosette: spread_ratio < 0.45 and uniform sizes
        if spread_ratio < 0.45 and area_cv < 1.0:
            return 'schizont'
        # Scattered dots = multi-ring infection → classify as ring
        return 'ring'

    # 4. Ring: default (most common stage in P. falciparum peripheral blood)
    return 'ring'


def classify_stages(infected_cells, nucleus_img):
    """Classify all infected cells by parasite stage.

    Args:
        infected_cells: list of dicts from detect_infected().
        nucleus_img: 2D uint8 array — full-field nucleus channel.

    Returns:
        list of dicts, each with:
            stage: str — classified stage.
            centroid: (row, col).
            area: float — RBC area.
            eccentricity: float.
    """
    H, W = nucleus_img.shape[:2]
    results = []

    for cell in infected_cells:
        r0, c0, r1, c1 = cell['bbox']
        coords = cell['coords']

        rbc_mask_full = np.zeros((H, W), dtype=bool)
        rbc_mask_full[coords[:, 0], coords[:, 1]] = True
        rbc_mask_crop = rbc_mask_full[r0:r1, c0:c1]
        nuc_crop = nucleus_img[r0:r1, c0:c1]

        stage = classify_stage(
            nuc_crop, rbc_mask_crop,
            cell['area'], cell['eccentricity'],
        )
        results.append({
            'stage': stage,
            'centroid': cell['centroid'],
            'area': cell['area'],
            'eccentricity': cell['eccentricity'],
        })

    return results


def compute_parasitemia(total_rbc_count, classified_parasites):
    """Compute parasitemia and stage distribution.

    Args:
        total_rbc_count: int — total RBCs examined.
        classified_parasites: list of dicts from classify_stages().

    Returns:
        dict with:
            parasitemia: float — fraction of infected RBCs (0.0-1.0).
            stage_counts: dict — {stage: count}.
            dominant_stage: str — most common stage.
            total_parasites: int.
            total_rbc: int.
            stage_fractions: dict — {stage: fraction of parasites}.
    """
    stage_counts = {'ring': 0, 'trophozoite': 0, 'schizont': 0, 'gametocyte': 0}
    for p in classified_parasites:
        stage = p['stage']
        if stage in stage_counts:
            stage_counts[stage] += 1

    total_parasites = sum(stage_counts.values())
    parasitemia = total_parasites / total_rbc_count if total_rbc_count > 0 else 0.0

    dominant = max(stage_counts, key=stage_counts.get) if total_parasites > 0 else 'ring'

    stage_fracs = {}
    for s, c in stage_counts.items():
        stage_fracs[s] = c / total_parasites if total_parasites > 0 else 0.0

    return {
        'parasitemia': float(parasitemia),
        'stage_counts': stage_counts,
        'dominant_stage': dominant,
        'total_parasites': total_parasites,
        'total_rbc': total_rbc_count,
        'stage_fractions': stage_fracs,
    }


def analyze_blood_smear(bf_gray, nucleus_img, membrane_img,
                        bf_threshold=220, mem_threshold=20):
    """Full blood smear analysis pipeline: count → detect → classify → report.

    Convenience function combining all steps for a single FOV.

    Args:
        bf_gray: 2D array — BF grayscale image.
        nucleus_img: 2D array — nucleus/chromatin channel.
        membrane_img: 2D array — membrane/infected-outline channel.
        bf_threshold: Intensity threshold for RBC detection in BF.
        mem_threshold: Intensity threshold for membrane channel.

    Returns:
        dict with:
            total_rbc: int.
            n_infected: int.
            parasitemia: float.
            stage_counts: dict.
            dominant_stage: str.
            classified: list of per-parasite dicts.
    """
    rbc_result = count_rbcs(bf_gray, threshold=bf_threshold)
    infected = detect_infected(membrane_img, threshold=mem_threshold)
    classified = classify_stages(infected, nucleus_img)
    result = compute_parasitemia(rbc_result['count'], classified)
    result['classified'] = classified
    result['n_infected'] = len(infected)
    return result
