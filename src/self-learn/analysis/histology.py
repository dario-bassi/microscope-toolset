"""Histology analysis for H&E stained tissue sections.

Provides standardized pipelines for tumor grading, nuclear
morphometry, gland architecture assessment, and necrosis detection
from RGB brightfield H&E images.

Functions:
    deconvolve_hae       -- Separate hematoxylin, eosin, and DAB stains
    segment_nuclei_hae   -- Segment nuclei from hematoxylin channel
    detect_necrosis      -- Find necrotic regions (eosinophilic + acellular)
    assess_glands        -- Classify gland architecture
    measure_pleomorphism -- Nuclear size variation metrics
    detect_mitoses       -- Find mitotic figures from nuclear properties
    grade_tumor          -- Nottingham grading from component scores

Real microscope note:
    H&E color deconvolution assumes standard color balance. Real tissue
    staining varies significantly by lab, fixation protocol, stain batch,
    and scanner/microscope setup. Color-based thresholds are unreliable.
    For production tissue analysis:
    - Apply stain normalization (Macenko, Vahadane, or similar)
    - Validate thresholds on known reference slides
    - Consider machine learning for nuclear detection (U-Net, Mask R-CNN)
    - Document staining protocol and scanner settings for reproducibility
"""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def deconvolve_hae(image):
    """Separate H&E stains from an RGB image.

    Uses the standard Ruifrok & Johnston color deconvolution via
    skimage.color.rgb2hed.

    Args:
        image: (H, W, 3) RGB uint8 image.

    Returns:
        dict with:
            hematoxylin: 2D float array (nuclei, higher = more stain).
            eosin: 2D float array (cytoplasm/stroma).
            dab: 2D float array (DAB if immunohistochemistry).
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image (H, W, 3)")

    hed = rgb2hed(image)
    return {
        'hematoxylin': hed[:, :, 0],
        'eosin': hed[:, :, 1],
        'dab': hed[:, :, 2],
    }


def segment_nuclei_hae(image, min_area=30, max_area=8000,
                       min_distance=8):
    """Segment nuclei from an H&E image using hematoxylin channel.

    Performs color deconvolution, Otsu thresholding, and watershed
    splitting of touching nuclei.

    Args:
        image: (H, W, 3) RGB image.
        min_area: int, minimum nucleus area in pixels.
        max_area: int, maximum nucleus area in pixels.
        min_distance: int, minimum distance between watershed seeds.

    Returns:
        dict with:
            labeled: 2D int array, labeled nuclei.
            props: list of regionprops.
            n_nuclei: int.
            hematoxylin: 2D float, hematoxylin channel used.
            threshold: float, Otsu threshold applied.
    """
    stains = deconvolve_hae(image)
    h = stains['hematoxylin']

    h_thresh = filters.threshold_otsu(h)
    mask = h > h_thresh
    mask = morphology.remove_small_objects(mask, max_size=min_area - 1)
    mask = ndimage.binary_fill_holes(mask)

    # Watershed splitting
    dist = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(dist, min_distance=min_distance, labels=mask)
    if len(coords) == 0:
        labeled = measure.label(mask)
    else:
        markers = np.zeros_like(mask, dtype=int)
        for i, (r, c) in enumerate(coords, 1):
            markers[r, c] = i
        labeled = watershed(-dist, markers, mask=mask)

    props = measure.regionprops(labeled)
    # Filter by size
    valid_labels = set()
    for p in props:
        if min_area <= p.area <= max_area:
            valid_labels.add(p.label)

    # Re-label keeping only valid
    filtered = np.where(np.isin(labeled, list(valid_labels)), labeled, 0)
    if valid_labels:
        filtered_props = [p for p in props if p.label in valid_labels]
    else:
        filtered_props = []

    return {
        'labeled': filtered,
        'props': filtered_props,
        'n_nuclei': len(filtered_props),
        'hematoxylin': h,
        'threshold': float(h_thresh),
    }


def detect_necrosis(image, min_area=200):
    """Detect necrotic regions in H&E tissue.

    Necrosis appears as eosinophilic (pink) acellular regions
    with absent or fragmented nuclei. Different from gland lumens
    (which are pale/white with minimal eosin stain).

    Args:
        image: (H, W, 3) RGB image.
        min_area: int, minimum necrotic patch size in pixels.

    Returns:
        dict with:
            necrosis_present: bool.
            necrosis_mask: 2D bool array.
            n_patches: int.
            necrosis_fraction: float, fraction of tissue that is necrotic.
            patch_areas: list of int, areas of necrotic patches.
    """
    stains = deconvolve_hae(image)
    h = stains['hematoxylin']
    e = stains['eosin']

    # Nuclear mask
    h_thresh = filters.threshold_otsu(h)
    nuc_mask = h > h_thresh

    # Nuclear-free zones (dilate nuclei to mark their neighborhood)
    nuc_dilated = ndimage.binary_dilation(nuc_mask, iterations=5)
    nuclear_free = ~nuc_dilated

    # Tissue mask (not pure white background)
    gray = np.mean(image.astype(float), axis=2)
    tissue = gray < 245

    # Necrotic: nuclear-free tissue with above-median eosin stain
    # (lumens have very low eosin; necrosis has moderate eosin from ghost cells)
    e_in_tissue = e[tissue]
    if len(e_in_tissue) == 0:
        e_thresh = 0
    else:
        e_thresh = np.percentile(e_in_tissue[e_in_tissue > 0], 50) \
            if (e_in_tissue > 0).any() else 0

    necrotic = nuclear_free & tissue & (e > e_thresh)
    necrotic = morphology.remove_small_objects(necrotic, max_size=min_area - 1)

    # Label patches
    labeled = measure.label(necrotic)
    props = measure.regionprops(labeled)
    patch_areas = [p.area for p in props]

    tissue_area = tissue.sum()
    necrosis_frac = necrotic.sum() / tissue_area if tissue_area > 0 else 0

    return {
        'necrosis_present': len(props) > 0,
        'necrosis_mask': necrotic,
        'n_patches': len(props),
        'necrosis_fraction': round(float(necrosis_frac), 4),
        'patch_areas': patch_areas,
    }


def assess_glands(image, min_cluster_area=1000):
    """Assess gland architecture in H&E tissue.

    Classifies nuclear clusters as ring-like glands (with lumens)
    or solid nests (without lumens) based on solidity.

    Args:
        image: (H, W, 3) RGB image.
        min_cluster_area: int, minimum cluster area to consider.

    Returns:
        dict with:
            n_ring_glands: int, glands with lumens.
            n_solid_nests: int, solid clusters.
            total_structures: int.
            gland_ratio: float, fraction that are ring-like.
            pattern: str, "well-formed", "poorly-formed", or "absent".
    """
    seg = segment_nuclei_hae(image, max_area=50000)
    # Get large connected regions from nuclear mask
    h = seg['hematoxylin']
    h_thresh = seg['threshold']
    nuc_mask = h > h_thresh
    nuc_mask = morphology.remove_small_objects(nuc_mask, max_size=29)
    nuc_mask = ndimage.binary_fill_holes(nuc_mask)

    labeled = measure.label(nuc_mask)
    props = measure.regionprops(labeled)

    gland_walls = [p for p in props if p.area > min_cluster_area]
    ring_glands = [g for g in gland_walls if g.solidity < 0.75]
    solid_nests = [g for g in gland_walls if g.solidity >= 0.75]

    total = len(gland_walls)
    ratio = len(ring_glands) / total if total > 0 else 0

    if ratio > 0.75:
        pattern = 'well-formed'
    elif ratio > 0.10:
        pattern = 'poorly-formed'
    else:
        pattern = 'absent'

    return {
        'n_ring_glands': len(ring_glands),
        'n_solid_nests': len(solid_nests),
        'total_structures': total,
        'gland_ratio': round(ratio, 3),
        'pattern': pattern,
    }


def measure_pleomorphism(props):
    """Measure nuclear size variation (pleomorphism).

    Args:
        props: list of regionprops from segmented nuclei.

    Returns:
        dict with:
            cv: float, coefficient of variation of nuclear area.
            q75_q25_ratio: float, inter-quartile area ratio.
            mean_area: float.
            std_area: float.
            n_nuclei: int.
    """
    if not props:
        return {'cv': 0, 'q75_q25_ratio': 1, 'mean_area': 0,
                'std_area': 0, 'n_nuclei': 0}

    areas = np.array([p.area for p in props], dtype=float)
    mean_a = float(areas.mean())
    std_a = float(areas.std())
    cv = std_a / mean_a if mean_a > 0 else 0

    q25 = np.percentile(areas, 25)
    q75 = np.percentile(areas, 75)
    ratio = q75 / q25 if q25 > 0 else 1

    return {
        'cv': round(cv, 3),
        'q75_q25_ratio': round(ratio, 2),
        'mean_area': round(mean_a, 1),
        'std_area': round(std_a, 1),
        'n_nuclei': len(areas),
    }


def detect_mitoses(props, hematoxylin, labeled, max_area=800,
                   stain_percentile=65, min_eccentricity=0.75,
                   max_solidity=0.80):
    """Detect mitotic figures from nuclear properties.

    Mitotic nuclei have condensed chromosomes: small-medium size,
    intense hematoxylin staining, and irregular shape.

    Args:
        props: list of regionprops.
        hematoxylin: 2D float, hematoxylin channel.
        labeled: 2D int, labeled nuclei image.
        max_area: int, maximum area for mitotic candidate.
        stain_percentile: float, minimum stain intensity percentile.
        min_eccentricity: float, minimum eccentricity OR...
        max_solidity: float, ...maximum solidity for shape irregularity.

    Returns:
        dict with:
            mitotic_count: int.
            mitotic_index: float, fraction of nuclei that are mitotic.
            candidates: list of regionprops of mitotic nuclei.
    """
    if not props:
        return {'mitotic_count': 0, 'mitotic_index': 0, 'candidates': []}

    # Stain intensity threshold from nuclear regions
    nuc_mask = labeled > 0
    h_values = hematoxylin[nuc_mask]
    if len(h_values) == 0:
        return {'mitotic_count': 0, 'mitotic_index': 0, 'candidates': []}
    stain_thresh = np.percentile(h_values, stain_percentile)

    candidates = []
    for p in props:
        if p.area > max_area:
            continue
        region_mask = labeled == p.label
        mean_h = hematoxylin[region_mask].mean()
        if mean_h < stain_thresh:
            continue
        # Shape criterion: irregular or elongated
        if p.eccentricity > min_eccentricity or p.solidity < max_solidity:
            candidates.append(p)

    mitotic_index = len(candidates) / len(props) if props else 0

    return {
        'mitotic_count': len(candidates),
        'mitotic_index': round(mitotic_index, 4),
        'candidates': candidates,
    }


def grade_tumor(gland_pattern, q75_q25_ratio, mitotic_index):
    """Compute Nottingham tumor grade from component scores.

    Args:
        gland_pattern: str, "well-formed", "poorly-formed", or "absent".
        q75_q25_ratio: float, Q75/Q25 nuclear area ratio.
        mitotic_index: float, fraction of mitotic nuclei.

    Returns:
        dict with:
            grade: int, 1-3.
            total_score: int, 3-9.
            tubule_score: int, 1-3.
            pleomorphism_score: int, 1-3.
            mitotic_score: int, 1-3.
    """
    # Tubule formation score
    if gland_pattern == 'well-formed':
        tubule = 1
    elif gland_pattern == 'poorly-formed':
        tubule = 2
    else:
        tubule = 3

    # Pleomorphism score
    if q75_q25_ratio < 2:
        pleo = 1
    elif q75_q25_ratio < 4:
        pleo = 2
    else:
        pleo = 3

    # Mitotic score
    if mitotic_index < 0.03:
        mitotic = 1
    elif mitotic_index < 0.10:
        mitotic = 2
    else:
        mitotic = 3

    total = tubule + pleo + mitotic
    if total <= 5:
        grade = 1
    elif total <= 7:
        grade = 2
    else:
        grade = 3

    return {
        'grade': grade,
        'total_score': total,
        'tubule_score': tubule,
        'pleomorphism_score': pleo,
        'mitotic_score': mitotic,
    }
