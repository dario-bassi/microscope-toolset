"""Histology analysis for H&E and Giemsa stained tissue/blood images.

Provides standardized pipelines for tumor grading, nuclear
morphometry, gland architecture assessment, and necrosis detection
from RGB brightfield H&E images. Also includes Giemsa stain separation
for blood smear analysis.

Functions:
    deconvolve_hae       -- Separate hematoxylin, eosin, and DAB stains
    deconvolve_giemsa    -- Separate Giemsa stain components (azur/methylene blue, eosin)
    segment_nuclei_hae   -- Segment nuclei from hematoxylin channel
    detect_necrosis      -- Find necrotic regions (eosinophilic + acellular)
    assess_glands        -- Classify gland architecture
    measure_pleomorphism -- Nuclear size variation metrics
    detect_mitoses       -- Find mitotic figures from nuclear properties
    grade_tumor          -- Nottingham grading from component scores
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
                       min_distance=8, pixel_size_um=None,
                       min_area_um2=None, max_area_um2=None,
                       min_distance_um=None):
    """Segment nuclei from an H&E image using hematoxylin channel.

    Performs color deconvolution, Otsu thresholding, and watershed
    splitting of touching nuclei.

    Physical defaults (when ``pixel_size_um`` is provided) mirror normal
    tissue: ``min_area_um2=7.5`` (≈3 µm-diameter nucleus), ``max_area_um2=500``
    (≈25 µm-diameter nucleus), ``min_distance_um=2.0``. Pass ``*_um`` /
    ``*_um2`` arguments to override.

    Raw pixel defaults (``min_area=30`` etc.) only make sense at roughly
    1 µm/px (10x on a 512 sensor) — at higher mags they're too generous.

    Args:
        image: (H, W, 3) RGB image.
        min_area: int, minimum nucleus area in pixels. Ignored if
            ``pixel_size_um`` is given (physical-unit defaults apply).
        max_area: int, maximum nucleus area in pixels. Ignored if
            ``pixel_size_um`` is given.
        min_distance: int, minimum distance between watershed seeds
            (pixels). Ignored if ``pixel_size_um`` is given.
        pixel_size_um: Optional float (µm/px from ``get_config(core)``).
            When provided, physical-unit defaults apply and ``areas_um2``
            is included in the output. **Highly recommended** — GT specs
            for area/count challenges are in world units, not camera px.
        min_area_um2: Optional float, overrides physical default (7.5 µm²).
        max_area_um2: Optional float, overrides physical default (500 µm²).
        min_distance_um: Optional float, overrides physical default (2.0 µm).

    Returns:
        dict with:
            labeled: 2D int array, labeled nuclei.
            props: list of regionprops.
            n_nuclei: int.
            hematoxylin: 2D float, hematoxylin channel used.
            threshold: float, Otsu threshold applied.
            areas_px: list of int, per-nucleus area in camera pixels.
            areas_um2: list of float, per-nucleus area in µm² (only if
                ``pixel_size_um`` was provided).
    """
    if pixel_size_um is not None:
        min_a_um2 = 7.5 if min_area_um2 is None else float(min_area_um2)
        max_a_um2 = 500.0 if max_area_um2 is None else float(max_area_um2)
        min_d_um = 2.0 if min_distance_um is None else float(min_distance_um)
        px2 = float(pixel_size_um) ** 2
        min_area = max(1, int(round(min_a_um2 / px2)))
        max_area = max(min_area + 1, int(round(max_a_um2 / px2)))
        min_distance = max(1, int(round(min_d_um / float(pixel_size_um))))
    stains = deconvolve_hae(image)
    h = stains['hematoxylin']

    h_thresh = filters.threshold_otsu(h)
    mask = h > h_thresh
    mask = morphology.remove_small_objects(mask, max_size=min_area)
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

    areas_px = [int(p.area) for p in filtered_props]
    result = {
        'labeled': filtered,
        'props': filtered_props,
        'n_nuclei': len(filtered_props),
        'hematoxylin': h,
        'threshold': float(h_thresh),
        'areas_px': areas_px,
    }
    if pixel_size_um is not None:
        scale = float(pixel_size_um) ** 2
        result['areas_um2'] = [a * scale for a in areas_px]
    return result


def detect_necrosis(image, min_area=200, pixel_size_um=None, min_area_um2=None,
                    nuclear_dilation_iter=5, bg_threshold=245):
    """Detect necrotic regions in H&E tissue.

    Necrosis appears as eosinophilic (pink) acellular regions
    with absent or fragmented nuclei. Different from gland lumens
    (which are pale/white with minimal eosin stain).

    Args:
        image: (H, W, 3) RGB image.
        min_area: int, minimum necrotic patch size in pixels.
            Ignored if `min_area_um2` + `pixel_size_um` given.
        pixel_size_um: float, pixel size in µm. If provided with
            `min_area_um2`, the size filter is scaled to the
            acquisition magnification.
        min_area_um2: float, minimum necrotic patch size in µm². A
            clinically meaningful necrosis patch is ≥ ~300 µm².

    Returns:
        dict with:
            necrosis_present: bool.
            necrosis_mask: 2D bool array.
            n_patches: int.
            necrosis_fraction: float, fraction of tissue that is necrotic.
            patch_areas: list of int, areas of necrotic patches.
    """
    # Honour physical-size filter if given
    if pixel_size_um is not None and min_area_um2 is not None:
        min_area = int(round(min_area_um2 / (pixel_size_um ** 2)))

    stains = deconvolve_hae(image)
    h = stains['hematoxylin']
    e = stains['eosin']

    # Nuclear mask
    h_thresh = filters.threshold_otsu(h)
    nuc_mask = h > h_thresh

    # Nuclear-free zones (dilate nuclei to mark their neighborhood)
    nuc_dilated = ndimage.binary_dilation(nuc_mask, iterations=int(nuclear_dilation_iter))
    nuclear_free = ~nuc_dilated

    # Tissue mask (not pure white background)
    gray = np.mean(image.astype(float), axis=2)
    tissue = gray < float(bg_threshold)

    # Necrotic: nuclear-free tissue with above-median eosin stain
    # (lumens have very low eosin; necrosis has moderate eosin from ghost cells)
    e_in_tissue = e[tissue]
    if len(e_in_tissue) == 0:
        e_thresh = 0
    else:
        e_thresh = np.percentile(e_in_tissue[e_in_tissue > 0], 50) \
            if (e_in_tissue > 0).any() else 0

    necrotic = nuclear_free & tissue & (e > e_thresh)
    necrotic = morphology.remove_small_objects(necrotic, max_size=min_area)

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


def detect_vessels(image, rbc_min_cluster_px=50, merge_distance_px=20,
                   min_rbc_saturation=50, require_nuclear_ring=False,
                   ring_min_hema_fraction=0.05,
                   ring_dilation_iter=3,
                   rbc_r_min=200, rbc_g_max=160, rbc_b_max=170):
    """Detect blood vessels via erythrocyte (RBC) clusters in H&E tissue.

    RBCs in H&E appear as bright, saturated red specks (R >> B); a blood
    vessel is a spatial cluster of RBCs bounded by a nuclear ring (endothelium).
    Detection strategy:
      1. Threshold for saturated red pixels (RBCs).
      2. Morphological closing to connect adjacent RBCs.
      3. Dilate each RBC cluster to merge nearby RBCs of the same vessel.
      4. Count distinct merged components containing >= rbc_min_cluster_px
         of actual RBC pixel area.

    Args:
        image: (H, W, 3) RGB H&E image.
        rbc_min_cluster_px: Minimum RBC pixel area per vessel (filters noise).
            At 1 um/px (10x), 50 ~= a 3-4 RBC cluster.
        merge_distance_px: Radius for merging nearby RBC clusters into one
            vessel. Dilation iterations.
        min_rbc_saturation: Minimum (R - G) to count as saturated red.
        require_nuclear_ring: If True, only keep RBC clusters that have at
            least ``ring_min_hema_fraction`` of pixels in their dilation-ring
            stained by hematoxylin (endothelial nuclei). Filters out
            extravasated RBCs sitting on collagen without vessel walls.
        ring_min_hema_fraction: Minimum fraction of ring pixels above the
            Otsu hematoxylin threshold when ``require_nuclear_ring=True``.
        ring_dilation_iter: Iterations for the dilation that defines the
            endothelial ring around an RBC cluster. Default 3 (~3 px at
            10x); raise on coarser samples or larger vessels.
        rbc_r_min, rbc_g_max, rbc_b_max: RGB cutoffs for the saturated-red
            mask. Tuned for standard H&E staining. Recipes that target a
            different stain protocol (Wright, Masson, IHC counterstains)
            should override.

    Returns:
        dict with:
            n_vessels: int.
            centroids: list of (cx, cy) pixel coordinates.
            rbc_areas: list of int, RBC pixel counts per vessel.
            vessel_labels: 2D int, labeled merged-cluster image.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image (H, W, 3)")

    R = image[..., 0].astype(float)
    G = image[..., 1].astype(float)
    B = image[..., 2].astype(float)

    # Saturated red: brighter than average, more red than blue/green
    rbc = ((R > rbc_r_min) & (G < rbc_g_max) & (B < rbc_b_max)
           & ((R - G) > min_rbc_saturation))
    rbc = morphology.remove_small_objects(rbc, max_size=3)
    rbc = ndimage.binary_closing(rbc, iterations=2)

    if not rbc.any():
        return {
            "n_vessels": 0,
            "centroids": [],
            "rbc_areas": [],
            "vessel_labels": np.zeros(image.shape[:2], dtype=np.int32),
        }

    # Merge nearby RBC clusters into a single vessel
    merged = ndimage.binary_dilation(rbc, iterations=merge_distance_px)
    labeled = measure.label(merged)

    # Precompute hematoxylin mask for nuclear-ring check
    hema_mask = None
    if require_nuclear_ring:
        hed = rgb2hed(image)
        h = hed[:, :, 0]
        h_thresh = filters.threshold_otsu(h)
        hema_mask = h > h_thresh

    vessels = []
    for region in measure.regionprops(labeled):
        region_mask = labeled == region.label
        rbc_area = int(rbc[region_mask].sum())
        if rbc_area < rbc_min_cluster_px:
            continue

        if require_nuclear_ring:
            # Ring = dilation of merged-cluster minus the cluster itself
            ring = ndimage.binary_dilation(
                region_mask, iterations=int(ring_dilation_iter)
            ) & ~region_mask
            n_ring = int(ring.sum())
            if n_ring == 0:
                continue
            hema_frac = float(hema_mask[ring].sum()) / n_ring
            if hema_frac < ring_min_hema_fraction:
                continue

        cy, cx = region.centroid
        vessels.append((float(cx), float(cy), rbc_area, region.label))

    # Sort by RBC area (largest first) for stable ordering
    vessels.sort(key=lambda v: v[2], reverse=True)
    kept_labels = {v[3] for v in vessels}
    vessel_labels = np.where(np.isin(labeled, list(kept_labels)), labeled, 0)

    return {
        "n_vessels": len(vessels),
        "centroids": [(v[0], v[1]) for v in vessels],
        "rbc_areas": [v[2] for v in vessels],
        "vessel_labels": vessel_labels.astype(np.int32),
    }


def assess_glands(image, min_cluster_area=1000,
                  pixel_size_um=None, min_cluster_area_um2=None,
                  nuc_min_area_px=29, nuc_min_area_um2=None,
                  nuc_max_area=50000, nuc_max_area_um2=None,
                  solidity_threshold=0.75,
                  well_formed_ratio=0.75,
                  poorly_formed_ratio=0.10):
    """Assess gland architecture in H&E tissue.

    Classifies nuclear clusters as ring-like glands (with lumens)
    or solid nests (without lumens) based on solidity.

    Args:
        image: (H, W, 3) RGB image.
        min_cluster_area: int, minimum cluster area to consider.
            Ignored if `min_cluster_area_um2` + `pixel_size_um` given.
        pixel_size_um: float, pixel size in µm. If provided with the
            ``*_um2`` kwargs below, scales the px filters from physical
            units. Recipes that move between magnifications should pass
            the µm² values and the current px size.
        min_cluster_area_um2: float, typical gland ≥ ~1500 µm².
        nuc_min_area_px: int, raw-pixel small-objects floor on the nuclear
            mask before the cluster filter (default 29 ~= ~7 µm² at 0.5
            µm/px). Ignored if ``nuc_min_area_um2`` + ``pixel_size_um``
            given.
        nuc_min_area_um2: float, alternate µm² escape hatch for the same.
        nuc_max_area, nuc_max_area_um2: same px / µm² pair for the
            ``segment_nuclei_hae(max_area=)`` ceiling — drops debris
            larger than realistic nuclei.
        solidity_threshold: float, ≥ → solid nest, < → ring gland.
            0.75 = literature default for prostate carcinoma grading.
        well_formed_ratio: float, ring-fraction above which the pattern
            is called ``well-formed``.
        poorly_formed_ratio: float, ring-fraction below ``well_formed`` but
            above this is ``poorly-formed``; below is ``absent``.

    Returns:
        dict with:
            n_ring_glands: int, glands with lumens.
            n_solid_nests: int, solid clusters.
            total_structures: int.
            gland_ratio: float, fraction that are ring-like.
            pattern: str, "well-formed", "poorly-formed", or "absent".
    """
    if pixel_size_um is not None and min_cluster_area_um2 is not None:
        min_cluster_area = int(round(min_cluster_area_um2 / (pixel_size_um ** 2)))
    if pixel_size_um is not None and nuc_min_area_um2 is not None:
        nuc_min_area_px = int(round(nuc_min_area_um2 / (pixel_size_um ** 2)))
    if pixel_size_um is not None and nuc_max_area_um2 is not None:
        nuc_max_area = int(round(nuc_max_area_um2 / (pixel_size_um ** 2)))

    seg = segment_nuclei_hae(image, max_area=int(nuc_max_area))
    # Get large connected regions from nuclear mask
    h = seg['hematoxylin']
    h_thresh = seg['threshold']
    nuc_mask = h > h_thresh
    nuc_mask = morphology.remove_small_objects(nuc_mask, max_size=int(nuc_min_area_px))
    nuc_mask = ndimage.binary_fill_holes(nuc_mask)

    labeled = measure.label(nuc_mask)
    props = measure.regionprops(labeled)

    gland_walls = [p for p in props if p.area > min_cluster_area]
    ring_glands = [g for g in gland_walls if g.solidity < solidity_threshold]
    solid_nests = [g for g in gland_walls if g.solidity >= solidity_threshold]

    total = len(gland_walls)
    ratio = len(ring_glands) / total if total > 0 else 0

    if ratio > well_formed_ratio:
        pattern = 'well-formed'
    elif ratio > poorly_formed_ratio:
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
                   max_solidity=0.80,
                   pixel_size_um=None, max_area_um2=None):
    """Detect mitotic figures from nuclear properties.

    Mitotic nuclei have condensed chromosomes: small-medium size,
    intense hematoxylin staining, and irregular shape.

    Physical default (when ``pixel_size_um`` is given) is
    ``max_area_um2=50`` (a condensed mitotic nucleus is smaller than an
    interphase one). The raw ``max_area=800`` default only makes sense
    around 20x (0.5 µm/px → 200 µm²) — at 40x it's too generous and at
    10x it's far too restrictive.

    Args:
        props: list of regionprops.
        hematoxylin: 2D float, hematoxylin channel.
        labeled: 2D int, labeled nuclei image.
        max_area: int, maximum area for mitotic candidate in pixels.
            Ignored if ``pixel_size_um`` is given.
        stain_percentile: float, minimum stain intensity percentile.
        min_eccentricity: float, minimum eccentricity OR...
        max_solidity: float, ...maximum solidity for shape irregularity.
        pixel_size_um: Optional float from ``get_config(core)``.
        max_area_um2: Optional float, overrides physical default (50 µm²).

    Returns:
        dict with:
            mitotic_count: int.
            mitotic_index: float, fraction of nuclei that are mitotic.
            candidates: list of regionprops of mitotic nuclei.
    """
    if pixel_size_um is not None:
        m_a_um2 = 50.0 if max_area_um2 is None else float(max_area_um2)
        max_area = max(1, int(round(m_a_um2 / (float(pixel_size_um) ** 2))))

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


def deconvolve_giemsa(image):
    """Separate Giemsa stain components from an RGB blood smear image.

    Giemsa stain consists of azure B, methylene blue, and eosin Y.
    In blood smears:
    - Blue/purple channel: nuclei, chromatin (parasite DNA, WBC nuclei)
    - Pink/red channel: RBC cytoplasm (eosinophilic), eosinophil granules
    - Background: pale/white

    Uses color-space separation rather than Beer-Lambert deconvolution,
    since Giemsa stain matrices are less standardized than H&E.

    Args:
        image: (H, W, 3) RGB uint8 image of Giemsa-stained blood smear.

    Returns:
        dict with:
            blue_purple: 2D float array — nuclei/chromatin intensity
                (higher = more blue-purple stain).
            pink_red: 2D float array — RBC/eosinophilic intensity
                (higher = more pink-red stain).
            background: 2D bool — background mask (unstained regions).
            dark_bodies: 2D float array — very dark regions (chromatin dots,
                hemozoin pigment, or dense nuclear material).
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected RGB image (H, W, 3)")

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # Blue-purple channel: B > R and B > G (parasite chromatin, WBC nuclei)
    blue_purple = np.clip(B - 0.5 * (R + G), 0, None)

    # Pink-red channel: R > B (RBC cytoplasm, eosinophilic material)
    pink_red = np.clip(R - B, 0, None)

    # Background: bright, low saturation
    brightness = (R + G + B) / 3.0
    saturation = (np.max(image[:, :, :3], axis=2) -
                  np.min(image[:, :, :3], axis=2))
    background = (brightness > 200) & (saturation < 40)

    # Dark bodies: regions with very low overall brightness
    # These include chromatin dots, hemozoin, and dense nuclear material
    dark_bodies = np.clip(150 - brightness, 0, None)

    return {
        'blue_purple': blue_purple,
        'pink_red': pink_red,
        'background': background,
        'dark_bodies': dark_bodies,
    }
