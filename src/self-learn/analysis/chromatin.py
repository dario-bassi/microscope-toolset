"""Chromatin and nuclear architecture analysis.

Analyzes nuclear organization from fluorescence (DAPI/Hoechst) images.
Quantifies chromatin state, nuclear shape integrity, and sub-nuclear
structures (nucleoli, heterochromatin foci) that reflect cell state,
drug effects, and genomic stress.

Stress fiber subtypes:
    heterochromatin -- dense, bright DAPI foci at nuclear periphery
    euchromatin     -- diffuse, faint DAPI signal in nuclear interior
    nucleolus       -- dark (excluded) DAPI region in nucleus center

Functions:
    measure_chromatin_texture    -- Entropy, contrast, homogeneity from DAPI
    heterochromatin_fraction     -- Bright pixel fraction within nucleus
    detect_nucleoli              -- Find dark nucleolus regions in DAPI
    nuclear_shape_metrics        -- Lobulation index, solidity, eccentricity
    detect_micronuclei           -- Find small satellite nuclei near main nucleus
    chromatin_condensation_score -- Overall condensation from texture + distribution
    chromatin_profile            -- Full per-nucleus analysis pipeline
    population_chromatin         -- Aggregate metrics across many cells
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.measure import label as sk_label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, disk


# ── Texture-based chromatin analysis ─────────────────────────────────────────

def measure_chromatin_texture(dapi_img, nucleus_mask):
    """Compute texture descriptors within nucleus from DAPI image.

    Uses local intensity statistics to characterize chromatin organization.
    Condensed chromatin = high contrast, clustered bright foci.
    Decondensed (open) chromatin = uniform, low contrast signal.

    Args:
        dapi_img: 2D float array, DAPI/Hoechst fluorescence image.
        nucleus_mask: 2D bool array, True within nuclear boundary.

    Returns:
        dict with:
            mean_intensity:   float, mean DAPI within nucleus
            std_intensity:    float, std of DAPI within nucleus
            coefficient_of_variation: float, std/mean (normalized variability)
            entropy:          float, Shannon entropy of DAPI histogram (bits)
            contrast_ratio:   float, (P90 - P10) / (P90 + P10)
            skewness:         float, distribution skewness (positive = bright foci)
    """
    dapi_img = np.asarray(dapi_img, dtype=float)
    if dapi_img.ndim == 3:
        dapi_img = dapi_img[:, :, 0]

    pixels = dapi_img[nucleus_mask]
    if len(pixels) == 0:
        return {k: 0.0 for k in ['mean_intensity', 'std_intensity',
                                  'coefficient_of_variation', 'entropy',
                                  'contrast_ratio', 'skewness']}

    mn = float(pixels.mean())
    sd = float(pixels.std())
    cv = sd / mn if mn > 0 else 0.0

    # Shannon entropy from histogram
    counts, _ = np.histogram(pixels, bins=32)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))

    # Contrast ratio from percentiles
    p10, p90 = float(np.percentile(pixels, 10)), float(np.percentile(pixels, 90))
    contrast = (p90 - p10) / (p90 + p10) if (p90 + p10) > 0 else 0.0

    # Skewness
    m3 = float(np.mean((pixels - mn) ** 3))
    skew = m3 / (sd ** 3) if sd > 0 else 0.0

    return {
        'mean_intensity': mn,
        'std_intensity': sd,
        'coefficient_of_variation': cv,
        'entropy': entropy,
        'contrast_ratio': contrast,
        'skewness': skew,
    }


def heterochromatin_fraction(dapi_img, nucleus_mask, bright_percentile=75):
    """Estimate heterochromatin fraction from DAPI intensity distribution.

    Heterochromatin appears as bright dense foci. The fraction of nuclear
    pixels above a percentile threshold approximates the condensed chromatin
    fraction. True heterochromatin ≈ 15-30% in normal cells.

    Args:
        dapi_img: 2D float array, DAPI image.
        nucleus_mask: 2D bool array, nuclear boundary.
        bright_percentile: float (0-100), percentile threshold for
            calling a pixel 'heterochromatin'. Default 75.

    Returns:
        dict with:
            hc_fraction: float [0-1], fraction of nuclear pixels above threshold
            hc_threshold: float, absolute intensity threshold used
            n_nuclear_pixels: int, total pixels in nucleus
            n_hc_pixels: int, bright pixels (heterochromatin)
    """
    dapi_img = np.asarray(dapi_img, dtype=float)
    if dapi_img.ndim == 3:
        dapi_img = dapi_img[:, :, 0]

    pixels = dapi_img[nucleus_mask]
    n_total = len(pixels)
    if n_total == 0:
        return {'hc_fraction': 0.0, 'hc_threshold': 0.0,
                'n_nuclear_pixels': 0, 'n_hc_pixels': 0}

    thresh = float(np.percentile(pixels, bright_percentile))
    n_hc = int(np.sum(pixels >= thresh))
    hc_frac = float(n_hc) / n_total

    return {
        'hc_fraction': hc_frac,
        'hc_threshold': thresh,
        'n_nuclear_pixels': n_total,
        'n_hc_pixels': n_hc,
    }


# ── Sub-nuclear structure detection ──────────────────────────────────────────

def detect_nucleoli(dapi_img, nucleus_mask, smooth_sigma=2.0, dark_threshold=0.4,
                    min_nucleolus_area=10, max_nucleolus_fraction=0.5):
    """Detect nucleoli as dark (DAPI-excluded) regions within the nucleus.

    Nucleoli exclude DAPI staining, appearing as dark holes within the
    bright nuclear signal. Typically 1-4 nucleoli per nucleus.

    Args:
        dapi_img: 2D float array, DAPI image.
        nucleus_mask: 2D bool array, nuclear boundary mask.
        smooth_sigma: float, Gaussian smoothing sigma before detection.
        dark_threshold: float [0-1], fraction of mean nuclear intensity
            below which a region is called a nucleolus hole.
            Default 0.4 = 40% of nuclear mean.
        min_nucleolus_area: int, minimum nucleolus size in pixels.
        max_nucleolus_fraction: float, max nucleolus/nucleus area ratio.
            Prevents entire nucleus from being called a nucleolus.

    Returns:
        dict with:
            n_nucleoli:      int, number of detected nucleoli
            nucleolus_mask:  2D bool array, True at nucleolus positions
            nucleolus_props: list of RegionProps, one per nucleolus
            total_nucleolus_area: int, total nucleolus pixel count
    """
    dapi_img = np.asarray(dapi_img, dtype=float)
    if dapi_img.ndim == 3:
        dapi_img = dapi_img[:, :, 0]

    if not nucleus_mask.any():
        empty = np.zeros(dapi_img.shape, dtype=bool)
        return {'n_nucleoli': 0, 'nucleolus_mask': empty,
                'nucleolus_props': [], 'total_nucleolus_area': 0}

    # Smooth within nucleus
    smoothed = gaussian(dapi_img, sigma=smooth_sigma)

    # Threshold: dark regions within nucleus
    nuc_pixels = smoothed[nucleus_mask]
    nuc_mean = float(nuc_pixels.mean())
    nucleolus_thresh = nuc_mean * dark_threshold

    dark_mask = (smoothed < nucleolus_thresh) & nucleus_mask

    # Remove small objects
    if dark_mask.any():
        dark_mask = remove_small_objects(dark_mask, max_size=min_nucleolus_area)

    # Exclude nucleoli that are too large (artifact protection)
    n_nuc_px = nucleus_mask.sum()
    labeled, n_raw = sk_label(dark_mask, return_num=True)
    props = regionprops(labeled)
    valid_props = [p for p in props
                   if p.area < n_nuc_px * max_nucleolus_fraction]

    nucleolus_mask = np.zeros(dapi_img.shape, dtype=bool)
    for p in valid_props:
        nucleolus_mask[labeled == p.label] = True

    return {
        'n_nucleoli': len(valid_props),
        'nucleolus_mask': nucleolus_mask,
        'nucleolus_props': valid_props,
        'total_nucleolus_area': int(nucleolus_mask.sum()),
    }


def detect_micronuclei(dapi_img, nucleus_mask, search_radius_px=50,
                       min_area=5, max_area_fraction=0.15):
    """Detect micronuclei — small satellite nuclei near the main nucleus.

    Micronuclei form from lagging chromosomes or nuclear envelope rupture.
    They appear as small bright DAPI dots near, but separate from, the
    main nucleus. A micronuclei index > 0 indicates genomic instability.

    Args:
        dapi_img: 2D float array, DAPI image.
        nucleus_mask: 2D bool array, main nucleus mask.
        search_radius_px: int, search radius beyond nucleus boundary.
        min_area: int, minimum micronucleus area in pixels.
        max_area_fraction: float, max area relative to main nucleus.
            Objects larger than this fraction are not micronuclei.

    Returns:
        dict with:
            n_micronuclei: int, number of micronuclei detected
            micronuclei_mask: 2D bool array
            micronuclei_props: list of RegionProps
            main_nucleus_area: int, area of main nucleus
    """
    dapi_img = np.asarray(dapi_img, dtype=float)
    if dapi_img.ndim == 3:
        dapi_img = dapi_img[:, :, 0]

    main_area = int(nucleus_mask.sum())
    empty = np.zeros(dapi_img.shape, dtype=bool)
    if main_area == 0:
        return {'n_micronuclei': 0, 'micronuclei_mask': empty,
                'micronuclei_props': [], 'main_nucleus_area': 0}

    # Define search zone: dilate nucleus boundary
    search_zone = binary_dilation(nucleus_mask, disk(search_radius_px))
    search_zone = search_zone & ~nucleus_mask  # ring around nucleus

    # Threshold the search zone
    if dapi_img[search_zone].size == 0:
        return {'n_micronuclei': 0, 'micronuclei_mask': empty,
                'micronuclei_props': [], 'main_nucleus_area': main_area}

    try:
        thresh = threshold_otsu(dapi_img[search_zone])
    except Exception:
        thresh = dapi_img[search_zone].mean() + dapi_img[search_zone].std()

    candidate_mask = (dapi_img > thresh) & search_zone
    if not candidate_mask.any():
        return {'n_micronuclei': 0, 'micronuclei_mask': empty,
                'micronuclei_props': [], 'main_nucleus_area': main_area}

    candidate_mask = remove_small_objects(candidate_mask, max_size=min_area)
    labeled, _ = sk_label(candidate_mask, return_num=True)
    props = regionprops(labeled)

    max_area = main_area * max_area_fraction
    mn_props = [p for p in props if min_area <= p.area <= max_area]

    micronuclei_mask = np.zeros(dapi_img.shape, dtype=bool)
    for p in mn_props:
        micronuclei_mask[labeled == p.label] = True

    return {
        'n_micronuclei': len(mn_props),
        'micronuclei_mask': micronuclei_mask,
        'micronuclei_props': mn_props,
        'main_nucleus_area': main_area,
    }


# ── Nuclear shape metrics ─────────────────────────────────────────────────────

def nuclear_shape_metrics(nucleus_mask):
    """Compute nuclear shape descriptors for integrity and deformability assessment.

    Normal nucleus: round-to-oval (eccentricity 0.5-0.8, lobulation ~1.2).
    Abnormal nucleus: multi-lobed (lobulation > 2), blebbed (high solidity),
    elongated (eccentricity > 0.95).

    Args:
        nucleus_mask: 2D bool array, nuclear region.

    Returns:
        dict with:
            area_px:           int, nuclear area in pixels
            perimeter:         float, nuclear perimeter in pixels
            lobulation_index:  float, perimeter²/(4π·area) — 1.0 for perfect circle
            solidity:          float, area / convex_hull_area (1=convex, <1=concave)
            eccentricity:      float, 0=circle, 1=line segment
            aspect_ratio:      float, major/minor axis length
            circularity:       float, 4π·area/perimeter² (inverse of lobulation)
    """
    nucleus_mask = np.asarray(nucleus_mask, dtype=bool)
    if not nucleus_mask.any():
        return {k: 0.0 for k in ['area_px', 'perimeter', 'lobulation_index',
                                   'solidity', 'eccentricity', 'aspect_ratio',
                                   'circularity']}

    labeled = sk_label(nucleus_mask.astype(np.uint8))
    props = regionprops(labeled)
    if not props:
        return {k: 0.0 for k in ['area_px', 'perimeter', 'lobulation_index',
                                   'solidity', 'eccentricity', 'aspect_ratio',
                                   'circularity']}

    p = max(props, key=lambda x: x.area)

    area = float(p.area)
    perim = float(p.perimeter) if p.perimeter > 0 else 1.0
    lobulation = (perim ** 2) / (4 * np.pi * area) if area > 0 else 0.0
    circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0
    major = float(p.axis_major_length) if p.axis_major_length > 0 else 1.0
    minor = float(p.axis_minor_length) if p.axis_minor_length > 0 else 1.0

    return {
        'area_px': int(area),
        'perimeter': perim,
        'lobulation_index': float(lobulation),
        'solidity': float(p.solidity),
        'eccentricity': float(p.eccentricity),
        'aspect_ratio': major / minor if minor > 0 else float('inf'),
        'circularity': float(circularity),
    }


# ── Chromatin condensation ────────────────────────────────────────────────────

def chromatin_condensation_score(dapi_img, nucleus_mask):
    """Compute a single condensation score from chromatin texture metrics.

    Combines coefficient of variation, contrast ratio, and skewness into
    a composite score. Higher score = more condensed chromatin.

    Range: 0 (fully decondensed, uniform) to 1 (maximally condensed, punctate).

    Args:
        dapi_img: 2D float array, DAPI image.
        nucleus_mask: 2D bool array, nuclear mask.

    Returns:
        float, condensation score in [0, 1].
    """
    texture = measure_chromatin_texture(dapi_img, nucleus_mask)
    if texture['mean_intensity'] == 0:
        return 0.0

    # Normalize each metric to [0,1] range using empirical bounds
    # CV: 0 (uniform) → 1 (maximally variable)
    cv_norm = min(1.0, texture['coefficient_of_variation'] / 1.0)

    # Contrast: already in [0,1]
    contrast_norm = texture['contrast_ratio']

    # Skewness: positive = bright foci (condensed). Clip at 3.
    skew_norm = min(1.0, max(0.0, texture['skewness'] / 3.0))

    # Weighted average
    score = 0.4 * cv_norm + 0.4 * contrast_norm + 0.2 * skew_norm
    return float(np.clip(score, 0.0, 1.0))


# ── Per-nucleus pipeline ──────────────────────────────────────────────────────

def chromatin_profile(dapi_img, nucleus_mask, detect_nuc=True, detect_mn=False,
                      **kwargs):
    """Full per-nucleus chromatin analysis.

    Combines texture, heterochromatin fraction, shape, and optionally
    nucleolus and micronuclei detection.

    Args:
        dapi_img: 2D float array, DAPI/Hoechst image.
        nucleus_mask: 2D bool array, nuclear boundary.
        detect_nuc: bool, whether to detect nucleoli. Default True.
        detect_mn: bool, whether to detect micronuclei. Default False.
        **kwargs: passed to detect_nucleoli() and detect_micronuclei().

    Returns:
        dict with all fields from:
            - measure_chromatin_texture()
            - heterochromatin_fraction()
            - nuclear_shape_metrics()
            - detect_nucleoli() (if detect_nuc=True)
            - detect_micronuclei() (if detect_mn=True)
            - condensation_score: float
    """
    result = {}
    result.update(measure_chromatin_texture(dapi_img, nucleus_mask))
    result.update(heterochromatin_fraction(dapi_img, nucleus_mask))
    result.update(nuclear_shape_metrics(nucleus_mask))
    result['condensation_score'] = chromatin_condensation_score(dapi_img, nucleus_mask)

    if detect_nuc:
        nuc_result = detect_nucleoli(dapi_img, nucleus_mask, **kwargs)
        result['n_nucleoli'] = nuc_result['n_nucleoli']
        result['total_nucleolus_area'] = nuc_result['total_nucleolus_area']
        result['nucleolus_mask'] = nuc_result['nucleolus_mask']

    if detect_mn:
        mn_result = detect_micronuclei(dapi_img, nucleus_mask, **kwargs)
        result['n_micronuclei'] = mn_result['n_micronuclei']
        result['micronuclei_mask'] = mn_result['micronuclei_mask']

    return result


# ── Population-level analysis ─────────────────────────────────────────────────

def population_chromatin(profiles):
    """Aggregate chromatin profiles across a cell population.

    Args:
        profiles: list of dicts from chromatin_profile(), one per cell.

    Returns:
        dict with:
            n_cells: int
            mean_condensation: float
            std_condensation: float
            mean_hc_fraction: float
            mean_n_nucleoli: float
            mean_lobulation: float
            abnormal_nuclear_fraction: float, fraction with lobulation > 2
            population_profiles: list of dicts (input profiles)
    """
    if not profiles:
        return {
            'n_cells': 0, 'mean_condensation': 0.0, 'std_condensation': 0.0,
            'mean_hc_fraction': 0.0, 'mean_n_nucleoli': 0.0,
            'mean_lobulation': 0.0, 'abnormal_nuclear_fraction': 0.0,
            'population_profiles': [],
        }

    condensations = [p.get('condensation_score', 0.0) for p in profiles]
    hc_fracs = [p.get('hc_fraction', 0.0) for p in profiles]
    n_nucleoli = [p.get('n_nucleoli', 0) for p in profiles]
    lobulations = [p.get('lobulation_index', 0.0) for p in profiles]

    abnormal = sum(1 for l in lobulations if l > 2.0)

    return {
        'n_cells': len(profiles),
        'mean_condensation': float(np.mean(condensations)),
        'std_condensation': float(np.std(condensations)),
        'mean_hc_fraction': float(np.mean(hc_fracs)),
        'mean_n_nucleoli': float(np.mean(n_nucleoli)),
        'mean_lobulation': float(np.mean(lobulations)),
        'abnormal_nuclear_fraction': float(abnormal) / len(profiles),
        'population_profiles': profiles,
    }
