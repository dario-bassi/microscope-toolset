"""Context-aware parameter suggestions for detection and segmentation.

Suggests optimal parameters based on physical context (magnification,
pixel size, expected object size, density) rather than requiring
users to guess pixel-domain values.

Functions:
    suggest_watershed_params   -- min_distance, dt_threshold from expected diameter
    suggest_tracking_params    -- max_dist, min_length from speed and dt
    suggest_detection_params   -- threshold, min_area from object type and pixel size
    suggest_log_params         -- sigma, peak_thresh for LoG blob detection
    parameter_report           -- formatted summary for logging
"""

import numpy as np


# ── Physical constants ──────────────────────────────────────────────────

# Typical object diameters in µm (median values from literature)
OBJECT_DIAMETERS_UM = {
    'nucleus': 10.0,        # mammalian nucleus
    'bacterium': 1.5,       # E. coli length
    'yeast': 5.0,           # S. cerevisiae
    'rbc': 7.0,             # red blood cell
    'wbc': 12.0,            # white blood cell
    'organoid': 200.0,      # varies widely
    'neuron_soma': 15.0,    # cortical neuron
    'c_elegans': 50.0,      # adult body width
}

# Typical speeds in µm/s
OBJECT_SPEEDS_UM_S = {
    'bacterium': 15.0,      # E. coli run speed
    'neutrophil': 5.0,      # crawling
    'fibroblast': 0.3,      # slow migration
    'c_elegans': 100.0,     # crawling speed
    'yeast': 0.0,           # sessile (budding)
}


def suggest_watershed_params(pixel_size_um, expected_diameter_um=None,
                              object_type=None):
    """Suggest watershed segmentation parameters from physical context.

    Converts physical object size to optimal pixel-domain parameters
    for :func:`~self_learn.detection.cells.watershed_split`.

    Args:
        pixel_size_um: Micrometers per pixel (e.g. 1.0 for 10x, 0.5 for 20x).
        expected_diameter_um: Expected object diameter in µm.
            If None, inferred from object_type.
        object_type: String key into OBJECT_DIAMETERS_UM
            (e.g. 'nucleus', 'yeast', 'bacterium').

    Returns:
        dict with:
            min_distance: int, minimum peak separation in pixels.
            dt_threshold: float, minimum distance-transform value.
            expected_radius_px: float, expected object radius in pixels.
            notes: str, explanation of parameter choices.
    """
    diameter_um = _resolve_diameter(expected_diameter_um, object_type)
    radius_px = (diameter_um / 2.0) / pixel_size_um

    # min_distance ≈ 0.7 * radius to allow splitting touching objects
    # while avoiding false splits from shape irregularity
    min_distance = max(3, int(round(0.7 * radius_px)))

    # dt_threshold: peaks must be at least this deep in the distance transform
    # Small objects (radius < 5px): lower threshold to avoid losing them
    # Large objects: higher threshold to suppress noise
    dt_threshold = max(1.5, min(radius_px * 0.3, 5.0))

    notes = (f"Object ~{diameter_um:.0f}µm diameter = {2*radius_px:.1f}px "
             f"at {pixel_size_um}µm/px. "
             f"min_distance={min_distance} (~0.7 × radius).")

    return {
        'min_distance': min_distance,
        'dt_threshold': round(dt_threshold, 1),
        'expected_radius_px': round(radius_px, 1),
        'notes': notes,
    }


def suggest_tracking_params(pixel_size_um, expected_speed_um_s=None,
                             dt_s=1.0, object_type=None):
    """Suggest tracking parameters from physical context.

    Converts expected object speed and frame interval to pixel-domain
    parameters for :func:`~self_learn.analysis.tracking.match_centroids` and
    :func:`~self_learn.analysis.run_tumble.build_tracks`.

    Args:
        pixel_size_um: Micrometers per pixel.
        expected_speed_um_s: Expected object speed in µm/s.
            If None, inferred from object_type.
        dt_s: Frame interval in seconds.
        object_type: String key into OBJECT_SPEEDS_UM_S.

    Returns:
        dict with:
            max_dist_px: float, maximum linking distance in pixels.
            min_track_length: int, minimum track length in frames.
            expected_displacement_px: float, expected per-frame displacement.
            notes: str, explanation.
    """
    if expected_speed_um_s is None and object_type is not None:
        expected_speed_um_s = OBJECT_SPEEDS_UM_S.get(object_type)
    if expected_speed_um_s is None:
        raise ValueError("Provide expected_speed_um_s or a known object_type")

    displacement_um = expected_speed_um_s * dt_s
    displacement_px = displacement_um / pixel_size_um

    # max_dist: allow 2x expected displacement (handles speed variability)
    # but at least 3px to handle centroid jitter
    max_dist_px = max(3.0, 2.0 * displacement_px)

    # min_track_length: at least 5 frames for meaningful dynamics
    # More for slow objects where noise dominates
    if displacement_px < 1.0:
        min_length = 10  # sub-pixel: need more frames for signal
    elif displacement_px < 3.0:
        min_length = 8
    else:
        min_length = 5

    # Warn about sub-pixel displacement
    notes_parts = [
        f"Speed ~{expected_speed_um_s:.1f}µm/s × {dt_s}s = "
        f"{displacement_um:.1f}µm = {displacement_px:.1f}px per frame."
    ]
    if displacement_px < 1.0:
        notes_parts.append(
            f"WARNING: Sub-pixel displacement ({displacement_px:.2f}px). "
            f"Consider higher magnification or longer interval."
        )
    if displacement_px < 2.0:
        notes_parts.append(
            f"Centroid jitter (~1-2px) may dominate. "
            f"Use Savitzky-Golay smoothing on positions before speed calc."
        )

    return {
        'max_dist_px': round(max_dist_px, 1),
        'min_track_length': min_length,
        'expected_displacement_px': round(displacement_px, 1),
        'notes': ' '.join(notes_parts),
    }


def suggest_detection_params(pixel_size_um, object_type=None,
                              expected_diameter_um=None, signal_type='bright'):
    """Suggest detection parameters for cell/object detection.

    Provides min_area, threshold guidance, and LoG sigma for
    :func:`~self_learn.detection.cells.detect_cells` and related functions.

    Args:
        pixel_size_um: Micrometers per pixel.
        object_type: String key into OBJECT_DIAMETERS_UM.
        expected_diameter_um: Expected diameter in µm. Overrides object_type.
        signal_type: 'bright' (fluorescence) or 'dark' (phase contrast).

    Returns:
        dict with:
            min_area_px: int, minimum object area in pixels.
            max_area_px: int or None, maximum object area.
            log_sigma: float, suggested LoG sigma for blob detection.
            threshold_sigma: float, suggested sigma multiplier for thresholding.
            notes: str, guidance text.
    """
    diameter_um = _resolve_diameter(expected_diameter_um, object_type)
    diameter_px = diameter_um / pixel_size_um
    radius_px = diameter_px / 2.0
    area_px = np.pi * radius_px ** 2

    # min_area: ~25% of expected area (catch small/partial objects)
    min_area = max(5, int(round(0.25 * area_px)))

    # max_area: 4x expected area (reject merged clusters)
    max_area = int(round(4.0 * area_px))

    # LoG sigma: radius / sqrt(2) is optimal for detecting Gaussian blobs
    log_sigma = max(1.5, radius_px / np.sqrt(2))

    # Threshold guidance
    if signal_type == 'bright':
        threshold_sigma = 2.0  # fluorescence: good SNR
    else:
        threshold_sigma = 1.5  # phase contrast: lower SNR

    notes_parts = [
        f"Object ~{diameter_um:.0f}µm = {diameter_px:.0f}px diameter, "
        f"area ~{area_px:.0f}px².",
    ]
    if diameter_px < 3:
        notes_parts.append(
            "WARNING: Object smaller than 3px — consider higher magnification."
        )
    if diameter_px > 200:
        notes_parts.append(
            "Large object — consider tiling or lower magnification."
        )

    return {
        'min_area_px': min_area,
        'max_area_px': max_area,
        'log_sigma': round(log_sigma, 1),
        'threshold_sigma': threshold_sigma,
        'expected_diameter_px': round(diameter_px, 1),
        'expected_area_px': round(area_px, 0),
        'notes': ' '.join(notes_parts),
    }


def suggest_log_params(pixel_size_um, expected_diameter_um=None,
                        object_type=None):
    """Suggest Laplacian of Gaussian parameters for blob detection.

    Optimized for :func:`~self_learn.detection.cells.count_blobs_log` and
    :func:`~self_learn.analysis.condensate.detect_foci`.

    Args:
        pixel_size_um: Micrometers per pixel.
        expected_diameter_um: Expected blob diameter in µm.
        object_type: String key into OBJECT_DIAMETERS_UM.

    Returns:
        dict with:
            sigma: float, LoG scale parameter.
            filter_size: int, local maximum filter size.
            notes: str.
    """
    diameter_um = _resolve_diameter(expected_diameter_um, object_type)
    radius_px = (diameter_um / 2.0) / pixel_size_um

    # LoG optimal sigma = radius / sqrt(2)
    sigma = max(1.0, radius_px / np.sqrt(2))

    # Filter size for non-maximum suppression: ~2*sigma + 1
    filter_size = max(3, int(round(2 * sigma + 1)))
    if filter_size % 2 == 0:
        filter_size += 1  # keep odd

    return {
        'sigma': round(sigma, 1),
        'filter_size': filter_size,
        'expected_radius_px': round(radius_px, 1),
        'notes': (f"LoG sigma={sigma:.1f} for ~{diameter_um:.0f}µm objects "
                  f"({radius_px:.1f}px radius) at {pixel_size_um}µm/px."),
    }


def suggest_nc_params(pixel_size_um, magnification=None):
    """Suggest nuclear-cytoplasmic ratio measurement parameters.

    Based on PSF size relative to nuclear boundary.

    Args:
        pixel_size_um: Micrometers per pixel.
        magnification: Objective magnification (10, 20, 40, etc.).

    Returns:
        dict with:
            guard_band: int, guard band pixels for compute_nc_ratio().
            dilation_size: int, structuring element for cytoplasm mask.
            notes: str.
    """
    # PSF FWHM in µm: ~0.5µm for high-NA objectives
    psf_fwhm_um = 0.5
    psf_px = psf_fwhm_um / pixel_size_um

    # Guard band: 1 PSF width to avoid boundary contamination
    guard_band = max(0, int(round(psf_px)))

    # Cytoplasm dilation: extend nucleus mask to create cytoplasm ring
    # Should be at least 2x PSF to sample beyond the PSF halo
    nuc_diameter_px = 10.0 / pixel_size_um  # ~10µm typical nucleus
    dilation_size = max(5, int(round(min(nuc_diameter_px * 0.5, 15))))

    notes_parts = [
        f"PSF ~{psf_px:.1f}px at {pixel_size_um}µm/px.",
    ]
    if psf_px > 1.5:
        notes_parts.append(
            f"PSF > 1.5px: significant boundary blur. "
            f"guard_band={guard_band} recommended. "
            f"Consider 40x for accurate N:C ratios."
        )
    else:
        notes_parts.append(
            f"PSF < 1.5px: good resolution for N:C measurement."
        )

    return {
        'guard_band': guard_band,
        'dilation_size': dilation_size,
        'psf_px': round(psf_px, 1),
        'notes': ' '.join(notes_parts),
    }


def suggest_sampling_params(expected_fraction, objects_per_fov,
                            target_ci_width=0.10, confidence=0.95):
    """Suggest how many FOVs to image for a given event frequency.

    Uses Wilson score interval width to determine minimum sample size
    for reliable proportion estimation.

    Args:
        expected_fraction: Expected fraction of positive events (0-1).
            E.g., 0.15 for 15% parasitemia, 0.05 for 5% mitotic index.
        objects_per_fov: Expected objects per field of view.
        target_ci_width: Target 95% CI half-width. Default 0.10 means
            ±10 percentage points. Use 0.05 for tighter estimates.
        confidence: Confidence level (default 0.95).

    Returns:
        dict with:
            min_objects: Minimum total objects to count.
            min_fovs: Minimum FOVs to image (ceil of min_objects/objects_per_fov).
            expected_positives: Expected positive events in sample.
            notes: str.
    """
    from scipy.stats import norm

    p = max(0.01, min(0.99, expected_fraction))
    z = norm.ppf(1 - (1 - confidence) / 2)

    # Wilson CI half-width: z * sqrt(p*(1-p)/n + z²/(4n²)) / (1 + z²/n)
    # Approximate: need n ≈ z² * p*(1-p) / target_ci_width²
    n_approx = z**2 * p * (1 - p) / target_ci_width**2
    min_objects = max(30, int(np.ceil(n_approx)))  # minimum 30 for CLT

    min_fovs = max(1, int(np.ceil(min_objects / max(1, objects_per_fov))))
    expected_positives = int(round(min_objects * p))

    notes_parts = []
    if expected_fraction < 0.1:
        notes_parts.append(
            f"Rare events ({expected_fraction:.0%}): need {min_fovs}+ FOVs "
            f"for ±{target_ci_width:.0%} precision."
        )
    if min_fovs > 10:
        notes_parts.append(
            "Consider multi-position MDA with grid scan."
        )
    if expected_positives < 10:
        notes_parts.append(
            f"Warning: only ~{expected_positives} positives expected. "
            f"Consider wider CI or more FOVs."
        )

    return {
        'min_objects': min_objects,
        'min_fovs': min_fovs,
        'expected_positives': expected_positives,
        'notes': ' '.join(notes_parts) if notes_parts else "Sample size adequate.",
    }


def parameter_report(params_dict, title="Parameter Suggestions"):
    """Format parameter suggestions as a readable report.

    Args:
        params_dict: Dict of parameter name → value (from suggest_* functions).
        title: Report title.

    Returns:
        str: Formatted multi-line report.
    """
    lines = [title, "=" * len(title)]
    notes = None
    for key, value in params_dict.items():
        if key == 'notes':
            notes = value
            continue
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.2f}")
        else:
            lines.append(f"  {key}: {value}")
    if notes:
        lines.append(f"  Notes: {notes}")
    return '\n'.join(lines)


# ── Internal ────────────────────────────────────────────────────────────

def _resolve_diameter(expected_diameter_um, object_type):
    """Resolve diameter from explicit value or object type lookup."""
    if expected_diameter_um is not None:
        return float(expected_diameter_um)
    if object_type is not None:
        d = OBJECT_DIAMETERS_UM.get(object_type)
        if d is not None:
            return d
        raise ValueError(
            f"Unknown object_type '{object_type}'. "
            f"Known types: {sorted(OBJECT_DIAMETERS_UM.keys())}"
        )
    raise ValueError("Provide expected_diameter_um or object_type")
