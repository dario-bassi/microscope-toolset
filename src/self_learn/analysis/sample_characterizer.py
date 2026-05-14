"""Automated sample characterization from initial images.

Analyzes brightfield and fluorescence channels to determine:
- Sample type (cells, tissue, organisms, particles)
- Staining method (H&E, fluorescence, unstained, Giemsa, etc.)
- Recommended analysis workflow
- Key features to check (necrosis, mitoses, etc.)

Functions:
    characterize_image  -- Classify a single image by content type
    characterize_sample -- Classify multi-channel sample
    suggest_workflow    -- Recommend analysis pipeline
    check_completeness  -- Verify all expected analysis steps done
"""

import numpy as np
from scipy import ndimage


def characterize_image(image):
    """Characterize a single image by its visual properties.

    Analyzes intensity distribution, texture, and spatial features
    to determine what kind of microscopy image this is.

    Args:
        image: 2D or 3D (RGB) array.

    Returns:
        dict with:
            is_rgb: bool, whether image is color (3-channel).
            is_fluorescence: bool, bright objects on dark background.
            is_brightfield: bool, objects on lighter background.
            mean_intensity: float.
            dynamic_range: float, fraction of bit depth used.
            has_texture: bool, significant spatial variation.
            sparsity: float, fraction of pixels near background.
            n_components: int, number of color channels.
            dominant_color: str or None, for RGB images.
    """
    image = np.asarray(image)
    is_rgb = image.ndim == 3 and image.shape[2] == 3

    if is_rgb:
        gray = np.mean(image.astype(float), axis=2)
        r, g, b = (image[:, :, i].astype(float) for i in range(3))
        # Dominant color channel
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
        max_ch = max(r_mean, g_mean, b_mean)
        if max_ch == r_mean:
            dominant_color = 'red'
        elif max_ch == g_mean:
            dominant_color = 'green'
        else:
            dominant_color = 'blue'
    else:
        gray = image.astype(float)
        dominant_color = None

    # Basic stats
    mean_val = float(gray.mean())
    bit_depth = 255.0 if gray.max() <= 255 else 65535.0
    dynamic_range = float((gray.max() - gray.min()) / bit_depth)

    # Fluorescence vs brightfield
    # Fluorescence: mostly dark, sparse bright objects
    # Brightfield: moderate mean, objects darker than background
    median_val = float(np.median(gray))
    is_fluorescence = median_val < 0.25 * bit_depth and dynamic_range > 0.3
    is_brightfield = median_val > 0.3 * bit_depth

    # Sparsity: fraction of pixels near background
    bg_threshold = np.percentile(gray, 10) + 0.1 * (np.percentile(gray, 90) - np.percentile(gray, 10))
    sparsity = float((gray < bg_threshold).mean())

    # Texture: spatial variation via Laplacian
    lap = ndimage.laplace(gray)
    has_texture = float(np.var(lap)) > 1.0

    return {
        'is_rgb': is_rgb,
        'is_fluorescence': is_fluorescence,
        'is_brightfield': is_brightfield,
        'mean_intensity': round(mean_val, 1),
        'dynamic_range': round(dynamic_range, 3),
        'has_texture': has_texture,
        'sparsity': round(sparsity, 3),
        'n_components': 3 if is_rgb else 1,
        'dominant_color': dominant_color,
    }


def characterize_sample(channels):
    """Characterize a multi-channel sample.

    Args:
        channels: dict mapping channel_name → image array.
            e.g. {"brightfield": img_bf, "GFP": img_gfp, "DAPI": img_dapi}

    Returns:
        dict with:
            sample_type: str, one of "tissue_hae", "fluorescence_cells",
                "brightfield_cells", "organisms", "particles", "unknown".
            staining: str, detected staining type.
            channels: dict of per-channel characterization.
            recommended_analysis: list of str, suggested analysis steps.
            checklist: list of str, things to verify before submitting.
    """
    ch_info = {}
    for name, img in channels.items():
        ch_info[name] = characterize_image(img)

    # Determine sample type from channel properties
    has_rgb = any(c['is_rgb'] for c in ch_info.values())
    has_fluor = any(c['is_fluorescence'] for c in ch_info.values())
    has_bf = any(c['is_brightfield'] for c in ch_info.values())
    n_channels = len(channels)

    # H&E tissue: RGB brightfield with blue-purple and pink staining
    if has_rgb and has_bf:
        # Check for H&E colors
        for name, img in channels.items():
            if ch_info[name]['is_rgb'] and ch_info[name]['is_brightfield']:
                rgb = np.asarray(img)
                r_mean = rgb[:, :, 0].astype(float).mean()
                b_mean = rgb[:, :, 2].astype(float).mean()
                if b_mean > r_mean * 0.5:  # Significant blue component
                    sample_type = 'tissue_hae'
                    staining = 'H&E'
                    break
        else:
            sample_type = 'brightfield_cells'
            staining = 'unstained'
    elif has_fluor and n_channels >= 2:
        sample_type = 'fluorescence_cells'
        staining = 'fluorescence'
    elif has_fluor:
        sample_type = 'fluorescence_cells'
        staining = 'single_channel_fluorescence'
    elif has_bf:
        sample_type = 'brightfield_cells'
        staining = 'unstained'
    else:
        sample_type = 'unknown'
        staining = 'unknown'

    # Generate recommendations
    recommended, checklist = suggest_workflow(sample_type, ch_info)

    return {
        'sample_type': sample_type,
        'staining': staining,
        'channels': ch_info,
        'recommended_analysis': recommended,
        'checklist': checklist,
    }


def suggest_workflow(sample_type, channel_info=None):
    """Recommend analysis pipeline based on sample type.

    Args:
        sample_type: str from characterize_sample().
        channel_info: optional dict of per-channel info.

    Returns:
        (recommended_steps, checklist) tuple of string lists.
    """
    recommended = []
    checklist = []

    if sample_type == 'tissue_hae':
        recommended = [
            '10x_overview',
            'hae_deconvolution',
            'nuclear_segmentation',
            'gland_architecture',
            'mitotic_figure_detection',
            'necrosis_check',
            'nuclear_pleomorphism',
            'tumor_grading',
        ]
        checklist = [
            'Check for necrosis (eosinophilic acellular regions)',
            'Assess gland formation (ring vs solid clusters)',
            'Quantify nuclear size CV',
            'Count mitotic figures at 20x or 40x',
            'Look for lymphocyte infiltrate',
            'Use watershed to split touching nuclei',
        ]
    elif sample_type == 'fluorescence_cells':
        recommended = [
            '10x_overview',
            'channel_scout',
            'background_subtraction',
            'cell_segmentation',
            'morphometry',
            'intensity_quantification',
        ]
        checklist = [
            'Verify which channel has the target signal',
            'Check for autofluorescence cross-talk',
            'Subtract background before measurements',
            'Exclude edge objects (>3px from border)',
            'Correct for photobleaching if timelapse',
        ]
    elif sample_type == 'brightfield_cells':
        recommended = [
            '10x_overview',
            'adaptive_threshold',
            'cell_detection',
            'morphometry',
            'counting',
        ]
        checklist = [
            'Verify cells are darker or lighter than background',
            'Choose correct threshold direction (bright vs dark objects)',
            'Exclude edge objects',
            'Use watershed for touching cells',
            'Cross-validate count with second method if critical',
        ]
    else:
        recommended = [
            '10x_overview',
            'channel_scout',
            'segmentation',
            'measurement',
        ]
        checklist = [
            'Verify channel selection',
            'Check magnification is appropriate',
            'Exclude edge artifacts',
        ]

    return recommended, checklist


def check_completeness(sample_type, completed_steps):
    """Check if all expected analysis steps have been performed.

    Args:
        sample_type: str from characterize_sample().
        completed_steps: list of str, steps that have been done.

    Returns:
        dict with:
            complete: bool, whether all critical steps are done.
            missing: list of str, steps that should still be done.
            warnings: list of str, non-critical suggestions.
    """
    recommended, _ = suggest_workflow(sample_type)
    completed_set = set(completed_steps)

    # Critical steps that must not be skipped
    critical = {
        'tissue_hae': ['hae_deconvolution', 'nuclear_segmentation',
                       'necrosis_check', 'tumor_grading'],
        'fluorescence_cells': ['channel_scout', 'cell_segmentation',
                              'background_subtraction'],
        'brightfield_cells': ['cell_detection', 'counting'],
    }

    type_critical = critical.get(sample_type, ['segmentation'])
    missing = [s for s in type_critical if s not in completed_set]
    suggested = [s for s in recommended if s not in completed_set
                 and s not in missing]

    return {
        'complete': len(missing) == 0,
        'missing': missing,
        'warnings': [f"Consider also: {s}" for s in suggested[:3]],
    }
