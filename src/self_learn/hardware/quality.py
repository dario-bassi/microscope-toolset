"""Image quality assessment and microscope optimization.

Provides functions to assess image quality, detect common issues
(out of focus, wrong exposure, wrong channel), and suggest corrections.

Inspired by AutoPilot (quality-driven microscopy) and the Smart Microscopy
Working Group's quality-driven paradigm.
"""

import numpy as np


def assess_image(img):
    """Classify an image's quality and likely channel type.

    Returns:
        dict with:
            channel_type: 'brightfield', 'fluorescence', or 'empty'
            quality: 'good', 'underexposed', 'overexposed', 'out_of_focus'
            mean: float
            std: float
            snr: float (signal-to-noise ratio estimate)
            laplacian_var: float (focus sharpness metric)
    """
    img_f = img.astype(float)
    mean = float(img_f.mean())
    std = float(img_f.std())
    snr = std / max(mean, 1.0)  # coefficient of variation as SNR proxy

    # Laplacian variance as focus metric
    # Higher = sharper. Normalized by image mean to be scale-invariant.
    from cv2 import Laplacian, CV_64F
    lap = Laplacian(img_f, CV_64F)
    lap_var = float(lap.var())

    # Channel classification
    if mean < 3 and std < 2:
        channel_type = 'empty'
    elif mean < 10 and snr > 1.0:
        # Low mean but high std/mean ratio = bright spots on dark background
        channel_type = 'fluorescence'
    elif mean > 20:
        channel_type = 'brightfield'
    else:
        channel_type = 'unknown'

    # Quality assessment
    if channel_type == 'empty':
        quality = 'no_signal'
    elif mean < 5 and channel_type != 'fluorescence':
        quality = 'underexposed'
    elif mean > 240:
        quality = 'overexposed'
    elif channel_type == 'brightfield' and lap_var < 10:
        quality = 'out_of_focus'
    elif channel_type == 'fluorescence' and lap_var < 5:
        quality = 'out_of_focus'
    else:
        quality = 'good'

    return {
        'channel_type': channel_type,
        'quality': quality,
        'mean': round(mean, 1),
        'std': round(std, 1),
        'snr': round(snr, 3),
        'laplacian_var': round(lap_var, 1),
    }


def check_focus_quality(core, z_range=5.0, z_step=1.0):
    """Sweep Z positions and find best focus using Laplacian variance.

    Uses MDA events via ``run_events()`` instead of a manual snap loop.

    Args:
        core: pymmcore-plus core.
        z_range: total Z range to sweep (centered on current Z).
        z_step: step size in microns.

    Returns:
        dict with:
            best_z: Z position with highest sharpness
            current_z: starting Z position
            focus_curve: list of (z, laplacian_var) tuples
            improvement: ratio of best focus to current focus
    """
    from cv2 import Laplacian, CV_64F
    from useq import MDAEvent
    from .core import run_events, set_z

    current_z = float(core.getPosition('ZStage'))
    z_positions = np.arange(
        current_z - z_range / 2,
        current_z + z_range / 2 + z_step / 2,
        z_step
    )

    focus_curve = []

    def on_frame(img, event):
        z = event.z_pos if event.z_pos is not None else current_z
        img_f = img.astype(float)
        lap_var = float(Laplacian(img_f, CV_64F).var())
        focus_curve.append((round(float(z), 2), round(lap_var, 1)))

    events = [
        MDAEvent(z_pos=float(z), index={"z": i})
        for i, z in enumerate(z_positions)
    ]
    run_events(core, events, on_frame=on_frame)

    # Find best Z
    best_idx = max(range(len(focus_curve)), key=lambda i: focus_curve[i][1])
    best_z = focus_curve[best_idx][0]
    best_sharpness = focus_curve[best_idx][1]

    # Current position sharpness
    current_sharpness = None
    for z, sharp in focus_curve:
        if abs(z - current_z) < z_step / 2:
            current_sharpness = sharp
            break

    improvement = (best_sharpness / current_sharpness
                   if current_sharpness and current_sharpness > 0 else 1.0)

    # Move to best focus
    set_z(core, best_z)

    return {
        'best_z': best_z,
        'current_z': round(current_z, 2),
        'focus_curve': focus_curve,
        'improvement': round(improvement, 2),
    }


def estimate_snr(img, signal_mask=None):
    """Estimate signal-to-noise ratio of an image.

    If signal_mask is provided, uses mask to separate signal from background.
    Otherwise, uses adaptive thresholding to estimate.

    Args:
        img: 2D image array.
        signal_mask: optional boolean mask (True = signal pixels).

    Returns:
        dict with:
            snr_db: SNR in decibels
            signal_mean: mean signal intensity
            noise_std: estimated noise standard deviation
    """
    img_f = img.astype(float)

    if signal_mask is not None:
        signal = img_f[signal_mask]
        noise = img_f[~signal_mask]
    else:
        # Simple threshold-based separation
        m, s = float(img_f.mean()), float(img_f.std())
        thresh = m + s
        signal = img_f[img_f > thresh]
        noise = img_f[img_f <= thresh]

    if len(signal) == 0 or len(noise) == 0:
        return {'snr_db': 0.0, 'signal_mean': 0.0, 'noise_std': 0.0}

    signal_mean = float(signal.mean())
    noise_std = float(noise.std())

    if noise_std > 0:
        snr_linear = signal_mean / noise_std
        snr_db = 20 * np.log10(snr_linear)
    else:
        snr_db = float('inf')

    return {
        'snr_db': round(snr_db, 1),
        'signal_mean': round(signal_mean, 1),
        'noise_std': round(noise_std, 2),
    }


def validate_acquisition(core, channel=None, group=None):
    """Quick validation that the microscope is in a usable state.

    Snaps a BF image and checks basic quality metrics.

    Args:
        core: CMMCorePlus instance.
        channel: channel config name. If None, autodiscovers a BF-like
            preset from available_channels and falls back to 'brightfield'.
        group: config group name. Auto-discovered if None.

    Returns:
        dict with:
            ok: bool — True if acquisition is usable
            issues: list of issue descriptions
            assessment: full image assessment dict
    """
    from .config import resolve_brightfield_channel, resolve_channel_group

    group = resolve_channel_group(core, group)
    channel = resolve_brightfield_channel(core, channel) or 'brightfield'
    if group is not None:
        core.setConfig(group, channel)
    core.snapImage()
    img = core.getImage().copy()

    assessment = assess_image(img)
    issues = []

    if assessment['quality'] == 'no_signal':
        issues.append('No signal detected — check illumination')
    elif assessment['quality'] == 'underexposed':
        issues.append(f'Underexposed (mean={assessment["mean"]})')
    elif assessment['quality'] == 'overexposed':
        issues.append(f'Overexposed (mean={assessment["mean"]})')
    elif assessment['quality'] == 'out_of_focus':
        issues.append(f'Out of focus (laplacian_var={assessment["laplacian_var"]})')

    if assessment['channel_type'] == 'fluorescence':
        issues.append('Expected BF but got fluorescence — wrong channel?')

    return {
        'ok': len(issues) == 0,
        'issues': issues,
        'assessment': assessment,
    }
