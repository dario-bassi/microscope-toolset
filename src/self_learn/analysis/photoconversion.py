"""Photoconversion and photoactivation analysis.

Analyzes fluorescence redistribution after photoactivation,
photoconversion, or photoswitching experiments. Tracks signal
spread/decay from the activated region to measure transport,
diffusion, and turnover.

Functions:
    count_converted_cells  -- Count converted cells using dual-channel verification
    measure_activation     -- Measure activated region signal over time
    signal_spread          -- Quantify spatial spread from activation zone
    half_life              -- Estimate signal half-life from decay curve
    transport_rate         -- Measure directional transport from activation zone
    activation_efficiency  -- Compute activation efficiency from pre/post
"""

import numpy as np
from scipy import optimize, ndimage
from skimage.feature import peak_local_max
from skimage import filters, morphology


def count_converted_cells(
    bf_image, nuc_pre, nuc_post, mem_post, roi_mask,
    bf_sigma=5.0, bf_min_distance=15, bf_threshold_abs=80,
    nuc_drop_ratio=0.7, mem_bright_ratio=1.5, window=5,
):
    """Count photoconverted cells using dual-channel verification.

    Finds cell centers inside the ROI (from brightfield), then verifies
    conversion by checking nucleus drop OR membrane increase around each
    cell center. More robust than single-channel counting.

    Args:
        bf_image: 2D array, brightfield image (for cell center detection).
        nuc_pre: 2D array, nucleus-channel before conversion.
        nuc_post: 2D array, nucleus-channel after conversion.
        mem_post: 2D array, membrane-channel after conversion.
        roi_mask: 2D bool array, SLM-illuminated region.
        bf_sigma: float, Gaussian smoothing sigma for BF peak detection.
        bf_min_distance: int, min_distance for peak_local_max on BF.
        bf_threshold_abs: float, minimum intensity for BF peaks.
        nuc_drop_ratio: float, nucleus considered dropped if post < pre * ratio.
        mem_bright_ratio: float, membrane considered bright if local > background * ratio.
        window: int, half-window around each cell for local intensity checks.

    Returns:
        dict with:
            n_converted: int, number of verified converted cells.
            positions: (N, 2) array of (row, col) positions of converted cells.
            n_candidates: int, total BF cells inside ROI.
            details: list of dicts with per-cell verification info.
    """
    bf = np.asarray(bf_image, dtype=float)
    nuc_pre = np.asarray(nuc_pre, dtype=float)
    nuc_post = np.asarray(nuc_post, dtype=float)
    mem_post = np.asarray(mem_post, dtype=float)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    h, w = bf.shape

    # Find cell centers in BF
    bf_smooth = ndimage.gaussian_filter(bf, sigma=bf_sigma)
    bf_peaks = peak_local_max(
        bf_smooth, min_distance=bf_min_distance,
        threshold_abs=bf_threshold_abs,
    )

    # Filter to ROI
    candidates = [(y, x) for y, x in bf_peaks if roi_mask[y, x]]
    mem_bg = float(mem_post[~roi_mask].mean()) if (~roi_mask).sum() > 0 else 1.0

    converted = []
    details = []
    for y, x in candidates:
        y0 = max(0, y - window)
        y1 = min(h, y + window + 1)
        x0 = max(0, x - window)
        x1 = min(w, x + window + 1)

        nuc_pre_local = float(nuc_pre[y0:y1, x0:x1].mean())
        nuc_post_local = float(nuc_post[y0:y1, x0:x1].mean())
        mem_post_local = float(mem_post[y0:y1, x0:x1].mean())

        nuc_dropped = nuc_post_local < nuc_pre_local * nuc_drop_ratio
        mem_bright = mem_post_local > mem_bg * mem_bright_ratio

        is_converted = nuc_dropped or mem_bright
        info = {
            'row': int(y), 'col': int(x),
            'nuc_pre': round(nuc_pre_local, 1),
            'nuc_post': round(nuc_post_local, 1),
            'mem_post': round(mem_post_local, 1),
            'nuc_dropped': nuc_dropped,
            'mem_bright': mem_bright,
            'converted': is_converted,
        }
        details.append(info)
        if is_converted:
            converted.append((y, x))

    positions = np.array(converted, dtype=int) if converted else np.empty((0, 2), dtype=int)

    return {
        'n_converted': len(converted),
        'positions': positions,
        'n_candidates': len(candidates),
        'details': details,
    }


def measure_activation(stack, roi_mask, reference_mask=None):
    """Measure signal in activated ROI over time.

    Args:
        stack: 3D array (T, H, W), timelapse after activation.
        roi_mask: 2D bool array, photoactivated/converted region.
        reference_mask: 2D bool array or None, unactivated reference region.

    Returns:
        dict with:
            roi_intensity: 1D array, mean intensity in ROI per frame.
            reference_intensity: 1D array or None, reference signal.
            corrected: 1D array, bleach-corrected if reference given.
            n_frames: int.
    """
    stack = np.asarray(stack, dtype=float)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    n = stack.shape[0]
    roi_int = np.array([float(stack[t][roi_mask].mean()) for t in range(n)])

    if reference_mask is not None:
        reference_mask = np.asarray(reference_mask, dtype=bool)
        ref_int = np.array([float(stack[t][reference_mask].mean()) for t in range(n)])
        ref0 = ref_int[0] if ref_int[0] > 0 else 1.0
        correction = ref0 / np.where(ref_int > 0, ref_int, 1.0)
        corrected = roi_int * correction
    else:
        ref_int = None
        corrected = roi_int.copy()

    return {
        'roi_intensity': roi_int,
        'reference_intensity': ref_int,
        'corrected': corrected,
        'n_frames': n,
    }


def signal_spread(stack, center_mask, timepoints=None, radial_bins=5):
    """Quantify spatial spread of signal from activation center.

    Measures how activated signal diffuses outward by computing
    mean intensity in concentric annuli around the activation zone.

    Args:
        stack: 3D array (T, H, W).
        center_mask: 2D bool array, initial activation zone.
        timepoints: 1D array or None (uses frame indices).
        radial_bins: int, number of concentric rings to measure.

    Returns:
        dict with:
            profiles: 2D array (T, radial_bins), radial intensity profiles.
            radii: 1D array, mean radius of each bin.
            spread_rate: float, rate of signal spread (px/frame).
            timepoints: 1D array.
    """
    stack = np.asarray(stack, dtype=float)
    center_mask = np.asarray(center_mask, dtype=bool)
    n_frames = stack.shape[0]

    if timepoints is None:
        timepoints = np.arange(n_frames, dtype=float)

    # Compute distance from center of activation
    cy, cx = ndimage.center_of_mass(center_mask)
    h, w = stack.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # Compute max radius of center mask for ring sizing
    max_center_r = float(dist[center_mask].max()) if center_mask.sum() > 0 else 10
    max_radius = max_center_r * (radial_bins + 1)
    edges = np.linspace(0, max_radius, radial_bins + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])

    profiles = np.zeros((n_frames, radial_bins))
    for t in range(n_frames):
        for b in range(radial_bins):
            ring = (dist >= edges[b]) & (dist < edges[b + 1])
            if ring.sum() > 0:
                profiles[t, b] = float(stack[t][ring].mean())

    # Estimate spread rate: find the radius where signal falls to 50%
    half_max_radii = []
    for t in range(n_frames):
        peak = profiles[t].max()
        if peak > 0:
            above = profiles[t] >= peak * 0.5
            if above.any():
                last_above = np.where(above)[0][-1]
                half_max_radii.append(float(radii[last_above]))

    if len(half_max_radii) >= 2:
        spread_rate = (half_max_radii[-1] - half_max_radii[0]) / max(n_frames - 1, 1)
    else:
        spread_rate = 0.0

    return {
        'profiles': profiles,
        'radii': radii,
        'spread_rate': round(spread_rate, 4),
        'timepoints': timepoints,
    }


def half_life(timepoints, intensity, method='exponential'):
    """Estimate signal half-life from decay curve.

    Args:
        timepoints: 1D array of time values.
        intensity: 1D array of intensity values (decaying).
        method: str, 'exponential' for fit, 'interpolation' for direct.

    Returns:
        dict with:
            half_life: float, time for signal to drop to 50%.
            decay_rate: float, exponential decay rate constant.
            r_squared: float, goodness of fit.
            fitted: 1D array, fitted curve.
    """
    t = np.asarray(timepoints, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if len(t) < 3 or y[0] <= 0:
        return {
            'half_life': 0.0, 'decay_rate': 0.0,
            'r_squared': 0.0, 'fitted': y.copy(),
        }

    if method == 'interpolation':
        half_val = y[0] / 2
        below = np.where(y <= half_val)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                # Linear interpolation
                t_half = t[idx-1] + (half_val - y[idx-1]) / (y[idx] - y[idx-1]) * (t[idx] - t[idx-1])
            else:
                t_half = t[0]
        else:
            t_half = t[-1]  # Never reached half

        return {
            'half_life': round(float(t_half - t[0]), 4),
            'decay_rate': 0.0,
            'r_squared': 0.0,
            'fitted': y.copy(),
        }

    # Exponential fit: y = A * exp(-k * t) + C
    y0 = float(y[0])
    y_end = float(y[-1])

    def model(t_val, A, k, C):
        return A * np.exp(-k * t_val) + C

    try:
        popt, _ = optimize.curve_fit(
            model, t - t[0], y,
            p0=[y0 - y_end, 0.1, y_end],
            bounds=([0, 1e-8, 0], [y0 * 2, 100, y0]),
            maxfev=5000,
        )
        A, k, C = popt
        fitted = model(t - t[0], *popt)
        ss_res = np.sum((y - fitted)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        t_half = np.log(2) / k if k > 0 else float('inf')

    except (RuntimeError, ValueError):
        k = 0.0
        t_half = 0.0
        r2 = 0.0
        fitted = y.copy()

    return {
        'half_life': round(float(t_half), 4),
        'decay_rate': round(float(k), 6),
        'r_squared': round(float(max(r2, 0.0)), 4),
        'fitted': fitted,
    }


def transport_rate(stack, roi_mask, direction='right', strip_width=10):
    """Measure directional transport from activation zone.

    Tracks how signal moves in a specific direction by measuring
    intensity in strips perpendicular to the transport direction.

    Args:
        stack: 3D array (T, H, W).
        roi_mask: 2D bool array, activation zone.
        direction: str, 'right', 'left', 'up', 'down'.
        strip_width: int, width of measurement strips in pixels.

    Returns:
        dict with:
            wavefront_positions: 1D array, position of signal front per frame.
            velocity: float, mean transport velocity (px/frame).
            peak_positions: 1D array, position of peak signal per frame.
    """
    stack = np.asarray(stack, dtype=float)
    n_frames = stack.shape[0]

    # Get center of ROI
    cy, cx = ndimage.center_of_mass(roi_mask)

    wavefronts = np.zeros(n_frames)
    peaks = np.zeros(n_frames)

    for t in range(n_frames):
        img = stack[t]
        threshold = img.mean() + 2 * img.std()

        if direction in ('right', 'left'):
            # Profile along x-axis
            profile = img[int(cy)-strip_width//2:int(cy)+strip_width//2, :].mean(axis=0)
        else:
            # Profile along y-axis
            profile = img[:, int(cx)-strip_width//2:int(cx)+strip_width//2].mean(axis=1)

        if direction in ('left', 'up'):
            profile = profile[::-1]

        # Find wavefront (furthest pixel above threshold)
        above = profile > threshold
        if above.any():
            wavefronts[t] = float(np.where(above)[0][-1])
            peaks[t] = float(np.argmax(profile))
        else:
            wavefronts[t] = 0
            peaks[t] = 0

    # Compute velocity from wavefront advance
    if n_frames >= 2:
        velocity = float(wavefronts[-1] - wavefronts[0]) / (n_frames - 1)
    else:
        velocity = 0.0

    return {
        'wavefront_positions': wavefronts,
        'velocity': round(velocity, 4),
        'peak_positions': peaks,
    }


def activation_efficiency(pre_image, post_image, roi_mask):
    """Compute photoactivation/conversion efficiency.

    efficiency = (post_ROI - pre_ROI) / pre_ROI for activation
    efficiency = post_ROI / pre_ROI for conversion (different channel)

    Args:
        pre_image: 2D array, image before activation.
        post_image: 2D array, image right after activation.
        roi_mask: 2D bool array, activated region.

    Returns:
        dict with:
            efficiency: float, fold change in ROI.
            pre_intensity: float, mean ROI intensity before.
            post_intensity: float, mean ROI intensity after.
            contrast_ratio: float, post_ROI / post_background.
    """
    pre = np.asarray(pre_image, dtype=float)
    post = np.asarray(post_image, dtype=float)
    roi = np.asarray(roi_mask, dtype=bool)
    bg_mask = ~roi

    pre_roi = float(pre[roi].mean()) if roi.sum() > 0 else 0.0
    post_roi = float(post[roi].mean()) if roi.sum() > 0 else 0.0
    post_bg = float(post[bg_mask].mean()) if bg_mask.sum() > 0 else 0.0

    fold = post_roi / pre_roi if pre_roi > 0 else 0.0
    contrast = post_roi / post_bg if post_bg > 0 else 0.0

    return {
        'efficiency': round(fold, 4),
        'pre_intensity': round(pre_roi, 4),
        'post_intensity': round(post_roi, 4),
        'contrast_ratio': round(contrast, 4),
    }
