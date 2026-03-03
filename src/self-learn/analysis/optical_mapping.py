"""Optical mapping analysis for excitable tissue (cardiac, neural).

Analyzes wave propagation in calcium/voltage imaging data:
frequency mapping, activation timing, conduction velocity,
and pacemaker/source detection.

Functions:
    frequency_map       -- Per-pixel dominant frequency from timelapse
    phase_map           -- Phase at a specific frequency (activation timing)
    conduction_velocity -- Wavefront speed from phase gradient
    detect_pacemaker    -- Find wave origin from activation timing
    activation_map      -- Per-pixel activation time for a single beat

Real microscope note:
    Simple frame-to-frame analysis is vulnerable to noise and motion
    artifacts in real optical recordings. Phase gradient calculations
    can be corrupted by camera noise, focus drift, and tissue movement.
    For robust real-world analysis:
    - Apply bandpass filtering (e.g., 2–4 Hz for cardiac) before FFT
    - Perform motion correction on the stack before activation analysis
    - Use cross-correlation or optical flow to detect/correct xy shift
    - Smooth with median filter to reduce salt-and-pepper noise
    - Validate against known reference pacing protocols
"""

import numpy as np
from scipy import ndimage


def frequency_map(stack, dt=1.0, block_size=16, min_freq=None, max_freq=None):
    """Compute spatial map of dominant oscillation frequency.

    Args:
        stack: 3D array (n_frames, height, width).
        dt: Time between frames (seconds).
        block_size: Spatial block size for averaging.
        min_freq: Minimum frequency to consider (Hz). Excludes DC.
        max_freq: Maximum frequency to consider (Hz).

    Returns:
        dict with:
            freq_map: 2D array of dominant frequency per block (Hz).
            power_map: 2D array of power at dominant frequency.
            mode_freq: Most common dominant frequency (Hz).
            block_size: Block size used.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n_frames, h, w = stack.shape
    freqs = np.fft.rfftfreq(n_frames, d=dt)

    h_b, w_b = h // block_size, w // block_size
    fmap = np.zeros((h_b, w_b))
    pmap = np.zeros((h_b, w_b))

    # Frequency bounds
    min_bin = 1  # skip DC
    if min_freq is not None:
        min_bin = max(min_bin, int(np.floor(min_freq * n_frames * dt)))
    max_bin = len(freqs)
    if max_freq is not None:
        max_bin = min(max_bin, int(np.ceil(max_freq * n_frames * dt)) + 1)

    for bi in range(h_b):
        for bj in range(w_b):
            region = stack[:, bi*block_size:(bi+1)*block_size,
                          bj*block_size:(bj+1)*block_size]
            trace = region.mean(axis=(1, 2))
            trace = trace - np.mean(trace)
            spec = np.abs(np.fft.rfft(trace))**2
            spec[:min_bin] = 0
            if max_bin < len(spec):
                spec[max_bin:] = 0
            dom_idx = np.argmax(spec)
            fmap[bi, bj] = freqs[dom_idx]
            pmap[bi, bj] = spec[dom_idx]

    # Mode frequency (most common dominant)
    rounded = np.round(fmap, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    mode_freq = float(unique[np.argmax(counts)])

    return {
        'freq_map': fmap,
        'power_map': pmap,
        'mode_freq': mode_freq,
        'block_size': block_size,
    }


def phase_map(stack, target_freq, dt=1.0, block_size=16):
    """Compute spatial phase map at a specific frequency.

    Phase represents relative activation timing — earlier phase means
    the wavefront arrives first at that location.

    Args:
        stack: 3D array (n_frames, height, width).
        target_freq: Target frequency in Hz.
        dt: Time between frames.
        block_size: Spatial block size.

    Returns:
        dict with:
            phase: 2D array of phase values (radians).
            power: 2D array of spectral power at target frequency.
            block_size: Block size used.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n_frames, h, w = stack.shape
    freqs = np.fft.rfftfreq(n_frames, d=dt)

    # Find nearest frequency bin
    target_bin = int(np.argmin(np.abs(freqs - target_freq)))

    h_b, w_b = h // block_size, w // block_size
    ph = np.zeros((h_b, w_b))
    pw = np.zeros((h_b, w_b))

    for bi in range(h_b):
        for bj in range(w_b):
            region = stack[:, bi*block_size:(bi+1)*block_size,
                          bj*block_size:(bj+1)*block_size]
            trace = region.mean(axis=(1, 2))
            trace = trace - np.mean(trace)
            fft_val = np.fft.rfft(trace)[target_bin]
            ph[bi, bj] = np.angle(fft_val)
            pw[bi, bj] = np.abs(fft_val)**2

    return {
        'phase': ph,
        'power': pw,
        'block_size': block_size,
    }


def conduction_velocity(phase_data, block_size=16, power_threshold_pct=25,
                        max_grad=0.1):
    """Measure wave conduction velocity from phase gradient.

    Uses the relationship: velocity = omega / |grad(phase)|
    where omega is the angular frequency.

    Args:
        phase_data: dict from phase_map() with 'phase', 'power'.
        block_size: Spatial block size (pixels per block).
        power_threshold_pct: Min power percentile for valid blocks.
        max_grad: Maximum phase gradient (rad/pixel) to include.

    Returns:
        dict with:
            velocity_median: Median conduction velocity (px/frame).
            velocity_mean: Mean conduction velocity.
            velocity_map: 2D map of local velocity.
            direction_map: 2D map of propagation direction (radians).
            n_valid: Number of valid measurement blocks.
    """
    ph = phase_data['phase']
    pw = phase_data['power']

    # Phase gradient
    grad_y = np.gradient(ph, block_size, axis=0)
    grad_x = np.gradient(ph, block_size, axis=1)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    # Valid blocks: sufficient power, reasonable gradient
    valid = (grad_mag > 0.001) & (grad_mag < max_grad) & (
        pw > np.percentile(pw, power_threshold_pct))

    # Velocity = omega / |grad_phase|
    # We don't know omega here, so return in units that depend on the
    # calling context. For proper units, caller should multiply by omega.
    safe_grad = np.where(grad_mag > 1e-6, grad_mag, 1.0)
    velocity_map = np.where(grad_mag > 1e-6, 1.0 / safe_grad, 0.0)

    if valid.any():
        velocities = velocity_map[valid]
        med_vel = float(np.median(velocities))
        mean_vel = float(np.mean(velocities))
    else:
        med_vel = 0.0
        mean_vel = 0.0

    return {
        'velocity_median': round(med_vel, 2),
        'velocity_mean': round(mean_vel, 2),
        'velocity_map': velocity_map,
        'direction_map': grad_dir,
        'grad_magnitude': grad_mag,
        'n_valid': int(valid.sum()),
    }


def detect_pacemaker(stack, dt=1.0, corner_size=64, min_distance=3):
    """Detect pacemaker location from earliest activation timing.

    Compares calcium peak timing at four corners of the field
    to determine where the wavefront originates.

    Args:
        stack: 3D array (n_frames, height, width). Should be smoothed.
        dt: Time between frames.
        corner_size: Size of corner regions to analyze (pixels).
        min_distance: Minimum distance between detected peaks.

    Returns:
        dict with:
            source_quadrant: Name of the quadrant with earliest peaks.
            peak_times: Dict of quadrant → first peak frame.
            mean_intervals: Dict of quadrant → mean beat interval.
            beat_rate_hz: Dominant beat rate in Hz.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n_frames, h, w = stack.shape
    cs = corner_size

    corners = {
        'top-left': stack[:, :cs, :cs],
        'top-right': stack[:, :cs, w-cs:],
        'bottom-left': stack[:, h-cs:, :cs],
        'bottom-right': stack[:, h-cs:, w-cs:],
    }

    peak_times = {}
    mean_intervals = {}
    earliest = None
    earliest_frame = n_frames

    for name, region in corners.items():
        trace = region.mean(axis=(1, 2))
        # Detect peaks
        peaks = _detect_peaks(trace, min_distance=min_distance)
        if len(peaks) > 0:
            peak_times[name] = int(peaks[0])
            if len(peaks) > 1:
                mean_intervals[name] = float(np.mean(np.diff(peaks)))
            if peaks[0] < earliest_frame:
                earliest_frame = int(peaks[0])
                earliest = name

    # Beat rate from mean interval
    all_intervals = [v for v in mean_intervals.values() if v > 0]
    beat_rate = 1.0 / (np.mean(all_intervals) * dt) if all_intervals else 0.0

    return {
        'source_quadrant': earliest or 'unknown',
        'peak_times': peak_times,
        'mean_intervals': mean_intervals,
        'beat_rate_hz': round(beat_rate, 3),
    }


def activation_map(stack, beat_frame, search_window=4, threshold_frac=0.3):
    """Compute per-pixel activation time for a single beat.

    Finds when each pixel first crosses its activation threshold
    near a specified beat (peak) frame.

    Args:
        stack: 3D array (n_frames, height, width). Should be smoothed.
        beat_frame: Frame index of the beat peak.
        search_window: Number of frames before peak to search.
        threshold_frac: Fraction of (peak-baseline) for activation.

    Returns:
        dict with:
            activation_time: 2D array of activation frame (NaN if not activated).
            relative_time: Activation time relative to beat_frame.
            earliest_pixel: (row, col) of earliest activation.
    """
    stack = np.asarray(stack, dtype=np.float64)
    n_frames, h, w = stack.shape

    pixel_mean = stack.mean(axis=0)
    pixel_std = stack.std(axis=0)
    threshold = pixel_mean + threshold_frac * pixel_std

    act = np.full((h, w), np.nan)
    search_start = max(0, beat_frame - search_window)
    search_end = min(n_frames, beat_frame + 1)

    for t in range(search_start, search_end):
        newly_active = (stack[t] > threshold) & np.isnan(act)
        if t > search_start:
            was_below = stack[t - 1] <= threshold
            newly_active = newly_active & was_below
        act[newly_active] = t

    relative = act - beat_frame
    valid = ~np.isnan(act)
    if valid.any():
        earliest_idx = np.unravel_index(np.nanargmin(act), act.shape)
    else:
        earliest_idx = (0, 0)

    return {
        'activation_time': act,
        'relative_time': relative,
        'earliest_pixel': earliest_idx,
    }


def _detect_peaks(signal, min_distance=3):
    """Simple peak detection for internal use."""
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 3:
        return np.array([], dtype=int)

    peaks = []
    for i in range(1, n - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if min_distance <= 1 or not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)

    return np.array(peaks, dtype=int)
