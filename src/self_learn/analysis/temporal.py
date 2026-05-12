"""Temporal signal analysis for microscopy time series.

Provides FFT-based frequency analysis, autocorrelation for periodicity
detection, and peak detection for event timing. Essential for:
- cAMP wave period measurement (Dictyostelium)
- Calcium oscillation frequency analysis
- Heartbeat/contraction rate measurement
- Drug response temporal dynamics
"""

import numpy as np


def fft_spectrum(signal, dt=1.0):
    """Compute FFT power spectrum of a 1D time series.

    Args:
        signal: 1D array of values over time.
        dt: Time step between samples (seconds or frames).

    Returns:
        dict with:
            frequencies: Array of frequency values (Hz or 1/frame).
            power: Array of spectral power values.
            dominant_freq: Frequency with highest power (excluding DC).
            dominant_period: 1/dominant_freq (in same units as dt).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 3:
        return {
            'frequencies': np.array([]),
            'power': np.array([]),
            'dominant_freq': 0.0,
            'dominant_period': float('inf'),
        }

    # Detrend (remove linear trend)
    t = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(t, signal, 1)
    detrended = signal - np.polyval(coeffs, t)

    # FFT
    fft_vals = np.fft.rfft(detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)

    # Find dominant frequency (skip DC component at index 0)
    if len(power) > 1:
        dom_idx = 1 + np.argmax(power[1:])
        dominant_freq = float(freqs[dom_idx])
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else float('inf')
    else:
        dominant_freq = 0.0
        dominant_period = float('inf')

    return {
        'frequencies': freqs,
        'power': power,
        'dominant_freq': dominant_freq,
        'dominant_period': dominant_period,
    }


def autocorrelation(signal, max_lag=None):
    """Compute normalized autocorrelation of a 1D signal.

    Useful for detecting periodicity. The first peak after lag=0
    gives the signal period.

    Args:
        signal: 1D array.
        max_lag: Maximum lag to compute (default: len/2).

    Returns:
        dict with:
            lags: Array of lag values.
            acf: Normalized autocorrelation values (-1 to 1).
            first_peak_lag: Lag of first autocorrelation peak (period estimate).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < 4:
        return {
            'lags': np.array([]),
            'acf': np.array([]),
            'first_peak_lag': 0,
        }

    if max_lag is None:
        max_lag = n // 2

    # Detrend and normalize
    mean = signal.mean()
    centered = signal - mean
    var = np.sum(centered ** 2)
    if var < 1e-10:
        return {
            'lags': np.arange(max_lag + 1),
            'acf': np.zeros(max_lag + 1),
            'first_peak_lag': 0,
        }

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.sum(centered[:n - lag] * centered[lag:]) / var

    # Find first peak (local max after initial decay)
    first_peak_lag = 0
    # Skip initial decay (look for first upward trend then peak)
    in_trough = False
    for i in range(1, len(acf) - 1):
        if acf[i] < acf[i - 1]:
            in_trough = True
        if in_trough and acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
            first_peak_lag = i
            break

    return {
        'lags': np.arange(max_lag + 1),
        'acf': acf,
        'first_peak_lag': int(first_peak_lag),
    }


def detect_peaks(signal, min_height=None, min_distance=1):
    """Detect peaks in a 1D signal.

    Args:
        signal: 1D array.
        min_height: Minimum peak height. If None, uses mean + std.
        min_distance: Minimum distance between peaks (in samples).

    Returns:
        dict with:
            peak_indices: Array of peak positions.
            peak_values: Array of peak heights.
            n_peaks: Number of peaks found.
            mean_interval: Mean inter-peak interval.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if min_height is None:
        min_height = signal.mean() + signal.std()

    # Find local maxima
    peaks = []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if signal[i] >= min_height:
                peaks.append(i)

    # Enforce minimum distance
    if min_distance > 1 and peaks:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= min_distance:
                filtered.append(p)
        peaks = filtered

    peak_indices = np.array(peaks, dtype=int)
    peak_values = signal[peak_indices] if len(peak_indices) > 0 else np.array([])

    # Mean interval between peaks
    if len(peak_indices) > 1:
        intervals = np.diff(peak_indices)
        mean_interval = float(intervals.mean())
    else:
        mean_interval = 0.0

    return {
        'peak_indices': peak_indices,
        'peak_values': peak_values,
        'n_peaks': len(peak_indices),
        'mean_interval': mean_interval,
    }


def detrend(signal, method='linear'):
    """Remove trend from a signal.

    Args:
        signal: 1D array.
        method: 'linear' (subtract best-fit line) or 'mean' (subtract mean).

    Returns:
        Detrended signal array.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if method == 'mean':
        return signal - signal.mean()
    elif method == 'linear':
        t = np.arange(len(signal), dtype=np.float64)
        coeffs = np.polyfit(t, signal, 1)
        return signal - np.polyval(coeffs, t)
    else:
        raise ValueError(f"Unknown detrend method: {method}")


def measure_periodic_rate(signal, dt=1.0, min_freq=None, max_freq=None, unit='bpm'):
    """Cross-validated periodic rate measurement using FFT + autocorrelation + peak counting.

    Combines three independent methods to estimate the frequency of a periodic
    signal (heartbeat, contraction, oscillation) and cross-validates them.

    Args:
        signal: 1D intensity time series.
        dt: Time step between samples (seconds).
        min_freq: Minimum expected frequency (Hz). Used for peak distance.
        max_freq: Maximum expected frequency (Hz). Used for FFT range.
        unit: Output unit — 'bpm' (beats per minute), 'hz', or 'period'.

    Returns:
        dict with:
            rate: Best estimate of the periodic rate (in requested unit).
            rate_hz: Rate in Hz.
            method_agreement: Dict of per-method estimates in Hz.
            confidence: 'high' if all methods agree within 10%, else 'low'.
            n_cycles: Estimated number of complete cycles in the signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    duration = n * dt

    if n < 10:
        return {
            'rate': 0.0, 'rate_hz': 0.0,
            'method_agreement': {}, 'confidence': 'low', 'n_cycles': 0,
        }

    # Detrend
    detrended = detrend(signal)

    # --- Method 1: FFT ---
    spectrum = fft_spectrum(detrended, dt=dt)
    fft_hz = spectrum['dominant_freq']

    # If frequency bounds given, restrict FFT search
    if (min_freq is not None or max_freq is not None) and len(spectrum['frequencies']) > 1:
        freqs = spectrum['frequencies']
        power = spectrum['power'].copy()
        lo = min_freq if min_freq is not None else 0
        hi = max_freq if max_freq is not None else freqs[-1]
        mask = (freqs >= lo) & (freqs <= hi) & (freqs > 0)
        if mask.any():
            restricted_idx = np.where(mask)[0]
            best_in_range = restricted_idx[np.argmax(power[restricted_idx])]
            fft_hz = float(freqs[best_in_range])

    # --- Method 2: Autocorrelation ---
    acorr = autocorrelation(detrended)
    acorr_hz = 0.0
    if acorr['first_peak_lag'] > 0:
        acorr_hz = 1.0 / (acorr['first_peak_lag'] * dt)

    # --- Method 3: Peak counting ---
    min_dist = 1
    if fft_hz > 0:
        # Use FFT estimate to set min_distance (half the expected period)
        expected_period_samples = 1.0 / (fft_hz * dt)
        min_dist = max(1, int(expected_period_samples * 0.5))
    elif min_freq is not None:
        min_dist = max(1, int(1.0 / (min_freq * dt) * 0.5))

    peaks = detect_peaks(detrended, min_distance=min_dist)
    peak_hz = 0.0
    if peaks['n_peaks'] > 1 and peaks['mean_interval'] > 0:
        peak_hz = 1.0 / (peaks['mean_interval'] * dt)

    # --- Cross-validation ---
    estimates = {}
    if fft_hz > 0:
        estimates['fft'] = fft_hz
    if acorr_hz > 0:
        estimates['autocorrelation'] = acorr_hz
    if peak_hz > 0:
        estimates['peak_counting'] = peak_hz

    # Best estimate: when freq bounds are given, prefer FFT (the only
    # method that respects bounds). Otherwise use median of all methods.
    has_bounds = min_freq is not None or max_freq is not None
    if has_bounds and 'fft' in estimates:
        best_hz = estimates['fft']
    elif estimates:
        values = list(estimates.values())
        best_hz = float(np.median(values))
    else:
        best_hz = 0.0

    # Confidence: all methods agree within 10%
    confidence = 'low'
    if len(estimates) >= 2 and best_hz > 0:
        max_dev = max(abs(v - best_hz) / best_hz for v in estimates.values())
        confidence = 'high' if max_dev < 0.10 else 'low'

    # Convert to requested unit
    if unit == 'bpm':
        rate = best_hz * 60
    elif unit == 'hz':
        rate = best_hz
    elif unit == 'period':
        rate = 1.0 / best_hz if best_hz > 0 else float('inf')
    else:
        rate = best_hz * 60  # default to bpm

    n_cycles = best_hz * duration if best_hz > 0 else 0

    return {
        'rate': round(rate, 2),
        'rate_hz': round(best_hz, 4),
        'method_agreement': {k: round(v, 4) for k, v in estimates.items()},
        'confidence': confidence,
        'n_cycles': round(n_cycles, 1),
    }


def measure_wave_speed(radial_profiles, dt=1.0, dr=1.0):
    """Measure wave propagation speed from time-series of radial profiles.

    Tracks the position of maximum gradient (wavefront) across time.
    Useful for cAMP waves in Dictyostelium, calcium waves, etc.

    Args:
        radial_profiles: List of 1D arrays (one radial profile per timepoint).
            Each array: intensity vs radius from some center.
        dt: Time between profiles (seconds or frames).
        dr: Spatial step between radial bins (µm or pixels).

    Returns:
        dict with:
            wavefront_positions: Radius of wavefront at each timepoint.
            wave_speed: Estimated speed (distance/time units).
            wave_speeds_per_step: Per-step speed estimates.
            r_squared: R² of linear fit to wavefront position vs time.
    """
    profiles = [np.asarray(p, dtype=np.float64) for p in radial_profiles]
    n_times = len(profiles)

    if n_times < 2:
        return {
            'wavefront_positions': [],
            'wave_speed': 0.0,
            'wave_speeds_per_step': [],
            'r_squared': 0.0,
        }

    # Find wavefront position at each timepoint
    # Wavefront = location of steepest gradient in radial profile
    positions = []
    for prof in profiles:
        if len(prof) < 3:
            positions.append(0.0)
            continue
        grad = np.abs(np.gradient(prof))
        # Smooth gradient to avoid noise peaks
        if len(grad) > 5:
            kernel = np.ones(3) / 3
            grad = np.convolve(grad, kernel, mode='same')
        peak_idx = int(np.argmax(grad))
        positions.append(float(peak_idx * dr))

    positions = np.array(positions)

    # Compute per-step speeds
    speeds = []
    for i in range(1, n_times):
        dr_step = positions[i] - positions[i - 1]
        speeds.append(float(dr_step / dt))

    # Linear fit for overall speed
    t = np.arange(n_times) * dt
    if n_times >= 2 and np.std(positions) > 0:
        coeffs = np.polyfit(t, positions, 1)
        wave_speed = float(coeffs[0])
        # R² computation
        pred = np.polyval(coeffs, t)
        ss_res = np.sum((positions - pred) ** 2)
        ss_tot = np.sum((positions - positions.mean()) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)
    else:
        wave_speed = 0.0
        r_squared = 0.0

    return {
        'wavefront_positions': positions.tolist(),
        'wave_speed': round(wave_speed, 4),
        'wave_speeds_per_step': [round(s, 4) for s in speeds],
        'r_squared': round(float(r_squared), 4),
    }


def track_mean_intensity(stack, method='whole_image', threshold=None,
                         baseline_frames=5):
    """Track mean intensity across a timelapse stack.

    Three methods are available to handle the threshold-crossing
    nonlinearity that biases kinetic measurements:

    - ``whole_image``: Mean of all pixels. Simplest, avoids threshold
      artifacts. Best for half-time measurements on decaying signals.
    - ``foreground_fixed``: Mean of pixels above a *fixed* threshold
      (set from baseline or provided). Higher SNR for sparse samples
      but introduces nonlinear step-down when dim pixels cross below
      the threshold.
    - ``foreground_adaptive``: Per-frame Otsu threshold. Tracks even
      dim signal but may mask real changes (adaptive thresholds adjust
      to the signal, hiding true decrease).

    Args:
        stack: 3D array (T, H, W) or list of 2D images.
        method: One of ``'whole_image'``, ``'foreground_fixed'``,
            ``'foreground_adaptive'``.
        threshold: Fixed threshold for ``'foreground_fixed'``. If None,
            computed from baseline frames as ``mean + 0.5 * std``.
        baseline_frames: Number of initial frames used to compute the
            default fixed threshold (only used when ``threshold=None``
            and ``method='foreground_fixed'``).

    Returns:
        dict with:
            values: 1D array of mean intensities per frame.
            method: Method string used.
            threshold: Threshold used (float or None).
    """
    if isinstance(stack, list):
        stack = np.array(stack)
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[np.newaxis]

    n_frames = stack.shape[0]
    values = np.zeros(n_frames)

    if method == 'whole_image':
        for i in range(n_frames):
            values[i] = float(np.mean(stack[i]))
        return {'values': values, 'method': method, 'threshold': None}

    elif method == 'foreground_fixed':
        if threshold is None:
            bl = stack[:min(baseline_frames, n_frames)]
            threshold = float(np.mean(bl) + 0.5 * np.std(bl))
        for i in range(n_frames):
            fg = stack[i][stack[i] > threshold]
            values[i] = float(np.mean(fg)) if len(fg) > 0 else 0.0
        return {'values': values, 'method': method, 'threshold': threshold}

    elif method == 'foreground_adaptive':
        from skimage.filters import threshold_otsu
        used_thresh = None
        for i in range(n_frames):
            frame = stack[i]
            if np.std(frame) < 1e-6:
                values[i] = float(np.mean(frame))
                continue
            try:
                t = threshold_otsu(frame)
            except ValueError:
                t = float(np.mean(frame))
            fg = frame[frame > t]
            values[i] = float(np.mean(fg)) if len(fg) > 0 else 0.0
            used_thresh = t
        return {'values': values, 'method': method, 'threshold': used_thresh}

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Use 'whole_image', 'foreground_fixed', or 'foreground_adaptive'."
        )


def measure_response_time(signal, baseline_frames=5, threshold_pct=50, dt=1.0):
    """Measure response time from a step-change experiment.

    Detects when a signal crosses a threshold after a perturbation.
    Useful for drug response timing, temperature shift effects, etc.

    Args:
        signal: 1D time series.
        baseline_frames: Number of initial frames to use as baseline.
        threshold_pct: Percentage of total change to define response time
            (e.g., 50 = time to reach 50% of final change).
        dt: Time between frames.

    Returns:
        dict with:
            baseline_mean: Mean of baseline period.
            final_mean: Mean of last baseline_frames frames (steady state).
            total_change: final_mean - baseline_mean.
            response_frame: Frame index where threshold is crossed.
            response_time: response_frame * dt.
            half_time: Time to reach 50% of total change (always computed).
            direction: 'increase' or 'decrease'.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if n < baseline_frames + 2:
        return {
            'baseline_mean': 0.0, 'final_mean': 0.0,
            'total_change': 0.0, 'response_frame': 0,
            'response_time': 0.0, 'half_time': 0.0,
            'direction': 'none',
        }

    baseline_mean = float(signal[:baseline_frames].mean())
    final_mean = float(signal[-baseline_frames:].mean())
    total_change = final_mean - baseline_mean
    direction = 'increase' if total_change > 0 else 'decrease'

    if abs(total_change) < 1e-10:
        return {
            'baseline_mean': baseline_mean, 'final_mean': final_mean,
            'total_change': 0.0, 'response_frame': 0,
            'response_time': 0.0, 'half_time': 0.0,
            'direction': 'none',
        }

    # Find frame where signal crosses threshold_pct of total change
    target = baseline_mean + total_change * threshold_pct / 100.0
    response_frame = 0
    for i in range(baseline_frames, n):
        if total_change > 0 and signal[i] >= target:
            response_frame = i
            break
        elif total_change < 0 and signal[i] <= target:
            response_frame = i
            break

    # Always compute half-time (50% crossing)
    half_target = baseline_mean + total_change * 0.5
    half_frame = 0
    for i in range(baseline_frames, n):
        if total_change > 0 and signal[i] >= half_target:
            half_frame = i
            break
        elif total_change < 0 and signal[i] <= half_target:
            half_frame = i
            break

    return {
        'baseline_mean': round(baseline_mean, 4),
        'final_mean': round(final_mean, 4),
        'total_change': round(total_change, 4),
        'response_frame': response_frame,
        'response_time': round(response_frame * dt, 4),
        'half_time': round(half_frame * dt, 4),
        'direction': direction,
    }
