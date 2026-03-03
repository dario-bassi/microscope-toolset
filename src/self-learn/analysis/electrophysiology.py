"""Electrophysiology signal analysis.

Analyzes action potentials, calcium transients, and other
periodic biological signals from timelapse or electrode data.

Functions:
    detect_action_potentials -- Find APs in a voltage/fluorescence trace
    firing_rate              -- Compute instantaneous and mean firing rate
    interspike_intervals     -- Measure ISI distribution
    action_potential_shape   -- Characterize AP waveform morphology
    burst_detection          -- Detect burst firing patterns
"""

import numpy as np
from scipy import signal, ndimage


def detect_action_potentials(trace, dt=1.0, min_height=None,
                             min_distance=None, threshold_sigma=3.0):
    """Detect action potentials / calcium transients in a trace.

    Uses peak detection with adaptive thresholding based on
    signal noise level.

    Args:
        trace: 1D array, signal trace (voltage, fluorescence, etc.).
        dt: float, time step between samples (seconds).
        min_height: float or None, minimum peak height above baseline.
            If None, uses baseline + threshold_sigma * noise_std.
        min_distance: int or None, minimum samples between peaks.
            If None, auto-detects from trace length.
        threshold_sigma: float, number of noise std deviations above
            baseline for automatic threshold.

    Returns:
        dict with:
            peak_indices: array of int, indices of detected peaks.
            peak_times: array of float, times of peaks.
            peak_values: array of float, values at peaks.
            n_peaks: int.
            threshold: float, threshold used for detection.
    """
    trace = np.asarray(trace, dtype=float)
    n = len(trace)

    if n < 5:
        return {
            'peak_indices': np.array([], dtype=int),
            'peak_times': np.array([]),
            'peak_values': np.array([]),
            'n_peaks': 0,
            'threshold': 0.0,
        }

    # Estimate baseline and noise from lower percentiles
    baseline = np.percentile(trace, 25)
    below_q75 = trace[trace <= np.percentile(trace, 75)]
    noise_std = float(np.std(below_q75)) if len(below_q75) > 1 else 0.0
    if noise_std == 0:
        noise_std = float(np.std(trace)) * 0.5

    if min_height is None:
        min_height = baseline + threshold_sigma * noise_std

    if min_distance is None:
        min_distance = max(3, n // 50)

    peaks, properties = signal.find_peaks(
        trace, height=min_height, distance=min_distance)

    peak_times = peaks.astype(float) * dt
    peak_values = trace[peaks]

    return {
        'peak_indices': peaks,
        'peak_times': peak_times,
        'peak_values': peak_values,
        'n_peaks': len(peaks),
        'threshold': round(float(min_height), 4),
    }


def firing_rate(peak_times, window=None, method='mean'):
    """Compute firing rate from detected peak times.

    Args:
        peak_times: array-like, times of detected peaks (seconds).
        window: float or None, time window for rate calculation.
            If None, uses total recording duration.
        method: str, 'mean' for average rate, 'instantaneous' for
            per-interval rates.

    Returns:
        dict with:
            mean_rate_hz: float, mean firing rate in Hz.
            mean_rate_bpm: float, rate in beats/min.
            instantaneous_rates: array of float, per-interval rates.
            n_peaks: int.
            duration: float, recording duration.
    """
    times = np.asarray(peak_times, dtype=float)

    if len(times) < 2:
        return {
            'mean_rate_hz': 0.0,
            'mean_rate_bpm': 0.0,
            'instantaneous_rates': np.array([]),
            'n_peaks': len(times),
            'duration': 0.0,
        }

    duration = times[-1] - times[0]
    if duration <= 0:
        return {
            'mean_rate_hz': 0.0,
            'mean_rate_bpm': 0.0,
            'instantaneous_rates': np.array([]),
            'n_peaks': len(times),
            'duration': 0.0,
        }

    # Mean rate
    n_intervals = len(times) - 1
    mean_rate = n_intervals / duration

    # Instantaneous rates
    intervals = np.diff(times)
    valid = intervals > 0
    inst_rates = np.zeros_like(intervals)
    inst_rates[valid] = 1.0 / intervals[valid]

    return {
        'mean_rate_hz': round(float(mean_rate), 4),
        'mean_rate_bpm': round(float(mean_rate * 60), 2),
        'instantaneous_rates': inst_rates,
        'n_peaks': len(times),
        'duration': round(float(duration), 4),
    }


def interspike_intervals(peak_times):
    """Compute inter-spike interval distribution.

    Args:
        peak_times: array-like, times of detected peaks.

    Returns:
        dict with:
            intervals: array of float, ISIs.
            mean_isi: float, mean ISI.
            std_isi: float, ISI standard deviation.
            cv_isi: float, coefficient of variation (regularity).
            min_isi: float.
            max_isi: float.
            n_intervals: int.
    """
    times = np.asarray(peak_times, dtype=float)

    if len(times) < 2:
        return {
            'intervals': np.array([]),
            'mean_isi': 0.0,
            'std_isi': 0.0,
            'cv_isi': 0.0,
            'min_isi': 0.0,
            'max_isi': 0.0,
            'n_intervals': 0,
        }

    intervals = np.diff(times)
    mean_isi = float(intervals.mean())
    std_isi = float(intervals.std())

    return {
        'intervals': intervals,
        'mean_isi': round(mean_isi, 6),
        'std_isi': round(std_isi, 6),
        'cv_isi': round(std_isi / mean_isi, 4) if mean_isi > 0 else 0.0,
        'min_isi': round(float(intervals.min()), 6),
        'max_isi': round(float(intervals.max()), 6),
        'n_intervals': len(intervals),
    }


def action_potential_shape(trace, peak_indices, dt=1.0,
                           window_before=10, window_after=20):
    """Characterize action potential waveform morphology.

    Extracts and averages AP waveforms around detected peaks.

    Args:
        trace: 1D array, full signal trace.
        peak_indices: array of int, indices of detected peaks.
        dt: float, time step (seconds).
        window_before: int, samples before peak to include.
        window_after: int, samples after peak to include.

    Returns:
        dict with:
            mean_waveform: 1D array, average AP waveform.
            waveform_time: 1D array, time axis for waveform.
            amplitude: float, peak-to-trough amplitude.
            rise_time: float, time from 10% to 90% of amplitude.
            decay_time: float, time from 90% to 10% after peak.
            half_width: float, duration at 50% amplitude.
            n_averaged: int.
    """
    trace = np.asarray(trace, dtype=float)
    peak_indices = np.asarray(peak_indices, dtype=int)
    n = len(trace)

    waveforms = []
    for pi in peak_indices:
        start = pi - window_before
        end = pi + window_after
        if start >= 0 and end < n:
            waveforms.append(trace[start:end])

    if not waveforms:
        wl = window_before + window_after
        return {
            'mean_waveform': np.zeros(wl),
            'waveform_time': np.arange(wl) * dt - window_before * dt,
            'amplitude': 0.0,
            'rise_time': 0.0,
            'decay_time': 0.0,
            'half_width': 0.0,
            'n_averaged': 0,
        }

    waveforms = np.array(waveforms)
    mean_wf = waveforms.mean(axis=0)
    wf_time = (np.arange(len(mean_wf)) - window_before) * dt

    # Amplitude
    peak_val = mean_wf[window_before]
    baseline_val = min(mean_wf[:window_before].min(),
                       mean_wf[-window_after // 2:].min())
    amplitude = peak_val - baseline_val

    # Rise/decay times
    if amplitude > 0:
        threshold_10 = baseline_val + 0.1 * amplitude
        threshold_90 = baseline_val + 0.9 * amplitude
        threshold_50 = baseline_val + 0.5 * amplitude

        # Rise: before peak
        rise_portion = mean_wf[:window_before + 1]
        t10_idx = np.where(rise_portion >= threshold_10)[0]
        t90_idx = np.where(rise_portion >= threshold_90)[0]
        rise_time = (t90_idx[0] - t10_idx[0]) * dt if len(t10_idx) > 0 and len(t90_idx) > 0 else 0

        # Decay: after peak
        decay_portion = mean_wf[window_before:]
        d90_idx = np.where(decay_portion <= threshold_90)[0]
        d10_idx = np.where(decay_portion <= threshold_10)[0]
        decay_time = (d10_idx[0] - d90_idx[0]) * dt if len(d90_idx) > 0 and len(d10_idx) > 0 else 0

        # Half-width: full width at half-maximum
        above_50 = mean_wf >= threshold_50
        first_above = np.where(above_50)[0]
        if len(first_above) >= 2:
            half_width = (first_above[-1] - first_above[0]) * dt
        else:
            half_width = 0.0
    else:
        rise_time = 0.0
        decay_time = 0.0
        half_width = 0.0

    return {
        'mean_waveform': mean_wf,
        'waveform_time': wf_time,
        'amplitude': round(float(amplitude), 4),
        'rise_time': round(float(rise_time), 6),
        'decay_time': round(float(decay_time), 6),
        'half_width': round(float(half_width), 6),
        'n_averaged': len(waveforms),
    }


def burst_detection(peak_times, max_isi=0.1, min_spikes=3):
    """Detect burst firing patterns.

    A burst is a cluster of spikes with short inter-spike intervals.

    Args:
        peak_times: array-like, times of detected peaks.
        max_isi: float, maximum inter-spike interval within a burst.
        min_spikes: int, minimum number of spikes to form a burst.

    Returns:
        dict with:
            n_bursts: int.
            burst_starts: list of float, start time of each burst.
            burst_durations: list of float, duration of each burst.
            spikes_per_burst: list of int.
            burst_fraction: float, fraction of spikes in bursts.
            interburst_intervals: list of float.
    """
    times = np.asarray(peak_times, dtype=float)

    if len(times) < min_spikes:
        return {
            'n_bursts': 0,
            'burst_starts': [],
            'burst_durations': [],
            'spikes_per_burst': [],
            'burst_fraction': 0.0,
            'interburst_intervals': [],
        }

    # Find burst boundaries
    intervals = np.diff(times)
    in_burst = intervals <= max_isi

    bursts = []
    current_burst = [0]  # spike indices

    for i, is_close in enumerate(in_burst):
        if is_close:
            current_burst.append(i + 1)
        else:
            if len(current_burst) >= min_spikes:
                bursts.append(current_burst)
            current_burst = [i + 1]

    # Check last burst
    if len(current_burst) >= min_spikes:
        bursts.append(current_burst)

    burst_starts = [float(times[b[0]]) for b in bursts]
    burst_durations = [float(times[b[-1]] - times[b[0]]) for b in bursts]
    spikes_per_burst = [len(b) for b in bursts]
    total_burst_spikes = sum(spikes_per_burst)
    burst_fraction = total_burst_spikes / len(times) if len(times) > 0 else 0

    # Interburst intervals
    ibi = []
    for i in range(1, len(bursts)):
        ibi.append(float(times[bursts[i][0]] - times[bursts[i - 1][-1]]))

    return {
        'n_bursts': len(bursts),
        'burst_starts': [round(s, 6) for s in burst_starts],
        'burst_durations': [round(d, 6) for d in burst_durations],
        'spikes_per_burst': spikes_per_burst,
        'burst_fraction': round(float(burst_fraction), 4),
        'interburst_intervals': [round(i, 6) for i in ibi],
    }
