"""Calcium imaging utilities: ROI trace extraction, ΔF/F, transient detection.

Extracted from intensity.py for modularity. All functions remain importable
from intensity.py and analysis/__init__.py for backward compatibility.
"""

import numpy as np


def extract_roi_traces(frames, roi_masks):
    """Extract mean intensity time series for multiple ROIs.

    Args:
        frames: List/array of 2D images (timelapse).
        roi_masks: Dict mapping label -> 2D boolean mask, or list of masks.

    Returns:
        Dict mapping label -> 1D numpy array of mean intensities per frame.
    """
    if isinstance(roi_masks, list):
        roi_masks = {i: m for i, m in enumerate(roi_masks)}

    traces = {label: [] for label in roi_masks}
    for frame in frames:
        img = np.asarray(frame, dtype=np.float64)
        for label, mask in roi_masks.items():
            pixels = img[mask]
            traces[label].append(float(pixels.mean()) if len(pixels) > 0 else 0.0)

    return {label: np.array(vals) for label, vals in traces.items()}


def compute_dff(traces, baseline_indices=None, n_baseline=None):
    """Compute ΔF/F₀ for calcium imaging traces.

    ΔF/F₀ = (F - F₀) / F₀, where F₀ is the mean of baseline frames.

    Args:
        traces: Dict mapping label -> 1D array (from extract_roi_traces),
            or a single 1D array.
        baseline_indices: Explicit indices for baseline frames. If None,
            uses first n_baseline frames.
        n_baseline: Number of initial frames to use as baseline.
            Defaults to all frames if neither parameter is given.

    Returns:
        If traces is a dict: dict mapping label -> dict with keys
            dff (full trace), dff_mean, dff_peak, f0, f_mean.
        If traces is array: single dict with those keys.
    """
    single = not isinstance(traces, dict)
    if single:
        traces = {'_single': np.asarray(traces, dtype=np.float64)}

    results = {}
    for label, trace in traces.items():
        trace = np.asarray(trace, dtype=np.float64)

        if baseline_indices is not None:
            f0 = trace[baseline_indices].mean()
        elif n_baseline is not None:
            f0 = trace[:n_baseline].mean()
        else:
            f0 = trace.mean()

        if f0 > 1e-6:
            dff = (trace - f0) / f0
        else:
            dff = trace - f0

        results[label] = {
            'dff': dff,
            'dff_mean': float(dff.mean()),
            'dff_peak': float(dff.max()),
            'f0': float(f0),
            'f_mean': float(trace.mean()),
        }

    return results['_single'] if single else results


def detect_calcium_transients(trace, baseline_frames=None, threshold_std=3.0,
                               min_separation=2):
    """Detect calcium transient events in a fluorescence trace.

    A transient is a frame where intensity exceeds baseline + threshold_std * baseline_std.

    Args:
        trace: 1D array of fluorescence intensities.
        baseline_frames: Indices to use for baseline statistics. If None,
            uses frames below the median (assumes sparse activity).
        threshold_std: Number of baseline standard deviations above mean
            to count as a transient.
        min_separation: Minimum frames between distinct events.

    Returns:
        dict with:
            event_indices: Array of frame indices where transients occur.
            event_amplitudes: Peak amplitude (F - F₀) of each event.
            n_events: Number of detected events.
            event_rate: Events per frame.
            baseline_mean: Estimated F₀.
            baseline_std: Noise level of baseline.
            threshold: Absolute intensity threshold used.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)

    if baseline_frames is not None:
        bl = trace[baseline_frames]
    else:
        med = np.median(trace)
        bl = trace[trace <= med]
        if len(bl) < 3:
            bl = trace

    bl_mean = float(bl.mean())
    bl_std = float(bl.std()) if len(bl) > 1 else 1.0
    threshold = bl_mean + threshold_std * max(bl_std, 1e-6)

    above = trace > threshold
    event_indices = []
    event_amplitudes = []

    i = 0
    while i < n:
        if above[i]:
            # Find peak of this event
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            peak_idx = i + int(np.argmax(trace[i:j + 1]))
            event_indices.append(peak_idx)
            event_amplitudes.append(float(trace[peak_idx] - bl_mean))
            i = j + min_separation
        else:
            i += 1

    return {
        'event_indices': np.array(event_indices, dtype=int),
        'event_amplitudes': np.array(event_amplitudes),
        'n_events': len(event_indices),
        'event_rate': len(event_indices) / max(n, 1),
        'baseline_mean': bl_mean,
        'baseline_std': bl_std,
        'threshold': float(threshold),
    }
