"""Automatic experimental phase segmentation.

Segments time-series data into distinct experimental phases
(baseline, ramp, plateau, decline, recovery) without requiring
manual phase boundary specification. Useful for drug response,
temperature shift, and perfusion experiments.

Functions:
    segment_phases       -- Auto-detect phase boundaries from signal
    classify_phases      -- Label segments as baseline/ramp/plateau/decline
    detect_steady_state  -- Find when signal reaches steady state
    phase_metrics        -- Compute per-phase statistics
    experiment_phases    -- One-call pipeline: segment + classify + metrics
"""

import numpy as np


def segment_phases(signal, min_segment=5, sensitivity=1.0,
                   noise_k=2.0, range_frac=0.05):
    """Auto-detect phase boundaries in an experimental time series.

    Uses a combination of derivative analysis and variance-based
    changepoint detection to find transitions.

    Args:
        signal: 1D array of measurements over time.
        min_segment: int, minimum frames per phase.
        sensitivity: float, lower = fewer phases (default 1.0).
            Values 0.5-2.0 are typical.
        noise_k: float, multiple of baseline noise that the derivative
            magnitude must exceed before a frame is flagged as in
            transition. Default 2.0 (two sigma); raise for noisier
            signals, lower if you need to catch subtler boundaries.
        range_frac: float in [0, 1], fraction of the total signal range
            used as the floor for the transition threshold (so flat
            regions don't trip on numerical noise). Default 0.05 (5%).

    Returns:
        dict with:
            boundaries: list of frame indices where phases change.
            n_phases: int, number of detected phases.
            segments: list of (start, end) tuples.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if n < 2 * min_segment:
        return {
            'boundaries': [],
            'n_phases': 1,
            'segments': [(0, n)],
        }

    # Smooth for derivative estimation
    kernel = max(3, min_segment // 2)
    if kernel % 2 == 0:
        kernel += 1
    smoothed = _moving_average(signal, kernel)

    # Compute derivative magnitude
    deriv = np.gradient(smoothed)
    deriv_mag = np.abs(deriv)

    # Threshold for significant change
    # Use the overall signal range to set a meaningful threshold
    signal_range = float(np.ptp(signal))
    baseline_noise = np.std(signal[:min(min_segment * 2, n)])
    if baseline_noise < 1e-10:
        baseline_noise = np.std(signal) * 0.1
    # Threshold must exceed both noise-based and range-based minimums
    noise_thresh = baseline_noise * sensitivity * float(noise_k)
    range_thresh = signal_range * float(range_frac) * sensitivity
    threshold = max(noise_thresh, range_thresh)

    # Find transition regions (where derivative exceeds threshold)
    is_transition = deriv_mag > threshold

    # Group transitions into boundary points
    boundaries = []
    in_transition = False
    trans_start = 0

    for i in range(len(is_transition)):
        if is_transition[i] and not in_transition:
            trans_start = i
            in_transition = True
        elif not is_transition[i] and in_transition:
            # End of transition — boundary is the midpoint
            mid = (trans_start + i) // 2
            # Check minimum segment length
            if boundaries:
                if mid - boundaries[-1] >= min_segment:
                    boundaries.append(mid)
            elif mid >= min_segment:
                boundaries.append(mid)
            in_transition = False

    # If still in transition at end, close it
    if in_transition:
        mid = (trans_start + n) // 2
        if not boundaries or mid - boundaries[-1] >= min_segment:
            if n - mid >= min_segment:
                boundaries.append(mid)

    # Refine boundaries using mean-shift within local windows
    refined = []
    for b in boundaries:
        window = min_segment
        start = max(0, b - window)
        end = min(n, b + window)
        seg = signal[start:end]
        if len(seg) >= 2 * min_segment:
            # Find the split that maximizes between-segment variance
            best_score = -1
            best_pos = b
            for j in range(min_segment, len(seg) - min_segment):
                left_mean = seg[:j].mean()
                right_mean = seg[j:].mean()
                score = abs(right_mean - left_mean)
                if score > best_score:
                    best_score = score
                    best_pos = start + j
            refined.append(best_pos)
        else:
            refined.append(b)

    # Remove duplicates and sort
    boundaries = sorted(set(refined))

    # Ensure last segment is long enough
    if boundaries and n - boundaries[-1] < min_segment:
        boundaries.pop()

    # Build segments
    all_bounds = [0] + boundaries + [n]
    segments = [(all_bounds[i], all_bounds[i + 1])
                for i in range(len(all_bounds) - 1)]

    return {
        'boundaries': boundaries,
        'n_phases': len(segments),
        'segments': segments,
    }


DEFAULT_CLASS_THRESHOLDS = {
    # σ-units above the baseline std at which a segment counts as "different"
    'plateau_deviation_sigma': 2.0,
    'recovery_deviation_sigma': 2.0,
    # current deviation must drop below this fraction of the previous
    # deviation before we call it 'recovery'
    'recovery_decay_ratio': 0.7,
    # rel-slope magnitude (|slope| / max(|baseline_mean|, baseline_std))
    # above which a segment is a ramp
    'ramp_rel_slope': 0.02,
    # plateau std-cap: seg_std < deviation * this → counted as plateau,
    # higher std → keep evaluating
    'plateau_std_frac': 0.3,
}


def classify_phases(signal, segments, baseline_std_floor_frac=0.01,
                    class_thresholds=None):
    """Label each segment with a phase type.

    Phase types:
        'baseline' — low variance, early in sequence
        'ramp_up' — increasing trend
        'ramp_down' — decreasing trend
        'plateau' — stable at non-baseline level
        'recovery' — returning toward baseline level

    Args:
        signal: 1D array of measurements.
        segments: list of (start, end) tuples from segment_phases().
        baseline_std_floor_frac: floor on baseline σ as a fraction of
            |baseline_mean| when the measured σ is ~0.
        class_thresholds: optional dict overriding the discriminator
            thresholds. See ``DEFAULT_CLASS_THRESHOLDS`` for keys; missing
            keys fall back to defaults. Recipes whose phase signal has
            different noise / dynamic-range characteristics (calcium vs
            fluorescence vs OD600) can override here without forking.

    Returns:
        dict with:
            labels: list of str, phase label for each segment.
            trends: list of float, slope for each segment.
            means: list of float, mean for each segment.
    """
    th = dict(DEFAULT_CLASS_THRESHOLDS)
    if class_thresholds:
        th.update(class_thresholds)
    signal = np.asarray(signal, dtype=float)

    labels = []
    trends = []
    means = []

    if not segments:
        return {'labels': [], 'trends': [], 'means': []}

    # Compute per-segment stats
    for start, end in segments:
        seg = signal[start:end]
        seg_mean = float(seg.mean())
        means.append(round(seg_mean, 4))

        # Compute trend via linear fit
        if len(seg) >= 2:
            x = np.arange(len(seg), dtype=float)
            slope = float(np.polyfit(x, seg, 1)[0])
        else:
            slope = 0.0
        trends.append(round(slope, 6))

    # Baseline reference: first segment's mean
    baseline_mean = means[0]
    baseline_std = float(np.std(signal[segments[0][0]:segments[0][1]]))
    if baseline_std < 1e-10:
        baseline_std = (
            abs(baseline_mean) * float(baseline_std_floor_frac)
            if baseline_mean != 0
            else 1.0
        )

    # Classify each segment
    for i, ((start, end), slope, seg_mean) in enumerate(
            zip(segments, trends, means)):
        seg = signal[start:end]
        seg_std = float(np.std(seg))
        deviation = abs(seg_mean - baseline_mean)
        rel_slope = abs(slope) / max(abs(baseline_mean), baseline_std, 1e-10)

        if i == 0:
            labels.append('baseline')
            continue

        # Check if this segment is returning toward baseline
        if i >= 2:
            prev_mean = means[i - 1]
            prev_deviation = abs(prev_mean - baseline_mean)
            curr_deviation = deviation
            if (prev_deviation > th['recovery_deviation_sigma'] * baseline_std and
                    curr_deviation < prev_deviation * th['recovery_decay_ratio']):
                labels.append('recovery')
                continue

        # Ramp detection: significant trend
        if rel_slope > th['ramp_rel_slope'] and abs(slope) * len(seg) > baseline_std:
            if slope > 0:
                labels.append('ramp_up')
            else:
                labels.append('ramp_down')
            continue

        # Plateau: stable but different from baseline
        if (deviation > th['plateau_deviation_sigma'] * baseline_std
                and seg_std < deviation * th['plateau_std_frac']):
            labels.append('plateau')
            continue

        # If close to baseline, it's a return to baseline
        if deviation < th['plateau_deviation_sigma'] * baseline_std:
            if i > 1:
                labels.append('recovery')
            else:
                labels.append('baseline')
            continue

        # Default: classify by trend
        if slope > 0:
            labels.append('ramp_up')
        elif slope < 0:
            labels.append('ramp_down')
        else:
            labels.append('plateau')

    return {
        'labels': labels,
        'trends': trends,
        'means': means,
    }


def detect_steady_state(signal, window=5, tolerance=0.05):
    """Find when a signal reaches steady state.

    Steady state is defined as the first frame where the rolling
    coefficient of variation drops below tolerance and stays there.

    Args:
        signal: 1D array.
        window: int, rolling window size for CV calculation.
        tolerance: float, CV threshold for steady state (0.05 = 5%).

    Returns:
        dict with:
            onset_index: int, frame where steady state begins (-1 if never).
            steady_value: float, mean value during steady state.
            time_to_steady: int, frames from start to steady state.
            cv_trace: 1D array, rolling CV at each frame.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if n < window:
        return {
            'onset_index': -1,
            'steady_value': float(signal.mean()) if n > 0 else 0.0,
            'time_to_steady': n,
            'cv_trace': np.zeros(n),
        }

    # Compute rolling CV
    cv_trace = np.zeros(n)
    for i in range(window - 1, n):
        seg = signal[i - window + 1:i + 1]
        seg_mean = seg.mean()
        if abs(seg_mean) > 1e-10:
            cv_trace[i] = seg.std() / abs(seg_mean)
        else:
            cv_trace[i] = 0.0

    # Find first sustained period below tolerance
    onset = -1
    sustain = window  # must stay below tolerance for this many frames
    count = 0
    for i in range(window - 1, n):
        if cv_trace[i] <= tolerance:
            count += 1
            if count >= sustain:
                onset = i - sustain + 1
                break
        else:
            count = 0

    if onset >= 0:
        steady_value = float(signal[onset:].mean())
    else:
        steady_value = float(signal[-window:].mean())

    return {
        'onset_index': onset,
        'steady_value': round(steady_value, 4),
        'time_to_steady': onset if onset >= 0 else n,
        'cv_trace': cv_trace,
    }


def phase_metrics(signal, segments, labels=None):
    """Compute detailed statistics for each experimental phase.

    Args:
        signal: 1D array.
        segments: list of (start, end) tuples.
        labels: list of str or None (auto-classified if None).

    Returns:
        dict with:
            phases: list of dicts, each containing:
                label, start, end, duration, mean, std, min, max,
                slope, fold_change_from_baseline.
            baseline_mean: float.
            max_deviation: float, maximum deviation from baseline.
    """
    signal = np.asarray(signal, dtype=float)

    if labels is None:
        result = classify_phases(signal, segments)
        labels = result['labels']

    phases = []
    baseline_mean = None

    for i, (start, end) in enumerate(segments):
        seg = signal[start:end]
        seg_mean = float(seg.mean())

        if i == 0:
            baseline_mean = seg_mean

        # Linear slope
        if len(seg) >= 2:
            x = np.arange(len(seg), dtype=float)
            slope = float(np.polyfit(x, seg, 1)[0])
        else:
            slope = 0.0

        # Fold change from baseline
        if baseline_mean and abs(baseline_mean) > 1e-10:
            fold_change = seg_mean / baseline_mean
        else:
            fold_change = 1.0

        label = labels[i] if i < len(labels) else 'unknown'
        phases.append({
            'label': label,
            'start': start,
            'end': end,
            'duration': end - start,
            'mean': round(seg_mean, 4),
            'std': round(float(seg.std()), 4),
            'min': round(float(seg.min()), 4),
            'max': round(float(seg.max()), 4),
            'slope': round(slope, 6),
            'fold_change_from_baseline': round(fold_change, 4),
        })

    if baseline_mean is None:
        baseline_mean = float(signal.mean())

    # Max deviation
    deviations = [abs(p['mean'] - baseline_mean) for p in phases]
    max_dev = max(deviations) if deviations else 0.0

    return {
        'phases': phases,
        'baseline_mean': round(baseline_mean, 4),
        'max_deviation': round(max_dev, 4),
    }


def experiment_phases(signal, min_segment=5, sensitivity=1.0):
    """One-call pipeline: segment + classify + metrics.

    Convenience function that runs the full phase analysis pipeline.

    Args:
        signal: 1D array of measurements over time.
        min_segment: int, minimum frames per phase.
        sensitivity: float, changepoint detection sensitivity.

    Returns:
        dict with all outputs from segment_phases, classify_phases,
        and phase_metrics combined.
    """
    signal = np.asarray(signal, dtype=float)

    seg_result = segment_phases(signal, min_segment, sensitivity)
    cls_result = classify_phases(signal, seg_result['segments'])
    met_result = phase_metrics(signal, seg_result['segments'],
                               cls_result['labels'])

    return {
        'boundaries': seg_result['boundaries'],
        'n_phases': seg_result['n_phases'],
        'segments': seg_result['segments'],
        'labels': cls_result['labels'],
        'trends': cls_result['trends'],
        'means': cls_result['means'],
        'phases': met_result['phases'],
        'baseline_mean': met_result['baseline_mean'],
        'max_deviation': met_result['max_deviation'],
    }


def _moving_average(signal, window):
    """Simple centered moving average with edge padding."""
    if window <= 1 or len(signal) < window:
        return signal.copy()
    pad = window // 2
    padded = np.pad(signal, pad, mode='edge')
    cumsum = np.cumsum(padded)
    return (cumsum[window:] - cumsum[:-window]) / window
