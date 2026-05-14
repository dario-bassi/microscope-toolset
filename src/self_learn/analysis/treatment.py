"""Treatment response analysis for before/after comparisons.

Quantifies changes between baseline and post-treatment conditions:
fold changes, percent changes, response classification, and
multi-condition comparison.

Functions:
    compute_change      -- Calculate various change metrics
    classify_response   -- Categorize response as increase/decrease/unchanged
    compare_conditions  -- Compare multiple treatment conditions
    temporal_response   -- Analyze response over time
    compute_half_time   -- Time to reach 50% of max response
    recovery_fraction   -- Fraction of response recovered after washout
    value_at_time       -- Interpolate value at exact target time
"""

import numpy as np


def compute_change(baseline, treated, metric='fold_change'):
    """Calculate change between baseline and treated measurements.

    Args:
        baseline: float or array, baseline measurement(s).
        treated: float or array, post-treatment measurement(s).
        metric: str, type of change metric:
            'fold_change' — treated / baseline
            'percent_change' — (treated - baseline) / baseline * 100
            'difference' — treated - baseline
            'log2_fold_change' — log2(treated / baseline)
            'dff' — (treated - baseline) / baseline (ΔF/F₀)

    Returns:
        dict with:
            change: float or array, the computed change metric.
            metric: str, name of metric used.
            baseline_mean: float.
            treated_mean: float.
    """
    baseline = np.asarray(baseline, dtype=float)
    treated = np.asarray(treated, dtype=float)

    b_mean = float(np.mean(baseline))
    t_mean = float(np.mean(treated))

    if metric == 'fold_change':
        change = treated / np.where(baseline != 0, baseline, np.nan)
    elif metric == 'percent_change':
        change = (treated - baseline) / np.where(
            baseline != 0, baseline, np.nan) * 100
    elif metric == 'difference':
        change = treated - baseline
    elif metric == 'log2_fold_change':
        ratio = treated / np.where(baseline > 0, baseline, np.nan)
        change = np.log2(np.where(ratio > 0, ratio, np.nan))
    elif metric == 'dff':
        change = (treated - baseline) / np.where(
            baseline != 0, baseline, np.nan)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Scalar output for scalar inputs
    if change.ndim == 0:
        change = float(change)

    return {
        'change': change,
        'metric': metric,
        'baseline_mean': round(b_mean, 4),
        'treated_mean': round(t_mean, 4),
    }


def classify_response(change_value, threshold=0.1, metric='fold_change'):
    """Classify response as increase, decrease, or unchanged.

    Args:
        change_value: float, change metric value.
        threshold: float, minimum change to be classified as response.
            For fold_change: threshold above/below 1.0.
            For percent_change: absolute percentage threshold.
            For dff: absolute ΔF/F threshold.
        metric: str, which metric was used (affects thresholds).

    Returns:
        dict with:
            response: str, 'increase', 'decrease', or 'unchanged'.
            magnitude: str, 'none', 'mild', 'moderate', or 'strong'.
            change_value: float.
    """
    if metric in ('fold_change',):
        # Fold change: 1.0 = no change
        deviation = abs(change_value - 1.0)
        is_increase = change_value > 1.0 + threshold
        is_decrease = change_value < 1.0 - threshold
    elif metric in ('percent_change', 'dff'):
        deviation = abs(change_value)
        is_increase = change_value > threshold * 100 if metric == 'percent_change' else change_value > threshold
        is_decrease = change_value < -threshold * 100 if metric == 'percent_change' else change_value < -threshold
    elif metric == 'log2_fold_change':
        deviation = abs(change_value)
        is_increase = change_value > np.log2(1 + threshold)
        is_decrease = change_value < -np.log2(1 + threshold)
    else:
        deviation = abs(change_value)
        is_increase = change_value > threshold
        is_decrease = change_value < -threshold

    if is_increase:
        response = 'increase'
    elif is_decrease:
        response = 'decrease'
    else:
        response = 'unchanged'

    # Magnitude classification
    if response == 'unchanged':
        magnitude = 'none'
    elif deviation < 0.3:
        magnitude = 'mild'
    elif deviation < 1.0:
        magnitude = 'moderate'
    else:
        magnitude = 'strong'

    return {
        'response': response,
        'magnitude': magnitude,
        'change_value': round(float(change_value), 4),
    }


def compare_conditions(conditions, metric='fold_change'):
    """Compare multiple treatment conditions against baseline.

    Args:
        conditions: dict mapping condition_name → (baseline, treated).
            Each value is a tuple of (float_or_array, float_or_array).
        metric: str, change metric to use.

    Returns:
        dict with:
            per_condition: dict of condition_name → change result.
            summary: dict with overall statistics.
            strongest: str, condition with largest absolute change.
            weakest: str, condition with smallest absolute change.
    """
    per_condition = {}
    changes = {}

    for name, (baseline, treated) in conditions.items():
        result = compute_change(baseline, treated, metric)
        change_val = result['change']
        if isinstance(change_val, np.ndarray):
            change_val = float(np.nanmean(change_val))
        response = classify_response(change_val, metric=metric)
        per_condition[name] = {**result, **response}
        changes[name] = change_val

    # Find strongest and weakest
    if changes:
        if metric == 'fold_change':
            abs_changes = {k: abs(v - 1.0) for k, v in changes.items()}
        else:
            abs_changes = {k: abs(v) for k, v in changes.items()}
        strongest = max(abs_changes, key=abs_changes.get)
        weakest = min(abs_changes, key=abs_changes.get)
    else:
        strongest = ''
        weakest = ''

    return {
        'per_condition': per_condition,
        'summary': {
            'n_conditions': len(conditions),
            'n_increased': sum(1 for c in per_condition.values()
                             if c['response'] == 'increase'),
            'n_decreased': sum(1 for c in per_condition.values()
                             if c['response'] == 'decrease'),
            'n_unchanged': sum(1 for c in per_condition.values()
                             if c['response'] == 'unchanged'),
        },
        'strongest': strongest,
        'weakest': weakest,
    }


def temporal_response(timepoints, values, baseline_end=None,
                      treatment_start=None):
    """Analyze temporal response curve.

    Identifies baseline, onset, peak, and steady-state phases.

    Args:
        timepoints: array-like, time values.
        values: array-like, measured values at each timepoint.
        baseline_end: int or None, index where baseline ends.
        treatment_start: int or None, index where treatment begins.

    Returns:
        dict with:
            baseline_mean: float, mean during baseline phase.
            baseline_std: float, std during baseline.
            peak_value: float, maximum post-treatment.
            peak_time: float, time of peak.
            steady_state: float, mean of last 20% of values.
            onset_index: int, first post-treatment point above baseline + 2*std.
            max_change: float, peak - baseline_mean.
            time_to_peak: float, time from treatment to peak.
    """
    timepoints = np.asarray(timepoints, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)

    if baseline_end is None:
        baseline_end = n // 5  # first 20%
    if treatment_start is None:
        treatment_start = baseline_end

    # Baseline statistics
    bl_values = values[:baseline_end]
    bl_mean = float(np.mean(bl_values)) if len(bl_values) > 0 else 0
    bl_std = float(np.std(bl_values)) if len(bl_values) > 0 else 0

    # Post-treatment analysis
    post = values[treatment_start:]
    post_times = timepoints[treatment_start:]

    if len(post) == 0:
        return {
            'baseline_mean': bl_mean, 'baseline_std': bl_std,
            'peak_value': bl_mean, 'peak_time': 0,
            'steady_state': bl_mean, 'onset_index': -1,
            'max_change': 0, 'time_to_peak': 0,
        }

    peak_idx = int(np.argmax(np.abs(post - bl_mean)))
    peak_value = float(post[peak_idx])
    peak_time = float(post_times[peak_idx])

    # Steady state: mean of last 20%
    last_20pct = max(1, len(post) // 5)
    steady_state = float(np.mean(post[-last_20pct:]))

    # Onset: first point where |value - baseline| > 2*std
    onset_index = -1
    threshold = bl_mean + 2 * bl_std if bl_std > 0 else bl_mean * 1.1
    for i, v in enumerate(post):
        if abs(v - bl_mean) > abs(threshold - bl_mean):
            onset_index = treatment_start + i
            break

    return {
        'baseline_mean': round(bl_mean, 4),
        'baseline_std': round(bl_std, 4),
        'peak_value': round(peak_value, 4),
        'peak_time': round(peak_time, 4),
        'steady_state': round(steady_state, 4),
        'onset_index': onset_index,
        'max_change': round(float(peak_value - bl_mean), 4),
        'time_to_peak': round(float(peak_time - timepoints[treatment_start]), 4),
    }


def compute_half_time(timepoints, values, baseline_value=None):
    """Compute time to reach 50% of maximum response.

    Works for both increasing and decreasing responses. The half-time is
    interpolated linearly between the two frames bracketing the 50% level.

    Args:
        timepoints: array-like, time values.
        values: array-like, measured values.
        baseline_value: float or None. If None, uses the first value.

    Returns:
        dict with:
            half_time: float, time to reach 50% of max change from baseline.
                Time is relative to the first timepoint. NaN if not reached.
            max_change: float, maximum change from baseline.
            direction: str, 'decrease' or 'increase'.
            half_target: float, the value at the 50% point.
    """
    t = np.asarray(timepoints, dtype=float)
    v = np.asarray(values, dtype=float)

    if len(v) < 2:
        return {'half_time': float('nan'), 'max_change': 0.0,
                'direction': 'none', 'half_target': 0.0}

    if baseline_value is None:
        baseline_value = float(v[0])

    # Find the steady-state (last 20% of points)
    last_n = max(1, len(v) // 5)
    steady = float(np.mean(v[-last_n:]))
    max_change = steady - baseline_value

    if abs(max_change) < 1e-10:
        return {'half_time': float('nan'), 'max_change': 0.0,
                'direction': 'none', 'half_target': baseline_value}

    direction = 'decrease' if max_change < 0 else 'increase'
    half_target = baseline_value + max_change * 0.5

    # Find crossing point with linear interpolation
    for i in range(1, len(v)):
        crossed = False
        if direction == 'decrease' and v[i] <= half_target and v[i - 1] > half_target:
            crossed = True
        elif direction == 'increase' and v[i] >= half_target and v[i - 1] < half_target:
            crossed = True

        if crossed:
            # Linear interpolation
            frac = (half_target - v[i - 1]) / (v[i] - v[i - 1])
            ht = t[i - 1] + frac * (t[i] - t[i - 1])
            return {
                'half_time': round(float(ht - t[0]), 2),
                'max_change': round(float(max_change), 4),
                'direction': direction,
                'half_target': round(float(half_target), 4),
            }

    return {'half_time': float('nan'), 'max_change': round(float(max_change), 4),
            'direction': direction, 'half_target': round(float(half_target), 4)}


def recovery_fraction(baseline_value, nadir_value, recovery_value):
    """Compute the fraction of response recovered after washout/reversal.

    For a decrease response: recovery_fraction = (recovery - nadir) / (baseline - nadir)
    For an increase response: recovery_fraction = (nadir - recovery) / (nadir - baseline)

    Args:
        baseline_value: float, pre-treatment baseline.
        nadir_value: float, maximum effect (minimum for decrease, max for increase).
        recovery_value: float, value after recovery period.

    Returns:
        float: Recovery fraction (0 = no recovery, 1 = full recovery).
            Clamped to [0, 1]. Returns 0 if there was no change.
    """
    change = abs(baseline_value - nadir_value)
    if change < 1e-10:
        return 0.0

    if baseline_value > nadir_value:
        # Decrease response
        frac = (recovery_value - nadir_value) / change
    else:
        # Increase response
        frac = (nadir_value - recovery_value) / change

    return max(0.0, min(float(frac), 1.0))


def value_at_time(timepoints, values, target_time):
    """Interpolate a value at an exact target time.

    Uses linear interpolation between the two bracketing timepoints.
    Useful for reporting measurements at specific times (e.g., "N:C ratio
    30s after drug removal") when actual frame times don't exactly match.

    Args:
        timepoints: 1D array of time values (must be monotonically increasing).
        values: 1D array of measured values at each timepoint.
        target_time: float, the time at which to interpolate.

    Returns:
        float: Interpolated value at target_time.
            If target_time is before the first timepoint, returns values[0].
            If target_time is after the last timepoint, returns values[-1].
    """
    t = np.asarray(timepoints, dtype=float)
    v = np.asarray(values, dtype=float)

    if len(t) == 0:
        return float('nan')
    if len(t) == 1 or target_time <= t[0]:
        return float(v[0])
    if target_time >= t[-1]:
        return float(v[-1])

    return float(np.interp(target_time, t, v))
