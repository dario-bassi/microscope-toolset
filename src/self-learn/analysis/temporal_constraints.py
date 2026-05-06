"""Temporal constraints for time-series data.

Enforces physical constraints on time-series measurements:
monotonicity (for wound closure, cell growth), continuity
(flag jumps), and smoothing under constraints. Prevents
artifacts from noisy frame-by-frame re-counting.

Functions:
    enforce_monotonic     -- Force monotonic increase or decrease
    detect_jumps          -- Flag discontinuities in a time series
    smooth_constrained    -- Smooth while respecting constraints
    validate_timecourse   -- Check if a timecourse is physically plausible
    interpolate_gaps      -- Fill missing/rejected frames with interpolation
"""

import numpy as np


def enforce_monotonic(values, direction="increasing", method="isotonic"):
    """Force a time series to be monotonically increasing or decreasing.

    Useful for cell counts (should only increase in growth), wound
    closure (gap should only decrease), and confluency (only increases).

    Args:
        values: 1D array of measurements over time.
        direction: 'increasing' or 'decreasing'.
        method: 'isotonic' (pool-adjacent-violators) or 'clip'
            (clip values to running max/min).

    Returns:
        dict with:
            corrected: 1D array, monotonic version.
            n_corrected: int, frames that were adjusted.
            violation_indices: list of int, frames that violated monotonicity.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    corrected = values.copy()
    violations = []

    if n < 2:
        return {
            "corrected": corrected,
            "n_corrected": 0,
            "violation_indices": [],
        }

    if direction == "decreasing":
        # Flip, enforce increasing, flip back
        result = enforce_monotonic(-values, direction="increasing", method=method)
        return {
            "corrected": -result["corrected"],
            "n_corrected": result["n_corrected"],
            "violation_indices": result["violation_indices"],
        }

    if method == "clip":
        # Simple: each value must be >= running max
        running_max = values[0]
        for i in range(1, n):
            if corrected[i] < running_max:
                violations.append(i)
                corrected[i] = running_max
            else:
                running_max = corrected[i]
    elif method == "isotonic":
        # Pool adjacent violators algorithm (PAVA)
        corrected = _pava_increasing(values)
        for i in range(n):
            if corrected[i] != values[i]:
                violations.append(i)
    else:
        raise ValueError(f"method must be 'isotonic' or 'clip', got '{method}'")

    return {
        "corrected": corrected,
        "n_corrected": len(violations),
        "violation_indices": violations,
    }


def detect_jumps(values, max_rate=None, window=1):
    """Flag discontinuities (jumps) in a time series.

    A jump is a frame-to-frame change that exceeds the expected
    maximum rate. Useful for detecting segmentation errors, tracking
    failures, or dropped frames.

    Args:
        values: 1D array.
        max_rate: float or None. Maximum expected change per frame.
            If None, auto-computed as mean + 3*std of frame-to-frame changes.
        window: int, compare values this many frames apart (default 1).

    Returns:
        dict with:
            jump_indices: list of int, frames with jumps.
            jump_sizes: list of float, magnitude of each jump.
            n_jumps: int.
            max_jump: float, largest jump.
            median_rate: float, typical frame-to-frame change.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < window + 1:
        return {
            "jump_indices": [],
            "jump_sizes": [],
            "n_jumps": 0,
            "max_jump": 0.0,
            "median_rate": 0.0,
        }

    # Compute changes
    changes = np.abs(values[window:] - values[:-window]) / window
    median_rate = float(np.median(changes))

    if max_rate is None:
        mean_change = float(np.mean(changes))
        std_change = float(np.std(changes))
        max_rate = mean_change + 3 * std_change
        max_rate = max(max_rate, 1e-10)

    jump_indices = []
    jump_sizes = []
    for i in range(len(changes)):
        if changes[i] > max_rate:
            jump_indices.append(i + window)
            jump_sizes.append(float(changes[i]))

    return {
        "jump_indices": jump_indices,
        "jump_sizes": jump_sizes,
        "n_jumps": len(jump_indices),
        "max_jump": float(max(jump_sizes)) if jump_sizes else 0.0,
        "median_rate": round(median_rate, 4),
    }


def smooth_constrained(values, window=5, constraint=None):
    """Smooth a time series while respecting physical constraints.

    Applies moving average smoothing, then optionally enforces
    monotonicity or bounds.

    Args:
        values: 1D array.
        window: int, smoothing window size.
        constraint: str or None. 'increasing', 'decreasing', 'positive',
            or None for unconstrained smoothing.

    Returns:
        dict with:
            smoothed: 1D array, smoothed values.
            residuals: 1D array, original - smoothed.
            rmse: float, root mean square of residuals.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < window:
        return {
            "smoothed": values.copy(),
            "residuals": np.zeros(n),
            "rmse": 0.0,
        }

    # Moving average via convolution (guaranteed correct length)
    kernel = np.ones(window) / window
    # Pad edges to avoid boundary effects
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    conv = np.convolve(padded, kernel, mode="valid")
    smoothed = conv[:n]

    # Apply constraints
    if constraint == "increasing":
        result = enforce_monotonic(smoothed, "increasing", "isotonic")
        smoothed = result["corrected"]
    elif constraint == "decreasing":
        result = enforce_monotonic(smoothed, "decreasing", "isotonic")
        smoothed = result["corrected"]
    elif constraint == "positive":
        smoothed = np.maximum(smoothed, 0)

    residuals = values - smoothed
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return {
        "smoothed": smoothed,
        "residuals": residuals,
        "rmse": round(rmse, 4),
    }


def validate_timecourse(values, expected_trend=None, max_cv=0.5, max_jump_fraction=0.5):
    """Check if a timecourse is physically plausible.

    Flags potential issues: excessive noise, wrong trend direction,
    suspicious jumps, or non-monotonic behavior where monotonicity
    is expected.

    Args:
        values: 1D array.
        expected_trend: 'increasing', 'decreasing', or None.
        max_cv: float, maximum acceptable coefficient of variation.
        max_jump_fraction: float, maximum single-frame change as fraction
            of the full range.

    Returns:
        dict with:
            valid: bool, True if no issues found.
            issues: list of str, description of each issue.
            trend: str, detected trend ('increasing'/'decreasing'/'flat'/'noisy').
            monotonicity_score: float, fraction of steps in expected direction.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    issues = []

    if n < 3:
        return {
            "valid": True,
            "issues": [],
            "trend": "unknown",
            "monotonicity_score": 1.0,
        }

    # Trend detection
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, values, 1)
    slope = coeffs[0]
    value_range = float(np.ptp(values))
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    # Detrend for noise estimation (remove linear trend before CV)
    detrended = values - np.polyval(coeffs, x)
    residual_std = float(np.std(detrended))
    cv_residual = residual_std / abs(mean_val) if abs(mean_val) > 1e-10 else 0

    # Flat: range is very small relative to mean, or std is near zero
    if value_range < abs(mean_val) * 0.01 or std_val < 1e-10:
        trend = "flat"
    elif abs(slope) * n < std_val * 0.5:
        trend = "flat"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"

    # Use detrended CV for noise assessment (don't penalize linear trends)
    if cv_residual > max_cv and trend != "flat":
        issues.append(f"High variability (residual CV={cv_residual:.2f} > {max_cv})")
        trend = "noisy"

    # Check expected trend
    if expected_trend is not None and trend not in (expected_trend, "flat", "noisy"):
        issues.append(f"Expected '{expected_trend}' trend but detected '{trend}'")

    # Monotonicity score
    diffs = np.diff(values)
    if expected_trend == "increasing":
        mono_score = float(np.sum(diffs >= 0) / len(diffs))
    elif expected_trend == "decreasing":
        mono_score = float(np.sum(diffs <= 0) / len(diffs))
    else:
        # Use dominant direction
        mono_score = float(max(np.sum(diffs >= 0), np.sum(diffs <= 0)) / len(diffs))

    if expected_trend is not None and mono_score < 0.6:
        issues.append(f"Poor monotonicity ({mono_score:.2f})")

    # Check for jumps
    value_range = float(np.ptp(values))
    if value_range > 0:
        max_change = float(np.max(np.abs(diffs)))
        if max_change / value_range > max_jump_fraction:
            issues.append(
                f"Large jump detected ({max_change:.1f}, "
                f"{max_change/value_range*100:.0f}% of range)"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "trend": trend,
        "monotonicity_score": round(mono_score, 4),
    }


def interpolate_gaps(values, gap_mask, method="linear"):
    """Fill missing or rejected frames with interpolation.

    Args:
        values: 1D array with gaps.
        gap_mask: 1D bool array, True = gap (needs filling).
        method: 'linear', 'nearest', or 'previous'.

    Returns:
        dict with:
            filled: 1D array with gaps interpolated.
            n_filled: int, number of frames filled.
    """
    values = np.asarray(values, dtype=float)
    gap_mask = np.asarray(gap_mask, dtype=bool)
    filled = values.copy()
    len(values)

    good_idx = np.where(~gap_mask)[0]
    gap_idx = np.where(gap_mask)[0]

    if len(good_idx) < 2 or len(gap_idx) == 0:
        return {"filled": filled, "n_filled": 0}

    if method == "linear":
        filled[gap_idx] = np.interp(gap_idx, good_idx, values[good_idx])
    elif method == "nearest":
        for gi in gap_idx:
            nearest = good_idx[np.argmin(np.abs(good_idx - gi))]
            filled[gi] = values[nearest]
    elif method == "previous":
        for gi in gap_idx:
            prev = good_idx[good_idx < gi]
            if len(prev) > 0:
                filled[gi] = values[prev[-1]]
            else:
                # No previous — use next
                nxt = good_idx[good_idx > gi]
                if len(nxt) > 0:
                    filled[gi] = values[nxt[0]]

    return {
        "filled": filled,
        "n_filled": int(gap_mask.sum()),
    }


def _pava_increasing(values):
    """Pool adjacent violators algorithm for isotonic regression."""
    n = len(values)
    result = values.copy()
    # Block structure: each block has (sum, count, start, end)
    blocks = [[result[i], 1] for i in range(n)]

    i = 0
    while i < len(blocks) - 1:
        # Check if current block mean > next block mean (violation)
        curr_mean = blocks[i][0] / blocks[i][1]
        next_mean = blocks[i + 1][0] / blocks[i + 1][1]

        if curr_mean > next_mean:
            # Merge blocks
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks.pop(i + 1)
            # Check backwards
            if i > 0:
                i -= 1
        else:
            i += 1

    # Reconstruct result
    idx = 0
    for block_sum, block_count in blocks:
        mean_val = block_sum / block_count
        for _j in range(block_count):
            result[idx] = mean_val
            idx += 1

    return result
