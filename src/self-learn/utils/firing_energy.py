"""Per-pixel firing-energy from a temporal stack — bleach-immune.

Any channel that bleaches (GCaMP, voltage indicators, FRAP, photo-
conversion, photoswitching) carries a global decay envelope across a
burst. Raw `stack.std(axis=0)` and `σ × mean` then rank pixels by how
much their *trace* changes — which on a bleached image picks
strongly-fluorescing-then-decaying pixels (e.g. wave-front cells with
no firings) as easily as it picks real pacemakers.

The fix is to detrend before taking variance. The simplest detrender
that survives non-linear bleach is to take ``np.diff`` along the time
axis and keep only the **positive** rises — bleach decay produces a
steady negative diff which contributes 0; firings produce positive
spikes.

Lesson source: ch651 r3 (2026-04-28). On the cardio backend a
wave-front pixel at (440, 214) had σ=18.6 (purely from a 143→79 bleach
trajectory over 30 frames) — *higher* than the GT primary at (144, 116)
with σ=16.1. σ × mean and σ-only top-K both ranked the wave-front ahead
of the real source.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def firing_energy(
    stack: np.ndarray,
    *,
    diff_threshold: Optional[float] = None,
) -> np.ndarray:
    """Per-pixel positive-rise energy across a temporal stack.

    For each pixel, computes ``Σ_t max(0, stack[t+1] - stack[t])`` — the
    sum of frame-to-frame brightness *increases*. A monotonic bleach
    decay contributes 0 (every diff is non-positive); a periodic firing
    contributes one unit per firing event proportional to the firing
    amplitude.

    Args:
        stack: ``(T, H, W)`` array.
        diff_threshold: optional minimum diff to count a rise. Set to
            e.g. 5-10 ADU to suppress shot-noise jitter in dim regions.
            If ``None``, all positive diffs contribute.

    Returns:
        ``(H, W)`` float32 array, the per-pixel firing-energy score.
        Higher value = more / brighter rises observed in the time series.

    Example:
        >>> stack = ...  # (T, H, W) GCaMP burst
        >>> energy = firing_energy(stack, diff_threshold=10)
        >>> # `energy` is bleach-immune; `stack.std(axis=0)` is not.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError(f"expected 3-D (T, H, W); got shape {stack.shape}")
    if stack.shape[0] < 2:
        return np.zeros(stack.shape[1:], dtype=np.float32)
    diff = np.diff(stack, axis=0)
    if diff_threshold is not None:
        diff = np.where(diff > float(diff_threshold), diff, 0.0)
    else:
        diff = np.clip(diff, 0.0, None)
    return diff.sum(axis=0).astype(np.float32)


def detrended_sigma(stack: np.ndarray) -> np.ndarray:
    """Per-pixel σ after subtracting a per-pixel linear time trend.

    Alternative bleach-immune variance estimator. Where
    :func:`firing_energy` counts **rises**, ``detrended_sigma`` measures
    the spread *around* the smooth bleach line — slower to compute
    (per-pixel polyfit) but preserves the standard "high σ = active
    pixel" framing for callers that already use σ × mean.

    Returns:
        ``(H, W)`` float32 array of detrended standard deviations.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError(f"expected 3-D (T, H, W); got shape {stack.shape}")
    T, H, W = stack.shape
    if T < 3:
        return stack.std(axis=0).astype(np.float32)
    t = np.arange(T, dtype=float)
    flat = stack.reshape(T, H * W)
    # Vectorised linear fit per-pixel.
    t_mean = t.mean()
    t_dev = t - t_mean
    denom = (t_dev * t_dev).sum()
    flat_mean = flat.mean(axis=0, keepdims=True)
    slopes = (t_dev[:, None] * (flat - flat_mean)).sum(axis=0) / denom
    intercepts = flat_mean[0] - slopes * t_mean
    fit = intercepts[None, :] + slopes[None, :] * t[:, None]
    residuals = flat - fit
    return residuals.std(axis=0).reshape(H, W).astype(np.float32)
