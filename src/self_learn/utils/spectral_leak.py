"""Spectral-leak primitives: top-percentile masks + N×N leak matrix + unmix.

Pure-data utilities — no live core required for tests.

Composes nothing (pure numpy + scipy.ndimage). Consumed by
:mod:`src.recipes.spectral_unmix`.

Three primitives:

1. :func:`top_percentile_mask` — sigma-free thresholding for the
   "single-stain control" spatial proxy. Top-k percent of intensity,
   optional ``exclude`` mask, connected-component cleanup.
2. :func:`measure_leak` — N×N leak matrix from
   ``{channel_name: image}`` + ``{channel_name: mask}``. Diagonal=1 by
   definition; off-diagonal ``K[i,j] = (channels[j][masks[i]].mean() -
   bg[j]) / (channels[i][masks[i]].mean() - bg[i])``. NaN-safe (zero
   in-channel signal → NaN entry, no crash).
3. :func:`unmix` — ``np.linalg.solve(K, observed)``. Caller provides
   the observed N-channel vector (per pixel / per ROI / per cell).
   Singular K raises ValueError.

Tested on: ch614 r1 → 10/10 (counter=104). The recipe in
:mod:`src.recipes.spectral_unmix` composes these into the
single-call leak-measurement kernel; this module is the
agent-portable substrate.

This module is intentionally NOT registered in
:mod:`src.core.utils.auto_recipe`. Spectral leak is a brief-level
signal ("crosstalk", "unmix", "bleed") that the agent picks
deliberately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


_DEFAULT_BG_PERCENTILE = 20.0


def top_percentile_mask(
    image: np.ndarray,
    *,
    percentile: float = 92.0,
    exclude: Optional[np.ndarray] = None,
    min_area_px: int = 50,
) -> np.ndarray:
    """Top-percentile threshold + CC cleanup (the ch614 r1 spatial proxy).

    Returns a boolean mask of pixels above the ``percentile``-th
    intensity AND outside ``exclude`` (if given), keeping only
    connected components ≥ ``min_area_px``.

    ``exclude`` is the mutual-exclusion partner — e.g. when building
    a membrane mask, pass the nucleus mask as ``exclude`` so cell
    interiors don't pollute the membrane bucket.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100); got {percentile}")
    img = np.asarray(image)
    threshold = float(np.percentile(img, percentile))
    raw = img > threshold
    if exclude is not None:
        if exclude.shape != img.shape:
            raise ValueError(
                f"exclude shape {exclude.shape} ≠ image {img.shape}"
            )
        raw = raw & (~exclude.astype(bool))
    labels, n = ndimage.label(raw)
    if n == 0:
        return np.zeros_like(raw, dtype=bool)
    sizes = ndimage.sum(raw, labels, range(1, n + 1))
    keep = np.where(sizes >= min_area_px)[0] + 1
    return np.isin(labels, keep)


def _default_background(images: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        name: float(np.percentile(img, _DEFAULT_BG_PERCENTILE))
        for name, img in images.items()
    }


def measure_leak(
    channels: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    background: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Build N×N leak matrix from per-channel images + per-channel masks.

    Returns ``(K, channel_order, backgrounds_used)``.

    ``K[i, j]`` is "the fraction of channel-i's signal that leaks
    into channel j", computed as

    .. code-block:: text

        K[i, j] = (channels[j][masks[i]].mean() - bg[j]) /
                  (channels[i][masks[i]].mean() - bg[i])

    Diagonal entries are exactly 1.0 (definition). NaN-safe: when
    the in-channel signal at the i-mask is ≤ 0 (no useful pixels),
    the row is NaN rather than crashing.

    ``background=None`` → 20th-percentile per channel (ch614 r1
    policy). Pass an explicit dict to override.
    """
    if not channels:
        raise ValueError("channels is empty")
    keys = list(channels.keys())
    if set(keys) != set(masks.keys()):
        raise ValueError(
            f"channels keys {set(keys)} ≠ masks keys {set(masks.keys())}"
        )
    bg = dict(background) if background is not None else _default_background(channels)

    n = len(keys)
    K = np.zeros((n, n), dtype=float)
    for i, ki in enumerate(keys):
        mask_i = masks[ki].astype(bool)
        if mask_i.shape != channels[ki].shape:
            raise ValueError(
                f"mask[{ki!r}] shape {mask_i.shape} ≠ image {channels[ki].shape}"
            )
        if not mask_i.any():
            K[i, :] = np.nan
            K[i, i] = 1.0
            continue
        in_signal = float(channels[ki][mask_i].mean()) - bg[ki]
        if in_signal <= 0:
            K[i, :] = np.nan
            K[i, i] = 1.0
            continue
        for j, kj in enumerate(keys):
            if i == j:
                K[i, i] = 1.0
                continue
            cross = float(channels[kj][mask_i].mean()) - bg[kj]
            K[i, j] = cross / in_signal
    return K, keys, bg


def unmix(observed: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Solve ``K @ true = observed`` via ``np.linalg.solve``.

    Use after :func:`measure_leak` has built K. ``observed`` is the
    raw N-channel signal (per pixel, per ROI, or per cell — the
    caller decides). Singular K raises ``ValueError``.
    """
    obs = np.asarray(observed, dtype=float)
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square; got shape {K.shape}")
    if obs.shape[-1] != K.shape[0]:
        raise ValueError(
            f"observed last-axis {obs.shape[-1]} ≠ K size {K.shape[0]}"
        )
    try:
        return np.linalg.solve(K, obs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"K is singular — channels are linearly dependent: {exc}"
        ) from exc
