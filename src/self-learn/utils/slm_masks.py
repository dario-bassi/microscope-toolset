"""SLM-mask builders: circle / gaussian / ring around named centroids.

Pure-data utilities — no live core required for tests. Small,
focused, vectorised over centre count.

Lifted from :mod:`scratch.solve_613` (counter=95) where the inline
Python loop ``for i in TARGET_CELLS: mask[circle] = 255`` was the
core of the ch613 r1 → 10/10 win. This module is the agent-portable
substrate so future SLM/optogenetics solves don't re-invent it.

Composes nothing (pure numpy).

Convention notes called out in the public API:

- **Out-of-bounds centres are clipped, not rejected.** A centroid at
  ``(550, 200)`` on a ``(512, 512)`` SLM contributes only the
  in-FOV portion of its disk. Partial coverage is honest and
  matches what the SLM physically does. (The
  :mod:`src.recipes.cybergenetic_per_cell_control` recipe uses a
  more aggressive policy — it skips out-of-bounds centroids
  outright; that's a sibling-but-different choice.)
- **Gaussian normalisation is peak-not-area.** ``peak`` is the
  centre value of an isolated centre. Overlapping centres sum
  then clip to ``peak``, matching SLM physics: pixel value caps at
  ``MAX_VAL``, overlap adds light up to the cap.
- **Single scalar radius for now.** No per-cell variable radii — the
  ch613 call sites all use a uniform value. Generalising the
  broadcast to ``(N, 1, 1)`` radii is a 1-line change when a future
  challenge demands it.

This module is intentionally NOT registered in
:mod:`src.core.utils.auto_recipe`: SLM availability + centroid
geometry are scenario-level signals, not first-contact-image signals.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np


DEFAULT_DTYPE = "uint8"
MAX_VAL = 255

CentersXY = Union[np.ndarray, Iterable[Tuple[float, float]]]


def _broadcast_r2(
    shape: Tuple[int, int],
    centers_xy: CentersXY,
) -> np.ndarray:
    """Squared-distance stack: ``r²[k, y, x] = (x - cx_k)² + (y - cy_k)²``.

    Returned shape is ``(N, H, W)``. Empty centre list returns an
    array of shape ``(0, H, W)`` so downstream ``.any(axis=0)`` /
    ``.sum(axis=0)`` reductions still work.
    """
    H, W = shape
    centers = np.asarray(list(centers_xy), dtype=float)
    if centers.size == 0:
        centers = centers.reshape(0, 2)
    elif centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(
            f"centers_xy must be (N, 2); got shape {centers.shape!r}"
        )
    yy, xx = np.ogrid[:H, :W]
    cx = centers[:, 0].reshape(-1, 1, 1)
    cy = centers[:, 1].reshape(-1, 1, 1)
    return (xx - cx) ** 2 + (yy - cy) ** 2  # (N, H, W)


def circle_mask(
    shape: Tuple[int, int],
    centers_xy: CentersXY,
    radius_px: float,
    *,
    value: int = MAX_VAL,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Boolean-OR of disks at ``centers_xy`` with shared ``radius_px``.

    ``shape`` is ``(H, W)`` (matching numpy convention). Pixels with
    ``r² ≤ radius_px²`` for ANY centre take ``value``, others 0.
    Vectorises over centre count via a single ``(N, H, W)`` broadcast.
    """
    if radius_px < 0:
        raise ValueError(f"radius_px must be ≥ 0 (got {radius_px})")
    r2 = _broadcast_r2(shape, centers_xy)
    if r2.shape[0] == 0:
        inside = np.zeros(shape, dtype=bool)
    else:
        inside = (r2 <= radius_px * radius_px).any(axis=0)
    return np.where(inside, value, 0).astype(dtype)


def gaussian_mask(
    shape: Tuple[int, int],
    centers_xy: CentersXY,
    sigma_px: float,
    *,
    peak: int = MAX_VAL,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Sum of gaussian peaks at ``centers_xy``, clipped to ``peak``.

    Per-pixel value = ``peak * Σ_k exp(-r²_k / (2 σ²))`` clipped at
    ``peak``. Soft-edged variant of :func:`circle_mask`; useful when
    SLM hardware has graded brightness or to taper edge effects on a
    cell.
    """
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be > 0 (got {sigma_px})")
    r2 = _broadcast_r2(shape, centers_xy)
    if r2.shape[0] == 0:
        return np.zeros(shape, dtype=dtype)
    weights = np.exp(-r2 / (2.0 * sigma_px * sigma_px))  # (N, H, W)
    summed = weights.sum(axis=0) * float(peak)
    return np.clip(summed, 0, peak).astype(dtype)


def ring_mask(
    shape: Tuple[int, int],
    centers_xy: CentersXY,
    inner_px: float,
    outer_px: float,
    *,
    value: int = MAX_VAL,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Annulus of ``[inner_px, outer_px]`` around each centre.

    Useful for cytoplasm-only stim (sparing nuclei). Inside
    ``inner_px`` is 0; ``[inner_px, outer_px]`` is ``value``; beyond
    ``outer_px`` is 0.
    """
    if inner_px < 0 or outer_px <= inner_px:
        raise ValueError(
            f"need 0 ≤ inner_px < outer_px (got {inner_px}, {outer_px})"
        )
    r2 = _broadcast_r2(shape, centers_xy)
    if r2.shape[0] == 0:
        inside = np.zeros(shape, dtype=bool)
    else:
        inside = (
            (r2 >= inner_px * inner_px) & (r2 <= outer_px * outer_px)
        ).any(axis=0)
    return np.where(inside, value, 0).astype(dtype)
