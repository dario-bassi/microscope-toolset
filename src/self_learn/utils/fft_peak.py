"""FFT-peak primitives: 2D power-spectrum peak + bandpass + orientation.

Pure-data utilities — no live core required for tests. Mirrors the
:mod:`src.core.utils.spectral_leak` template.

Lifted from :mod:`scratch.solve_615` (counter=106) where the inline
``fft2 + bandpass + argmax`` block was the core of the ch615 r1 →
10/10 win. This module is the agent-portable substrate so future
periodic-structure solves don't re-invent it.

Composes nothing (numpy only). Useful for:

- SIM orientation extraction (ch615 r1 — single-frame; ch617+ may
  exercise the 9-frame collection path).
- Full SR-SIM reconstruction (combine the 9 FFT spectra).
- Detecting OTHER periodic structures (sarcomeres, focal adhesions,
  microvilli) via the same FFT-peak-then-bandpass approach.

This module is intentionally NOT registered in
:mod:`src.core.utils.auto_recipe`. Frequency content isn't yet a
sample-classifier signal (the classifier looks at spatial
features); same rationale as :mod:`src.core.utils.spectral_leak`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FFTPeak:
    """One FFT magnitude peak relative to the FOV centre.

    Attributes:
      kx, ky: Signed wavevector components in pixels relative to the
        DC peak (i.e. shifted-FFT pixel offset from the image
        centre). Image-coords convention: y increases downward.
      k_mag: ``hypot(kx, ky)``.
      angle_rad: ``atan2(ky, kx) % pi`` — folds the ±k symmetry of
        the real-valued FFT (every real grating has peaks at both
        ``+k`` and ``-k``); reported in [0, π).
      angle_deg: ``angle_rad * 180/pi``, in [0, 180).
      wavelength_px: ``max(shape) / k_mag`` — exact for square FOVs;
        approximate (within ~aspect-ratio factor) for non-square.
        Use as a sanity-check against the expected fringe period.
    """

    kx: float
    ky: float
    k_mag: float
    angle_rad: float
    angle_deg: float
    wavelength_px: float


def fft_magnitude(
    image: np.ndarray,
    *,
    subtract_mean: bool = True,
) -> np.ndarray:
    """Shifted 2D FFT magnitude, optionally bg-subtracted.

    ``subtract_mean=True`` (default) subtracts the image mean before
    fft2 so the DC peak doesn't dominate the magnitude spectrum.
    Without this step, even a strong off-DC peak gets dwarfed and
    the bandpass mask has to be large to beat it.
    """
    img = np.asarray(image, dtype=float)
    if subtract_mean:
        img = img - float(img.mean())
    return np.fft.fftshift(np.abs(np.fft.fft2(img)))


def annular_mask(
    shape: Tuple[int, int],
    *,
    r_lo: float,
    r_hi: float,
) -> np.ndarray:
    """Boolean annulus around the FOV centre: ``r_lo ≤ r ≤ r_hi``.

    Use to exclude the DC peak (``r < r_lo``) and the corner /
    edge artifacts (``r > r_hi``) before argmax.
    """
    if not 0 <= r_lo < r_hi:
        raise ValueError(f"need 0 ≤ r_lo < r_hi (got {r_lo}, {r_hi})")
    H, W = shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.hypot(yy - cy, xx - cx)
    return (rr >= r_lo) & (rr <= r_hi)


def _peak_to_dataclass(
    py: int, px: int, shape: Tuple[int, int],
) -> FFTPeak:
    H, W = shape
    cy, cx = H // 2, W // 2
    ky = float(py - cy)
    kx = float(px - cx)
    k_mag = float(np.hypot(kx, ky))
    if k_mag == 0:
        raise ValueError("found peak at DC — annular mask did not exclude it")
    angle_rad = float(np.arctan2(ky, kx)) % float(np.pi)
    angle_deg = float(np.degrees(angle_rad))
    wavelength_px = float(max(H, W)) / k_mag
    return FFTPeak(
        kx=kx, ky=ky, k_mag=k_mag,
        angle_rad=angle_rad, angle_deg=angle_deg,
        wavelength_px=wavelength_px,
    )


def find_fft_peak(
    image: np.ndarray,
    *,
    r_lo: float = 15.0,
    r_hi: Optional[float] = None,
    subtract_mean: bool = True,
    magnitude: Optional[np.ndarray] = None,
) -> FFTPeak:
    """The ch615 r1 win packaged: bg-sub → fft2 → bandpass → argmax → angle.

    Pass a precomputed ``magnitude`` to skip the fft2 step (useful
    when iterating multiple peak searches on the same image).
    ``r_hi=None`` resolves to ``min(shape) / 2 - 5`` — i.e. just
    inside the inner-square boundary of a square FOV.

    Raises ``ValueError`` on degenerate input (all-zero magnitude
    inside the annular mask, image entirely flat after mean-sub).
    """
    img = np.asarray(image)
    H, W = img.shape[:2]
    if magnitude is None:
        mag = fft_magnitude(img, subtract_mean=subtract_mean)
    else:
        mag = np.asarray(magnitude, dtype=float)
        if mag.shape != (H, W):
            raise ValueError(
                f"magnitude shape {mag.shape} ≠ image shape {(H, W)}"
            )
    r_hi_resolved = float(r_hi) if r_hi is not None else max(min(H, W) / 2 - 5, r_lo + 1)
    mask = annular_mask((H, W), r_lo=r_lo, r_hi=r_hi_resolved)
    masked = np.where(mask, mag, 0.0)
    if masked.max() <= 0:
        raise ValueError(
            "no FFT power inside the annular mask — image flat or "
            "mask too restrictive"
        )
    py, px = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return _peak_to_dataclass(int(py), int(px), (H, W))


def find_top_n_fft_peaks(
    image: np.ndarray,
    n: int,
    *,
    r_lo: float = 15.0,
    r_hi: Optional[float] = None,
    suppression_radius: Optional[float] = None,
    subtract_mean: bool = True,
) -> List[FFTPeak]:
    """Greedy top-N peak finder with ±k symmetric suppression.

    Each iteration: argmax → record → zero a ``suppression_radius``
    disk around the found pixel AND its ``(2*cy - py, 2*cx - px)``
    symmetric partner so the same orientation isn't reported twice.

    Stops early if the remaining masked spectrum is empty. Returns
    peaks in descending magnitude order.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0; got {n}")
    img = np.asarray(image)
    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    mag = fft_magnitude(img, subtract_mean=subtract_mean).copy()
    r_hi_resolved = float(r_hi) if r_hi is not None else max(min(H, W) / 2 - 5, r_lo + 1)
    sup = float(suppression_radius) if suppression_radius is not None else float(r_lo)
    mask = annular_mask((H, W), r_lo=r_lo, r_hi=r_hi_resolved)

    peaks: List[FFTPeak] = []
    yy, xx = np.ogrid[:H, :W]
    for _ in range(n):
        masked = np.where(mask, mag, 0.0)
        if masked.max() <= 0:
            break
        py, px = np.unravel_index(int(np.argmax(masked)), masked.shape)
        peaks.append(_peak_to_dataclass(int(py), int(px), (H, W)))
        # Zero a disk around the found pixel + its symmetric partner.
        for cy0, cx0 in ((py, px), (2 * cy - py, 2 * cx - px)):
            disk = np.hypot(yy - cy0, xx - cx0) <= sup
            mag[disk] = 0.0
    return peaks
