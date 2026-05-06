"""Drift detection and correction for timelapse microscopy.

Patterns extracted from Ch183, Ch189 (stage drift).

Two complementary methods:
  - Phase/cross-correlation: pixel-level, subpixel accurate, works on any image
  - Centroid tracking: cell-level, robust to intensity changes, needs detectable cells

Key lesson (Ch183): Cross-correlation on BF/membrane is most reliable.
Phase correlation (OpenCV) is fast but can be noisy for small drifts.
Centroid-based tracking with Hungarian matching gives interpretable results
but requires consistent cell detection across frames.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def phase_correlate(ref, target):
    """Measure (dx, dy) translation from ref to target using OpenCV phase correlation.

    Uses Hanning window to reduce edge effects.

    Args:
        ref: reference image (2D array)
        target: target image (2D array, same shape)

    Returns:
        (dx, dy) float tuple — positive means target shifted right/down vs ref
    """
    ref_f = ref.astype(np.float64)
    tgt_f = target.astype(np.float64)
    h, w = ref_f.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_64F)
    (dx, dy), response = cv2.phaseCorrelate(ref_f, tgt_f, window)
    return float(dx), float(dy)


def fft_cross_correlate(ref, target):
    """Measure (dx, dy) translation using FFT cross-correlation with subpixel refinement.

    More robust than phase correlation for small drifts. Uses Hanning window
    and parabolic peak interpolation for subpixel accuracy.

    Args:
        ref: reference image (2D array)
        target: target image (2D array, same shape)

    Returns:
        (dx, dy) float tuple — positive means target shifted right/down vs ref
    """
    ref_f = ref.astype(np.float64)
    tgt_f = target.astype(np.float64)
    ref_f -= ref_f.mean()
    tgt_f -= tgt_f.mean()

    h, w = ref_f.shape
    wy = np.hanning(h)
    wx = np.hanning(w)
    win = np.outer(wy, wx)
    ref_f *= win
    tgt_f *= win

    F_ref = np.fft.fft2(ref_f)
    F_tgt = np.fft.fft2(tgt_f)
    cross = F_tgt * np.conj(F_ref)
    corr = np.fft.ifft2(cross).real

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    iy, ix = peak_idx
    py = iy - h if iy > h // 2 else iy
    px = ix - w if ix > w // 2 else ix
    dy, dx = float(py), float(px)

    # Subpixel parabolic refinement
    if 1 <= iy <= h - 2:
        y_vals = [corr[iy - 1, ix], corr[iy, ix], corr[iy + 1, ix]]
        denom = 2 * (2 * y_vals[1] - y_vals[0] - y_vals[2])
        if abs(denom) > 1e-10:
            dy = py + (y_vals[0] - y_vals[2]) / denom

    if 1 <= ix <= w - 2:
        x_vals = [corr[iy, ix - 1], corr[iy, ix], corr[iy, ix + 1]]
        denom = 2 * (2 * x_vals[1] - x_vals[0] - x_vals[2])
        if abs(denom) > 1e-10:
            dx = px + (x_vals[0] - x_vals[2]) / denom

    return float(dx), float(dy)


def centroid_drift(ref_centroids, tgt_centroids, max_dist=50):
    """Measure drift via median displacement of matched centroids (Hungarian).

    Args:
        ref_centroids: list/array of (x, y) positions in reference frame
        tgt_centroids: list/array of (x, y) positions in target frame
        max_dist: maximum matching distance (reject outliers)

    Returns:
        (dx, dy) float tuple — median displacement of matched cells
    """
    if not len(ref_centroids) or not len(tgt_centroids):
        return 0.0, 0.0

    ref = np.asarray(ref_centroids, dtype=float)
    tgt = np.asarray(tgt_centroids, dtype=float)

    cost = np.zeros((len(ref), len(tgt)))
    for i in range(len(ref)):
        for j in range(len(tgt)):
            cost[i, j] = np.sqrt((ref[i, 0] - tgt[j, 0]) ** 2 + (ref[i, 1] - tgt[j, 1]) ** 2)

    ri, ci = linear_sum_assignment(cost)

    dxs, dys = [], []
    for r, c in zip(ri, ci, strict=False):
        if cost[r, c] < max_dist:
            dxs.append(float(tgt[c, 0] - ref[r, 0]))
            dys.append(float(tgt[c, 1] - ref[r, 1]))

    if dxs:
        return float(np.median(dxs)), float(np.median(dys))
    return 0.0, 0.0


def measure_drift_timelapse(frames, method="fft", ref_frame=0):
    """Measure drift for each frame relative to a reference.

    Args:
        frames: list of 2D images (timelapse)
        method: 'fft' (cross-correlation), 'phase' (OpenCV phase), or 'both'
        ref_frame: index of reference frame (default: 0)

    Returns:
        list of dicts with frame, dx, dy, magnitude per timepoint
    """
    ref = frames[ref_frame]
    func = fft_cross_correlate if method == "fft" else phase_correlate

    results = []
    for i, frame in enumerate(frames):
        if i == ref_frame:
            results.append({"frame": i, "dx": 0.0, "dy": 0.0, "magnitude": 0.0})
            continue

        if method == "both":
            dx_f, dy_f = fft_cross_correlate(ref, frame)
            dx_p, dy_p = phase_correlate(ref, frame)
            # Average both methods
            dx = (dx_f + dx_p) / 2
            dy = (dy_f + dy_p) / 2
        else:
            dx, dy = func(ref, frame)

        mag = float(np.sqrt(dx**2 + dy**2))
        results.append(
            {
                "frame": i,
                "dx": round(float(dx), 3),
                "dy": round(float(dy), 3),
                "magnitude": round(mag, 3),
            }
        )

    return results


def measure_drift_incremental(frames, method="fft"):
    """Measure frame-to-frame drift and accumulate.

    Useful when absolute correlation fails for large total drifts.

    Args:
        frames: list of 2D images (timelapse)
        method: 'fft' or 'phase'

    Returns:
        list of dicts with frame, dx_inc, dy_inc, dx_cum, dy_cum, magnitude
    """
    func = fft_cross_correlate if method == "fft" else phase_correlate
    cum_dx, cum_dy = 0.0, 0.0
    results = [
        {"frame": 0, "dx_inc": 0.0, "dy_inc": 0.0, "dx_cum": 0.0, "dy_cum": 0.0, "magnitude": 0.0}
    ]

    for i in range(1, len(frames)):
        dx, dy = func(frames[i - 1], frames[i])
        cum_dx += dx
        cum_dy += dy
        mag = float(np.sqrt(cum_dx**2 + cum_dy**2))
        results.append(
            {
                "frame": i,
                "dx_inc": round(float(dx), 3),
                "dy_inc": round(float(dy), 3),
                "dx_cum": round(float(cum_dx), 3),
                "dy_cum": round(float(cum_dy), 3),
                "magnitude": round(mag, 3),
            }
        )

    return results


def correct_drift(image, dx, dy):
    """Shift image by (-dx, -dy) to correct for measured drift.

    Args:
        image: 2D array
        dx, dy: measured drift (will be inverted for correction)

    Returns:
        corrected image (same shape, zero-padded edges)
    """
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
