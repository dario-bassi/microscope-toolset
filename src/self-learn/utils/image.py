"""Image preprocessing utilities.

Common operations needed at the start of most analysis pipelines:
grayscale conversion, normalization, and contrast adjustment.
"""

import numpy as np


def to_grayscale(img):
    """Convert an image to 2D grayscale float64.

    Handles:
    - Already 2D: returned as float64.
    - 3-channel (H, W, 3): luminance-weighted conversion (ITU-R BT.601).
    - 4-channel (H, W, 4): drops alpha, then converts RGB.
    - (3, H, W) or (4, H, W): channel-first layout, transposed first.

    Args:
        img: numpy array, 2D or 3D.

    Returns:
        2D float64 array.
    """
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

    # Channel-first layout: (C, H, W) → (H, W, C)
    if arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    c = arr.shape[2]
    if c == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] == 3:
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    raise ValueError(f"Unsupported channel count: {c}")


def normalize(img, pmin=0, pmax=100):
    """Normalize image intensity to [0, 1] using percentile clipping.

    Args:
        img: 2D numpy array.
        pmin: Lower percentile for clipping (0-100).
        pmax: Upper percentile for clipping (0-100).

    Returns:
        2D float64 array with values in [0, 1].
    """
    arr = np.asarray(img, dtype=np.float64)
    lo = np.percentile(arr, pmin)
    hi = np.percentile(arr, pmax)
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def auto_contrast(img, clip_pct=1.0):
    """Stretch image contrast by clipping extreme percentiles.

    Maps the [clip_pct, 100-clip_pct] intensity range to [0, 255] uint8.
    Useful for display/saving preview images.

    Args:
        img: 2D numpy array.
        clip_pct: Percentile to clip at each end (default 1%).

    Returns:
        2D uint8 array with enhanced contrast.
    """
    normed = normalize(img, pmin=clip_pct, pmax=100 - clip_pct)
    return (normed * 255).astype(np.uint8)
