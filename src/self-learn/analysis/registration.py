"""Image registration for timelapse stabilization.

Corrects translational drift between frames in a time series using
FFT-based phase correlation. Works with any 2D grayscale images.

Functions:
    register_translation  -- Find translation shift between two images
    register_stack        -- Register all frames to reference frame
    apply_shift           -- Apply integer translation to an image
    compute_drift         -- Extract drift trajectory from a stack
    stabilize_stack       -- Register + apply shifts to produce stabilized stack
"""

import numpy as np


def register_translation(reference, moving):
    """Find translation shift to align moving image to reference.

    Uses FFT phase correlation for sub-pixel accuracy.

    Args:
        reference: 2D reference image.
        moving: 2D image to align.

    Returns:
        dict with:
            shift_y: Vertical shift (pixels).
            shift_x: Horizontal shift (pixels).
            confidence: Peak-to-noise ratio (higher = more reliable).
    """
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)

    if ref.ndim != 2 or mov.ndim != 2:
        raise ValueError("Images must be 2D")

    # Crop to common size if needed
    h = min(ref.shape[0], mov.shape[0])
    w = min(ref.shape[1], mov.shape[1])
    ref = ref[:h, :w]
    mov = mov[:h, :w]

    if h < 4 or w < 4:
        return {'shift_y': 0, 'shift_x': 0, 'confidence': 0.0}

    # Subtract mean to reduce DC component
    ref = ref - ref.mean()
    mov = mov - mov.mean()

    # Apply Hanning window to reduce edge effects
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = wy[:, None] * wx[None, :]
    ref = ref * window
    mov = mov * window

    # FFT cross-power spectrum
    # Convention: peak at (dy,dx) means moving is displaced by (dy,dx) from ref
    F_ref = np.fft.fft2(ref)
    F_mov = np.fft.fft2(mov)
    cross = F_mov * np.conj(F_ref)
    denom = np.abs(cross)
    denom[denom < 1e-10] = 1e-10
    normalized = cross / denom

    # Inverse FFT
    corr = np.real(np.fft.ifft2(normalized))

    # Find peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[peak_idx])

    # Convert to signed shift
    dy = peak_idx[0]
    dx = peak_idx[1]
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w

    # Confidence: peak / mean
    mean_corr = float(np.mean(np.abs(corr)))
    confidence = peak_val / max(mean_corr, 1e-10)

    return {
        'shift_y': int(dy),
        'shift_x': int(dx),
        'confidence': round(float(confidence), 2),
    }


def register_stack(images, reference_index=0, max_shift=None):
    """Register all frames in a stack to a reference frame.

    Args:
        images: List of 2D arrays, or 3D array (T, H, W).
        reference_index: Index of the reference frame (default: 0).
        max_shift: Maximum allowed shift in pixels. Shifts exceeding
            this are clamped. None = no limit.

    Returns:
        dict with:
            shifts: List of (dy, dx) tuples, one per frame.
            confidences: List of confidence values.
            max_drift: Maximum absolute drift from reference.
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        stack = [images[i] for i in range(images.shape[0])]
    else:
        stack = list(images)

    n = len(stack)
    if n == 0:
        return {'shifts': [], 'confidences': [], 'max_drift': 0.0}

    ref = stack[reference_index]
    shifts = []
    confidences = []

    for i in range(n):
        if i == reference_index:
            shifts.append((0, 0))
            confidences.append(1.0)
            continue

        result = register_translation(ref, stack[i])
        dy, dx = result['shift_y'], result['shift_x']

        if max_shift is not None:
            dy = max(-max_shift, min(max_shift, dy))
            dx = max(-max_shift, min(max_shift, dx))

        shifts.append((dy, dx))
        confidences.append(result['confidence'])

    # Max drift
    max_drift = 0.0
    for dy, dx in shifts:
        drift = np.sqrt(dy**2 + dx**2)
        max_drift = max(max_drift, drift)

    return {
        'shifts': shifts,
        'confidences': confidences,
        'max_drift': round(float(max_drift), 2),
    }


def apply_shift(image, shift, fill_value=0):
    """Apply integer translation shift to an image.

    Args:
        image: 2D array.
        shift: (dy, dx) tuple.
        fill_value: Value for uncovered pixels.

    Returns:
        Shifted 2D array (same shape as input).
    """
    img = np.asarray(image)
    dy, dx = int(shift[0]), int(shift[1])

    if dy == 0 and dx == 0:
        return img.copy()

    result = np.full_like(img, fill_value)
    h, w = img.shape[:2]

    # Source and destination slices
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)

    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]

    return result


def compute_drift(images, reference_index=0):
    """Extract drift trajectory from a timelapse stack.

    Computes cumulative drift relative to reference frame, and
    estimates drift rate (pixels/frame).

    Args:
        images: List of 2D arrays, or 3D array (T, H, W).
        reference_index: Reference frame index.

    Returns:
        dict with:
            drift_y: 1D array of Y drift per frame.
            drift_x: 1D array of X drift per frame.
            drift_magnitude: 1D array of total drift magnitude.
            drift_rate: Mean drift in pixels/frame.
            total_drift: Total drift from first to last frame.
    """
    reg = register_stack(images, reference_index=reference_index)
    shifts = reg['shifts']
    n = len(shifts)

    if n == 0:
        return {
            'drift_y': np.array([]),
            'drift_x': np.array([]),
            'drift_magnitude': np.array([]),
            'drift_rate': 0.0,
            'total_drift': 0.0,
        }

    dy = np.array([s[0] for s in shifts], dtype=np.float64)
    dx = np.array([s[1] for s in shifts], dtype=np.float64)
    mag = np.sqrt(dy**2 + dx**2)

    # Drift rate: total displacement / number of frames
    total = float(mag[-1]) if n > 1 else 0.0
    rate = total / max(n - 1, 1)

    return {
        'drift_y': dy,
        'drift_x': dx,
        'drift_magnitude': mag,
        'drift_rate': round(rate, 3),
        'total_drift': round(total, 2),
    }


def stabilize_stack(images, reference_index=0, max_shift=None, fill_value=0):
    """Register and shift all frames to produce a stabilized stack.

    Args:
        images: List of 2D arrays, or 3D array (T, H, W).
        reference_index: Reference frame index.
        max_shift: Maximum allowed shift per frame.
        fill_value: Fill value for uncovered pixels.

    Returns:
        dict with:
            stabilized: List of stabilized 2D arrays.
            shifts: List of (dy, dx) tuples applied.
            max_drift: Maximum drift corrected.
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        stack = [images[i] for i in range(images.shape[0])]
    else:
        stack = list(images)

    if not stack:
        return {'stabilized': [], 'shifts': [], 'max_drift': 0.0}

    reg = register_stack(stack, reference_index=reference_index,
                         max_shift=max_shift)

    stabilized = []
    for i, img in enumerate(stack):
        dy, dx = reg['shifts'][i]
        # Apply negative shift to undo drift
        corrected = apply_shift(img, (-dy, -dx), fill_value=fill_value)
        stabilized.append(corrected)

    return {
        'stabilized': stabilized,
        'shifts': reg['shifts'],
        'max_drift': reg['max_drift'],
    }
