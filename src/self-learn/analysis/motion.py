"""Timelapse motion analysis: kymographs, optical flow, and contraction.

Provides tools for analyzing bulk tissue motion, migration patterns,
and periodic contractions from timelapse image sequences.

Key functions:
    kymograph       -- Extract a kymograph along a line across frames
    optical_flow    -- Dense optical flow between consecutive frames
    contraction_amplitude -- Measure periodic contraction from motion field
    migration_front -- Track the leading edge of a migrating cell sheet
"""

import numpy as np
from scipy import ndimage


def kymograph(frames, y=None, x_range=None, axis='horizontal', width=3):
    """Extract a kymograph (space-time image) from a timelapse.

    A kymograph shows how intensity along a line changes over time.
    Each row (or column) corresponds to one timepoint.

    Args:
        frames: (N, H, W) array or list of 2D arrays.
        y: Y-coordinate of the line (for horizontal kymograph).
            If None, uses the center row.
        x_range: (x_start, x_end) to crop horizontally. If None, full width.
        axis: 'horizontal' extracts a row, 'vertical' extracts a column.
            For vertical, y is reinterpreted as x-coordinate.
        width: Number of rows/columns to average for noise reduction.

    Returns:
        dict with:
            image: 2D array (time × space) — the kymograph
            spatial_axis: pixel positions along the line
            time_axis: frame indices
    """
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, h, w = frames.shape

    if axis == 'horizontal':
        if y is None:
            y = h // 2
        half_w = width // 2
        y_lo = max(0, int(y) - half_w)
        y_hi = min(h, int(y) + half_w + 1)

        if x_range is not None:
            x0, x1 = int(x_range[0]), int(x_range[1])
        else:
            x0, x1 = 0, w

        kymo = frames[:, y_lo:y_hi, x0:x1].mean(axis=1)
        spatial = np.arange(x0, x1)

    elif axis == 'vertical':
        x = y if y is not None else w // 2  # reinterpret y as x
        half_w = width // 2
        x_lo = max(0, int(x) - half_w)
        x_hi = min(w, int(x) + half_w + 1)

        if x_range is not None:
            y0, y1 = int(x_range[0]), int(x_range[1])
        else:
            y0, y1 = 0, h

        kymo = frames[:, y0:y1, x_lo:x_hi].mean(axis=2)
        spatial = np.arange(y0, y1)
    else:
        raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}")

    return {
        'image': kymo,
        'spatial_axis': spatial,
        'time_axis': np.arange(n_frames),
    }


def optical_flow(frame0, frame1, window=15):
    """Compute dense optical flow between two frames using Lucas-Kanade.

    Uses a block-matching approach with normalized cross-correlation.
    For more accurate flow, consider using cv2.calcOpticalFlowFarneback
    if OpenCV is available.

    Args:
        frame0: First frame (2D array).
        frame1: Second frame (2D array).
        window: Window size for block matching.

    Returns:
        dict with:
            dx: 2D array of horizontal displacement per pixel
            dy: 2D array of vertical displacement per pixel
            magnitude: 2D array of displacement magnitude
            mean_magnitude: scalar mean displacement
    """
    f0 = np.asarray(frame0, dtype=np.float64)
    f1 = np.asarray(frame1, dtype=np.float64)
    h, w = f0.shape

    # Simple block-matching on downsampled grid
    step = max(window // 2, 1)
    search_range = window

    # Sample points on a grid
    grid_y = np.arange(window, h - window, step)
    grid_x = np.arange(window, w - window, step)

    dx_grid = np.zeros((len(grid_y), len(grid_x)))
    dy_grid = np.zeros((len(grid_y), len(grid_x)))

    half = window // 2
    for iy, cy in enumerate(grid_y):
        for ix, cx in enumerate(grid_x):
            template = f0[cy - half:cy + half + 1, cx - half:cx + half + 1]
            t_mean = template.mean()
            t_std = template.std()
            if t_std < 1e-6:
                continue

            best_score = -1
            best_dy, best_dx = 0, 0

            for dy_s in range(-search_range, search_range + 1, 2):
                for dx_s in range(-search_range, search_range + 1, 2):
                    sy = cy + dy_s
                    sx = cx + dx_s
                    if sy - half < 0 or sy + half + 1 > h:
                        continue
                    if sx - half < 0 or sx + half + 1 > w:
                        continue

                    patch = f1[sy - half:sy + half + 1, sx - half:sx + half + 1]
                    p_std = patch.std()
                    if p_std < 1e-6:
                        continue

                    # Normalized cross-correlation
                    ncc = np.mean((template - t_mean) * (patch - patch.mean())) / (t_std * p_std)
                    if ncc > best_score:
                        best_score = ncc
                        best_dy = dy_s
                        best_dx = dx_s

            dx_grid[iy, ix] = best_dx
            dy_grid[iy, ix] = best_dy

    # Interpolate to full resolution
    from scipy.interpolate import RegularGridInterpolator

    interp_dx = RegularGridInterpolator(
        (grid_y.astype(float), grid_x.astype(float)), dx_grid,
        method='linear', bounds_error=False, fill_value=0.0
    )
    interp_dy = RegularGridInterpolator(
        (grid_y.astype(float), grid_x.astype(float)), dy_grid,
        method='linear', bounds_error=False, fill_value=0.0
    )

    yy, xx = np.mgrid[:h, :w]
    pts = np.stack([yy.ravel().astype(float), xx.ravel().astype(float)], axis=-1)

    dx_full = interp_dx(pts).reshape(h, w)
    dy_full = interp_dy(pts).reshape(h, w)
    mag = np.sqrt(dx_full**2 + dy_full**2)

    return {
        'dx': dx_full,
        'dy': dy_full,
        'magnitude': mag,
        'mean_magnitude': float(mag.mean()),
    }


def contraction_amplitude(frames, roi_mask=None, dt=1.0):
    """Measure periodic contraction amplitude from a timelapse.

    Computes frame-to-frame intensity variance within an ROI as a proxy
    for tissue motion/contraction. Useful for cardiac beating analysis.

    Args:
        frames: (N, H, W) array.
        roi_mask: Optional 2D boolean mask to restrict analysis.
        dt: Time between frames (seconds).

    Returns:
        dict with:
            signal: 1D array of motion signal per frame
            times: 1D array of time points
            mean_amplitude: mean of absolute signal
            peak_amplitude: max of absolute signal
    """
    frames = np.asarray(frames, dtype=np.float64)
    n_frames = frames.shape[0]

    signal = np.zeros(n_frames - 1)
    for i in range(1, n_frames):
        diff = frames[i] - frames[i - 1]
        if roi_mask is not None:
            diff = diff[roi_mask]
        signal[i - 1] = float(np.std(diff))

    times = np.arange(n_frames - 1) * dt

    return {
        'signal': signal,
        'times': times,
        'mean_amplitude': float(np.mean(signal)),
        'peak_amplitude': float(np.max(signal)),
    }


def migration_front(frames, axis='horizontal', threshold_method='otsu',
                    direction='left_to_right'):
    """Track the leading edge of a migrating cell sheet over time.

    Detects where the cell sheet boundary is along the specified axis
    in each frame. Useful for wound healing assays.

    Args:
        frames: (N, H, W) array.
        axis: 'horizontal' tracks left-right boundary, 'vertical' tracks up-down.
        threshold_method: 'otsu' or 'mean' for detecting cell vs background.
        direction: 'left_to_right' or 'right_to_left' — which edge to track.

    Returns:
        dict with:
            positions: 1D array of leading edge position per frame
            speeds: 1D array of speed between consecutive frames
            mean_speed: mean migration speed (pixels/frame)
            total_distance: total distance traveled
    """
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, h, w = frames.shape

    positions = np.zeros(n_frames)

    for i in range(n_frames):
        img = frames[i]

        if threshold_method == 'otsu':
            from skimage.filters import threshold_otsu
            try:
                thresh = threshold_otsu(img)
            except ValueError:
                thresh = img.mean()
        else:
            thresh = img.mean()

        binary = img > thresh

        if axis == 'horizontal':
            # Project along y-axis to get x-profile
            profile = binary.mean(axis=0)  # fraction of rows with cells at each x

            if direction == 'left_to_right':
                # Find rightmost x where cells are present
                cell_cols = np.where(profile > 0.2)[0]
                positions[i] = float(cell_cols[-1]) if len(cell_cols) > 0 else 0
            else:
                cell_cols = np.where(profile > 0.2)[0]
                positions[i] = float(cell_cols[0]) if len(cell_cols) > 0 else w

        elif axis == 'vertical':
            profile = binary.mean(axis=1)

            if direction == 'left_to_right':
                cell_rows = np.where(profile > 0.2)[0]
                positions[i] = float(cell_rows[-1]) if len(cell_rows) > 0 else 0
            else:
                cell_rows = np.where(profile > 0.2)[0]
                positions[i] = float(cell_rows[0]) if len(cell_rows) > 0 else h

    speeds = np.abs(np.diff(positions))
    total = float(np.abs(positions[-1] - positions[0]))

    return {
        'positions': positions,
        'speeds': speeds,
        'mean_speed': float(np.mean(speeds)),
        'total_distance': total,
    }
