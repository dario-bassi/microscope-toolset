"""Bulk flow and velocity field analysis.

Quantifies fluid flow patterns from timelapse microscopy.
Computes flow fields, velocity profiles, vorticity, and
flow uniformity metrics for microfluidics and perfusion.

Functions:
    compute_flow_field    -- Velocity field from consecutive frames
    flow_statistics       -- Magnitude/direction statistics from flow field
    velocity_profile      -- Cross-sectional velocity profile
    vorticity_map         -- Local rotation from flow field
    flow_uniformity       -- Quantify spatial variation in flow

Vessel flow (kymograph / particle tracking):
    build_kymograph       -- Build kymograph (time × X) from vessel timelapse
    find_vessel_row       -- Find vessel Y row from temporal variance
    kymograph_velocity    -- Measure flow velocity from kymograph slope
    particle_velocity     -- Track dark particles for flow velocity
"""

import numpy as np
from scipy import ndimage


def compute_flow_field(frame1, frame2, block_size=16, search_radius=None):
    """Compute velocity field between two frames using block matching.

    Divides frame1 into blocks and finds best match in frame2 using
    normalized cross-correlation. Returns displacement vectors on a
    regular grid.

    Args:
        frame1: 2D array, first frame.
        frame2: 2D array, second frame.
        block_size: int, size of correlation blocks in pixels.
        search_radius: int or None, max displacement to search.
            Defaults to block_size.

    Returns:
        dict with:
            vx: 2D array, x-displacement per grid point.
            vy: 2D array, y-displacement per grid point.
            magnitude: 2D array, speed at each grid point.
            angle: 2D array, direction in radians.
            grid_y: 1D array, y-coordinates of grid centers.
            grid_x: 1D array, x-coordinates of grid centers.
    """
    frame1 = np.asarray(frame1, dtype=float)
    frame2 = np.asarray(frame2, dtype=float)

    if search_radius is None:
        search_radius = block_size

    h, w = frame1.shape
    half = block_size // 2
    ny = max(1, (h - block_size) // block_size + 1)
    nx = max(1, (w - block_size) // block_size + 1)

    vx = np.zeros((ny, nx))
    vy = np.zeros((ny, nx))
    grid_y = np.zeros(ny)
    grid_x = np.zeros(nx)

    for iy in range(ny):
        cy = min(half + iy * block_size, h - half - 1)
        grid_y[iy] = cy
        for ix in range(nx):
            cx = min(half + ix * block_size, w - half - 1)
            grid_x[ix] = cx

            # Extract block from frame1
            y0 = max(0, cy - half)
            y1 = min(h, cy + half)
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)
            block = frame1[y0:y1, x0:x1]

            if block.std() < 1e-6:
                continue

            # Search in frame2
            best_score = -1
            best_dy, best_dx = 0, 0

            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    sy0 = y0 + dy
                    sy1 = y1 + dy
                    sx0 = x0 + dx
                    sx1 = x1 + dx

                    if sy0 < 0 or sy1 > h or sx0 < 0 or sx1 > w:
                        continue

                    candidate = frame2[sy0:sy1, sx0:sx1]
                    if candidate.shape != block.shape or candidate.std() < 1e-6:
                        continue

                    # Normalized cross-correlation
                    b_norm = block - block.mean()
                    c_norm = candidate - candidate.mean()
                    denom = np.sqrt((b_norm**2).sum() * (c_norm**2).sum())
                    if denom < 1e-10:
                        continue
                    ncc = float((b_norm * c_norm).sum() / denom)

                    if ncc > best_score:
                        best_score = ncc
                        best_dy, best_dx = dy, dx

            vx[iy, ix] = best_dx
            vy[iy, ix] = best_dy

    magnitude = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(vy, vx)

    return {
        'vx': vx,
        'vy': vy,
        'magnitude': magnitude,
        'angle': angle,
        'grid_y': grid_y,
        'grid_x': grid_x,
    }


def flow_statistics(magnitude, angle=None):
    """Compute summary statistics from a flow field.

    Args:
        magnitude: 2D array, speed at each grid point.
        angle: 2D array or None, direction at each grid point (radians).

    Returns:
        dict with:
            mean_speed: float.
            max_speed: float.
            std_speed: float.
            mean_direction: float (radians), circular mean if angle given.
            direction_coherence: float (0-1), how aligned the flow is.
            n_points: int.
    """
    mag = np.asarray(magnitude, dtype=float)
    flat_mag = mag.ravel()

    mean_speed = float(np.mean(flat_mag))
    max_speed = float(np.max(flat_mag)) if flat_mag.size > 0 else 0.0

    result = {
        'mean_speed': round(mean_speed, 4),
        'max_speed': round(max_speed, 4),
        'std_speed': round(float(np.std(flat_mag)), 4),
        'n_points': int(flat_mag.size),
    }

    if angle is not None:
        ang = np.asarray(angle, dtype=float).ravel()
        # Weight directions by magnitude for meaningful circular mean
        weights = flat_mag / (flat_mag.sum() + 1e-10)
        sin_mean = float(np.sum(weights * np.sin(ang)))
        cos_mean = float(np.sum(weights * np.cos(ang)))
        result['mean_direction'] = round(float(np.arctan2(sin_mean, cos_mean)), 4)
        result['direction_coherence'] = round(
            float(np.sqrt(sin_mean**2 + cos_mean**2)), 4
        )
    else:
        result['mean_direction'] = 0.0
        result['direction_coherence'] = 0.0

    return result


def velocity_profile(magnitude, axis='x'):
    """Compute cross-sectional velocity profile.

    Averages speed along one axis to create a 1D profile across
    the perpendicular axis. Useful for parabolic flow profiles
    in microfluidic channels.

    Args:
        magnitude: 2D array, speed field.
        axis: str, 'x' to profile along y (average over x),
              'y' to profile along x (average over y).

    Returns:
        dict with:
            profile: 1D array, mean velocity across channel.
            positions: 1D array, position indices.
            peak_position: int, index of maximum velocity.
            peak_velocity: float.
            profile_width: float, FWHM of velocity profile.
    """
    mag = np.asarray(magnitude, dtype=float)

    if axis == 'x':
        profile = np.mean(mag, axis=1)  # average over columns → profile along rows
    else:
        profile = np.mean(mag, axis=0)  # average over rows → profile along cols

    positions = np.arange(len(profile), dtype=float)
    peak_idx = int(np.argmax(profile))
    peak_vel = float(profile[peak_idx])

    # FWHM
    half_max = peak_vel / 2
    above = profile >= half_max
    if above.any():
        indices = np.where(above)[0]
        width = float(indices[-1] - indices[0] + 1)
    else:
        width = 0.0

    return {
        'profile': profile,
        'positions': positions,
        'peak_position': peak_idx,
        'peak_velocity': round(peak_vel, 4),
        'profile_width': width,
    }


def vorticity_map(vx, vy):
    """Compute vorticity (curl) from velocity field.

    Vorticity = ∂vy/∂x - ∂vx/∂y
    Positive = counter-clockwise rotation.

    Args:
        vx: 2D array, x-velocity component.
        vy: 2D array, y-velocity component.

    Returns:
        dict with:
            vorticity: 2D array, local rotation values.
            mean_vorticity: float.
            max_vorticity: float.
            has_rotation: bool, whether significant rotation detected.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    if vx.shape[0] < 2 or vx.shape[1] < 2:
        return {
            'vorticity': np.zeros_like(vx),
            'mean_vorticity': 0.0,
            'max_vorticity': 0.0,
            'has_rotation': False,
        }

    # Central differences
    dvy_dx = np.gradient(vy, axis=1)
    dvx_dy = np.gradient(vx, axis=0)
    vort = dvy_dx - dvx_dy

    max_mag = float(np.max(np.sqrt(vx**2 + vy**2)))
    threshold = max_mag * 0.1 if max_mag > 0 else 0.1

    return {
        'vorticity': vort,
        'mean_vorticity': round(float(np.mean(np.abs(vort))), 4),
        'max_vorticity': round(float(np.max(np.abs(vort))), 4),
        'has_rotation': bool(np.max(np.abs(vort)) > threshold),
    }


def flow_uniformity(magnitude, angle=None):
    """Quantify spatial uniformity of flow.

    Args:
        magnitude: 2D array, speed field.
        angle: 2D array or None, direction field.

    Returns:
        dict with:
            cv: float, coefficient of variation of speed (0=uniform).
            uniformity_index: float (0-1), 1=perfectly uniform.
            dead_zone_fraction: float, fraction of field with <10% max speed.
            speed_ratio: float, max/mean speed ratio (1=uniform, >1=peaked).
    """
    mag = np.asarray(magnitude, dtype=float)
    flat = mag.ravel()

    mean_v = float(np.mean(flat))
    max_v = float(np.max(flat)) if flat.size > 0 else 0.0
    std_v = float(np.std(flat))

    cv = std_v / mean_v if mean_v > 0 else 0.0
    uniformity = max(0.0, 1.0 - cv)

    dead_threshold = max_v * 0.1 if max_v > 0 else 0.0
    dead_fraction = float(np.mean(flat < dead_threshold))

    speed_ratio = max_v / mean_v if mean_v > 0 else 1.0

    return {
        'cv': round(cv, 4),
        'uniformity_index': round(uniformity, 4),
        'dead_zone_fraction': round(dead_fraction, 4),
        'speed_ratio': round(speed_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Vessel flow: kymograph and particle tracking
# ---------------------------------------------------------------------------

def find_vessel_row(imgs, smooth_sigma=5):
    """Find vessel Y-row from temporal variance peak.

    The vessel row has the highest temporal variance because dark RBCs
    repeatedly pass through, creating large intensity fluctuations.

    Args:
        imgs: array-like of shape (n_frames, H, W) or list of 2D arrays.
        smooth_sigma: Gaussian smoothing sigma for row-std curve.

    Returns:
        dict with:
            vessel_row: int, Y-pixel of peak temporal variance.
            row_std: 1D array, temporal std per row (smoothed).
    """
    frames = np.asarray(imgs, dtype=float)
    if frames.ndim == 2:
        return {'vessel_row': frames.shape[0] // 2, 'row_std': np.zeros(frames.shape[0])}

    # Mean temporal std across columns for each row
    row_std = frames.std(axis=0).mean(axis=1)  # shape: (H,)
    row_std_sm = ndimage.gaussian_filter1d(row_std, sigma=smooth_sigma)
    vessel_row = int(np.argmax(row_std_sm))

    return {
        'vessel_row': vessel_row,
        'row_std': row_std_sm,
    }


def build_kymograph(imgs, vessel_row, half_width=5):
    """Build kymograph (time × X) from vessel timelapse.

    Averages ±half_width rows around vessel_row to create a 1D
    intensity profile per frame. Stacks into a 2D kymograph.

    Args:
        imgs: array-like of shape (n_frames, H, W) or list of 2D arrays.
        vessel_row: int, Y-row of vessel center.
        half_width: int, number of rows to average on each side.

    Returns:
        kymo: 2D array of shape (n_frames, W), the kymograph.
    """
    frames = np.asarray(imgs, dtype=float)
    h = frames.shape[1] if frames.ndim == 3 else frames.shape[0]
    r_min = max(0, vessel_row - half_width)
    r_max = min(h - 1, vessel_row + half_width)

    if frames.ndim == 3:
        kymo = frames[:, r_min:r_max+1, :].mean(axis=1)
    else:
        kymo = frames[r_min:r_max+1, :].mean(axis=0).reshape(1, -1)

    return kymo


def kymograph_velocity(kymo, ps, dt, max_shift=50):
    """Measure flow velocity from kymograph cross-correlation.

    For each consecutive frame pair, cross-correlates the 1D profiles
    to find how many pixels the pattern shifted (+X = rightward = +X flow).

    velocity = mean_shift_px * ps / dt

    Positive velocity = +X direction (e.g. posterior in zebrafish).

    Args:
        kymo: 2D array (n_frames, n_cols), the kymograph.
        ps: float, µm per pixel.
        dt: float, seconds per frame.
        max_shift: int, maximum shift to consider (in pixels).

    Returns:
        dict with:
            velocity: float, mean velocity in µm/s (positive = +X).
            mean_shift_px: float, mean per-frame shift in pixels.
            shifts: 1D array of per-frame shift values.
            direction: str, '+X' or '-X'.
    """
    n = kymo.shape[0]
    shifts = []

    for i in range(n - 1):
        a = kymo[i] - kymo[i].mean()
        b = kymo[i + 1] - kymo[i + 1].mean()
        std_a, std_b = np.std(a), np.std(b)
        if std_a < 0.1 or std_b < 0.1:
            continue

        n_cols = len(a)
        # Cross-correlation: ifft(conj(fft(a)) * fft(b))
        # Peak at s > 0 → b is shifted right by s compared to a → +X flow
        fa = np.fft.rfft(a, n=n_cols)
        fb = np.fft.rfft(b, n=n_cols)
        corr = np.fft.irfft(fa.conj() * fb, n=n_cols)

        # Convert to signed shift
        peak_idx = int(np.argmax(corr))
        shift = peak_idx if peak_idx <= n_cols // 2 else peak_idx - n_cols

        if abs(shift) <= max_shift:
            shifts.append(float(shift))

    if not shifts:
        return {
            'velocity': 0.0,
            'mean_shift_px': 0.0,
            'shifts': np.array([]),
            'direction': '+X',
        }

    shifts_arr = np.array(shifts)
    # Remove outliers (>3 std from median)
    med = np.median(shifts_arr)
    std = np.std(shifts_arr) + 0.1
    filtered = shifts_arr[np.abs(shifts_arr - med) < 3 * std]
    if len(filtered) == 0:
        filtered = shifts_arr

    mean_shift = float(np.mean(filtered))
    velocity = mean_shift * ps / dt

    return {
        'velocity': round(velocity, 2),
        'mean_shift_px': round(mean_shift, 3),
        'shifts': filtered,
        'direction': '+X' if velocity >= 0 else '-X',
    }


def particle_velocity(imgs, dt, ps, min_area=80, max_area=3000, max_dist=20):
    """Track dark particles (RBCs) to measure flow velocity.

    Detects dark blobs per frame via adaptive threshold, then uses
    Hungarian matching to link particles between frames. Reports
    median displacement to estimate bulk flow velocity.

    Args:
        imgs: list of 2D arrays (n_frames × H × W).
        dt: float, seconds per frame.
        ps: float, µm per pixel.
        min_area: int, minimum particle area in pixels.
        max_area: int, maximum particle area in pixels.
        max_dist: float, maximum matching distance in pixels.

    Returns:
        dict with:
            velocity_x: float, mean X-velocity (µm/s). Positive = +X.
            velocity_y: float, mean Y-velocity (µm/s). Positive = +Y.
            speed: float, mean speed magnitude (µm/s).
            direction: str, primary direction ('+X','-X','+Y','-Y').
            mean_dx_px: float, mean X displacement per frame (px).
            mean_dy_px: float, mean Y displacement per frame (px).
            step_dx: 1D array, per-frame X displacements.
            n_particles_mean: float, mean particles per frame.
    """
    from scipy.ndimage import label as nd_label
    from skimage.measure import regionprops
    from scipy.optimize import linear_sum_assignment

    # Detect dark particles per frame
    all_pos = []
    for img in imgs:
        img_f = np.asarray(img, dtype=float)
        thresh = img_f.mean() - 1.5 * img_f.std()
        dark = img_f < thresh
        lbl, _ = nd_label(dark)
        props = regionprops(lbl)
        cells = [p for p in props if min_area <= p.area <= max_area]
        if cells:
            all_pos.append(np.array([[p.centroid[0], p.centroid[1]] for p in cells]))
        else:
            all_pos.append(np.zeros((0, 2)))

    # Hungarian matching between frames
    step_dx = []
    step_dy = []
    for i in range(1, len(all_pos)):
        p1, p2 = all_pos[i-1], all_pos[i]
        if len(p1) < 2 or len(p2) < 2:
            step_dx.append(np.nan)
            step_dy.append(np.nan)
            continue
        cost = np.sqrt(((p1[:, None, :] - p2[None, :, :])**2).sum(axis=2))
        r, c = linear_sum_assignment(cost)
        valid = cost[r, c] < max_dist
        if valid.sum() < 2:
            step_dx.append(np.nan)
            step_dy.append(np.nan)
            continue
        disp = p2[c[valid]] - p1[r[valid]]
        step_dx.append(float(np.median(disp[:, 1])))
        step_dy.append(float(np.median(disp[:, 0])))

    dx_arr = np.array(step_dx)
    dy_arr = np.array(step_dy)
    valid_mask = ~np.isnan(dx_arr) & ~np.isnan(dy_arr)

    if valid_mask.sum() < 2:
        return {
            'velocity_x': 0.0, 'velocity_y': 0.0, 'speed': 0.0,
            'direction': '+X', 'mean_dx_px': 0.0, 'mean_dy_px': 0.0,
            'step_dx': dx_arr, 'n_particles_mean': 0.0,
        }

    mean_dx = float(np.median(dx_arr[valid_mask]))
    mean_dy = float(np.median(dy_arr[valid_mask]))
    vx = mean_dx * ps / dt
    vy = mean_dy * ps / dt
    speed = float(np.sqrt(vx**2 + vy**2))

    if abs(vx) >= abs(vy):
        direction = '+X' if vx >= 0 else '-X'
    else:
        direction = '+Y' if vy >= 0 else '-Y'

    n_mean = float(np.mean([len(p) for p in all_pos]))

    return {
        'velocity_x': round(vx, 2),
        'velocity_y': round(vy, 2),
        'speed': round(speed, 2),
        'direction': direction,
        'mean_dx_px': round(mean_dx, 3),
        'mean_dy_px': round(mean_dy, 3),
        'step_dx': dx_arr,
        'n_particles_mean': round(n_mean, 1),
    }
