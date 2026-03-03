"""Directed cell migration and chemotaxis analysis.

Analyzes migration trajectories for directionality, persistence,
and response to gradients. Extends basic tracking with direction-
specific metrics used in wound healing, chemotaxis, and immune
cell studies.

Functions:
    directionality_index  -- Ratio of net displacement to total path
    migration_angles      -- Extract direction of motion at each step
    angular_histogram     -- Rose diagram data for migration directions
    persistence_time      -- Estimate directional persistence from trajectory
    chemotactic_index     -- Forward migration index along gradient axis
"""

import numpy as np
from scipy import stats as scipy_stats


def directionality_index(trajectory):
    """Compute directionality (persistence ratio) of a trajectory.

    Directionality = net_displacement / total_path_length.
    Ranges from 0 (random walk) to 1 (perfectly straight).

    Args:
        trajectory: (N, 2) array of (x, y) positions over time.

    Returns:
        dict with:
            directionality: float, D/L ratio.
            net_displacement: float, straight-line start-to-end distance.
            total_path: float, sum of step distances.
            n_steps: int.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError("trajectory must be (N, 2)")
    if len(traj) < 2:
        return {
            'directionality': 0.0,
            'net_displacement': 0.0,
            'total_path': 0.0,
            'n_steps': 0,
        }

    steps = np.diff(traj, axis=0)
    step_lengths = np.sqrt(np.sum(steps**2, axis=1))
    total_path = float(step_lengths.sum())
    net_disp = float(np.sqrt(np.sum((traj[-1] - traj[0])**2)))

    di = net_disp / total_path if total_path > 0 else 0.0

    return {
        'directionality': round(di, 4),
        'net_displacement': round(net_disp, 4),
        'total_path': round(total_path, 4),
        'n_steps': len(steps),
    }


def migration_angles(trajectory, degrees=True):
    """Extract direction of motion at each step.

    Args:
        trajectory: (N, 2) array of positions.
        degrees: bool, return angles in degrees (True) or radians (False).

    Returns:
        dict with:
            angles: 1D array of migration angles (0=right, 90=up).
            step_lengths: 1D array of step sizes.
            mean_angle: float, circular mean of angles.
            angular_std: float, circular std (dispersion).
            n_steps: int.
    """
    traj = np.asarray(trajectory, dtype=float)
    if len(traj) < 2:
        return {
            'angles': np.array([]),
            'step_lengths': np.array([]),
            'mean_angle': 0.0,
            'angular_std': 0.0,
            'n_steps': 0,
        }

    steps = np.diff(traj, axis=0)
    lengths = np.sqrt(np.sum(steps**2, axis=1))

    # Filter out zero-length steps
    valid = lengths > 1e-10
    if not valid.any():
        return {
            'angles': np.array([]),
            'step_lengths': np.array([]),
            'mean_angle': 0.0,
            'angular_std': 0.0,
            'n_steps': 0,
        }

    angles_rad = np.arctan2(steps[valid, 1], steps[valid, 0])

    # Circular mean
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    mean_angle = np.arctan2(mean_sin, mean_cos)

    # Circular std (based on resultant length)
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    angular_std = np.sqrt(-2 * np.log(max(R, 1e-10))) if R < 1 else 0.0

    if degrees:
        angles_out = np.degrees(angles_rad)
        mean_out = float(np.degrees(mean_angle))
        std_out = float(np.degrees(angular_std))
    else:
        angles_out = angles_rad
        mean_out = float(mean_angle)
        std_out = float(angular_std)

    return {
        'angles': angles_out,
        'step_lengths': lengths[valid],
        'mean_angle': round(mean_out, 2),
        'angular_std': round(std_out, 2),
        'n_steps': int(valid.sum()),
    }


def angular_histogram(angles, n_bins=12, weights=None):
    """Compute angular histogram for rose diagram visualization.

    Args:
        angles: 1D array of angles in degrees.
        n_bins: int, number of angular bins (default 12 = 30° bins).
        weights: optional 1D array, weight per angle (e.g., step length).

    Returns:
        dict with:
            bin_edges: array of bin edge angles in degrees.
            bin_centers: array of bin center angles.
            counts: array of counts per bin.
            frequencies: array, normalized counts (fraction of total).
            dominant_direction: float, bin center with most entries.
    """
    angles = np.asarray(angles, dtype=float)

    # Normalize to [0, 360)
    angles_norm = angles % 360

    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts, _ = np.histogram(angles_norm, bins=bin_edges, weights=weights)
    total = counts.sum()
    frequencies = counts / total if total > 0 else counts.copy()

    dominant_idx = int(np.argmax(counts))

    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'counts': counts,
        'frequencies': frequencies,
        'dominant_direction': float(bin_centers[dominant_idx]),
    }


def persistence_time(trajectory, dt=1.0, max_lag=None):
    """Estimate directional persistence from velocity autocorrelation.

    Persistence time is the timescale over which cells maintain their
    direction of motion. Computed from exponential fit of the velocity
    direction autocorrelation.

    Args:
        trajectory: (N, 2) array of positions.
        dt: float, time between positions.
        max_lag: int or None, maximum lag for autocorrelation.
            Default: N // 3.

    Returns:
        dict with:
            persistence_time: float, characteristic direction memory
                time (in units of dt).
            autocorrelation: 1D array, direction autocorrelation.
            lags: 1D array, lag times.
            mean_speed: float, mean step speed.
    """
    traj = np.asarray(trajectory, dtype=float)
    if len(traj) < 4:
        return {
            'persistence_time': 0.0,
            'autocorrelation': np.array([]),
            'lags': np.array([]),
            'mean_speed': 0.0,
        }

    steps = np.diff(traj, axis=0)
    lengths = np.sqrt(np.sum(steps**2, axis=1))
    valid = lengths > 1e-10
    if valid.sum() < 3:
        return {
            'persistence_time': 0.0,
            'autocorrelation': np.array([]),
            'lags': np.array([]),
            'mean_speed': 0.0,
        }

    # Unit direction vectors
    directions = steps[valid] / lengths[valid, np.newaxis]
    n = len(directions)

    if max_lag is None:
        max_lag = max(n // 3, 2)
    max_lag = min(max_lag, n - 1)

    # Velocity direction autocorrelation: <v(t)·v(t+lag)>
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(directions[:n - lag] * directions[lag:], axis=1)
        autocorr[lag] = float(np.mean(dots))

    lags = np.arange(max_lag) * dt

    # Fit exponential decay: C(t) = exp(-t/tau)
    # Use log-linear fit on positive values
    positive = autocorr > 0.01
    if positive.sum() >= 2:
        log_ac = np.log(autocorr[positive])
        lag_vals = lags[positive]
        if len(lag_vals) >= 2:
            slope, _ = np.polyfit(lag_vals, log_ac, 1)
            if slope < -1e-6:
                tau = -1.0 / slope
            else:
                # No decay → infinite persistence; report total duration
                tau = float(lags[-1]) * 10
        else:
            tau = 0.0
    else:
        tau = 0.0

    return {
        'persistence_time': round(max(tau, 0.0), 4),
        'autocorrelation': autocorr,
        'lags': lags,
        'mean_speed': round(float(lengths[valid].mean() / dt), 4),
    }


def chemotactic_index(trajectory, gradient_direction=0.0, degrees=True):
    """Compute forward migration index along a gradient axis.

    The chemotactic index (CI) measures how much migration is biased
    toward a specified gradient direction. CI ranges from -1 (migrating
    away) through 0 (no bias) to +1 (perfectly toward gradient).

    Args:
        trajectory: (N, 2) array of positions.
        gradient_direction: float, direction of chemical gradient.
            0 = rightward (+x), 90 = upward (+y).
        degrees: bool, whether gradient_direction is in degrees.

    Returns:
        dict with:
            chemotactic_index: float, CI = d_parallel / total_path.
            forward_migration_index: float, same as CI.
            perpendicular_index: float, d_perp / total_path.
            net_displacement_parallel: float, displacement along gradient.
            net_displacement_perpendicular: float, displacement across gradient.
            p_value: float, significance from Rayleigh test.
    """
    traj = np.asarray(trajectory, dtype=float)
    if len(traj) < 2:
        return {
            'chemotactic_index': 0.0,
            'forward_migration_index': 0.0,
            'perpendicular_index': 0.0,
            'net_displacement_parallel': 0.0,
            'net_displacement_perpendicular': 0.0,
            'p_value': 1.0,
        }

    if degrees:
        grad_rad = np.radians(gradient_direction)
    else:
        grad_rad = gradient_direction

    # Gradient direction unit vector
    grad_vec = np.array([np.cos(grad_rad), np.sin(grad_rad)])
    perp_vec = np.array([-np.sin(grad_rad), np.cos(grad_rad)])

    # Net displacement
    net = traj[-1] - traj[0]
    d_parallel = float(np.dot(net, grad_vec))
    d_perp = float(np.dot(net, perp_vec))

    # Total path length
    steps = np.diff(traj, axis=0)
    step_lengths = np.sqrt(np.sum(steps**2, axis=1))
    total_path = float(step_lengths.sum())

    ci = d_parallel / total_path if total_path > 0 else 0.0
    pi = d_perp / total_path if total_path > 0 else 0.0

    # Rayleigh test for directional bias
    if total_path > 0 and len(steps) >= 3:
        angles = np.arctan2(steps[:, 1], steps[:, 0])
        valid = step_lengths > 1e-10
        if valid.sum() >= 3:
            angles_v = angles[valid]
            n = len(angles_v)
            R = np.sqrt(np.sum(np.cos(angles_v))**2 +
                        np.sum(np.sin(angles_v))**2) / n
            # Rayleigh test: p ≈ exp(-n * R^2) for large n
            p_val = float(np.exp(-n * R**2))
            p_val = min(max(p_val, 0.0), 1.0)
        else:
            p_val = 1.0
    else:
        p_val = 1.0

    return {
        'chemotactic_index': round(ci, 4),
        'forward_migration_index': round(ci, 4),
        'perpendicular_index': round(pi, 4),
        'net_displacement_parallel': round(d_parallel, 4),
        'net_displacement_perpendicular': round(d_perp, 4),
        'p_value': round(p_val, 6),
    }


def analyze_migration(trajectories, dt=1.0, gradient_direction=None):
    """Comprehensive migration analysis for a population of cells.

    Args:
        trajectories: list of (N_i, 2) arrays, one per cell.
        dt: float, time between positions.
        gradient_direction: float or None, gradient direction in degrees.
            If None, skips chemotactic analysis.

    Returns:
        dict with:
            n_cells: int.
            mean_speed: float, mean over all cells.
            mean_directionality: float.
            mean_persistence_time: float.
            speeds: list of per-cell speeds.
            directionalities: list of per-cell DI.
            persistence_times: list.
            chemotactic_index: float or None.
            per_cell: list of per-cell result dicts.
    """
    results = []
    for traj in trajectories:
        di = directionality_index(traj)
        pt = persistence_time(traj, dt=dt)
        cell_result = {
            'directionality': di['directionality'],
            'net_displacement': di['net_displacement'],
            'total_path': di['total_path'],
            'speed': pt['mean_speed'],
            'persistence_time': pt['persistence_time'],
        }
        if gradient_direction is not None:
            ci = chemotactic_index(traj, gradient_direction)
            cell_result['chemotactic_index'] = ci['chemotactic_index']
        results.append(cell_result)

    speeds = [r['speed'] for r in results]
    dis = [r['directionality'] for r in results]
    pts = [r['persistence_time'] for r in results]

    summary = {
        'n_cells': len(trajectories),
        'mean_speed': round(float(np.mean(speeds)), 4) if speeds else 0.0,
        'mean_directionality': round(float(np.mean(dis)), 4) if dis else 0.0,
        'mean_persistence_time': round(float(np.mean(pts)), 4) if pts else 0.0,
        'speeds': speeds,
        'directionalities': dis,
        'persistence_times': pts,
        'per_cell': results,
    }

    if gradient_direction is not None:
        cis = [r['chemotactic_index'] for r in results]
        summary['chemotactic_index'] = round(float(np.mean(cis)), 4) if cis else 0.0
    else:
        summary['chemotactic_index'] = None

    return summary
