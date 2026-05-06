"""Diffusion and mean squared displacement analysis.

Quantifies random vs directed motion from particle tracks.
Works with trajectory data from src.analysis.tracking.

Functions:
    compute_msd      -- Mean squared displacement vs lag time
    fit_diffusion     -- Fit diffusion coefficient from MSD
    classify_motion   -- Classify as confined/diffusive/directed
    ensemble_msd      -- MSD averaged over multiple trajectories
"""

import numpy as np


def compute_msd(positions, max_lag=None):
    """Compute mean squared displacement as a function of lag time.

    MSD(tau) = <|r(t+tau) - r(t)|^2>

    Args:
        positions: (N, 2) array of (x, y) positions at each time step.
        max_lag: Maximum lag in frames. Default: N//4 to ensure statistics.

    Returns:
        dict with:
            lags: 1D array of lag values (1, 2, ..., max_lag).
            msd: 1D array of MSD values at each lag.
            n_pairs: 1D array of number of pairs used at each lag.
    """
    pos = np.asarray(positions, dtype=np.float64)
    N = len(pos)

    if N < 2:
        return {"lags": np.array([]), "msd": np.array([]), "n_pairs": np.array([])}

    if max_lag is None:
        max_lag = max(1, N // 4)
    max_lag = min(max_lag, N - 1)

    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(max_lag)
    n_pairs = np.zeros(max_lag, dtype=int)

    for tau in lags:
        displacements = pos[tau:] - pos[:-tau]
        sq_disp = np.sum(displacements**2, axis=1)
        msd[tau - 1] = float(sq_disp.mean())
        n_pairs[tau - 1] = len(sq_disp)

    return {"lags": lags, "msd": msd, "n_pairs": n_pairs}


def fit_diffusion(lags, msd, dt=1.0, n_dims=2, max_fit_lag=None):
    """Fit diffusion coefficient from MSD curve.

    For normal diffusion: MSD = 2*n_dims*D*t
    For anomalous diffusion: MSD = A * t^alpha

    Args:
        lags: Array of lag times (in frames).
        msd: Array of MSD values at each lag.
        dt: Time between frames (seconds).
        n_dims: Number of spatial dimensions (2 for xy tracking).
        max_fit_lag: Use only first N lags for linear fit (default: 1/3).

    Returns:
        dict with:
            D: Diffusion coefficient (µm²/s if pixel_size applied).
            alpha: Anomalous exponent (1=normal, <1=confined, >1=directed).
            D_error: Standard error of D estimate.
            fit_type: 'normal' (alpha~1), 'confined' (alpha<0.8),
                'directed' (alpha>1.2), or 'anomalous'.
    """
    lags = np.asarray(lags, dtype=np.float64)
    msd = np.asarray(msd, dtype=np.float64)

    if len(lags) < 2:
        return {"D": 0.0, "alpha": 1.0, "D_error": 0.0, "fit_type": "insufficient_data"}

    times = lags * dt

    # Fit power law: log(MSD) = alpha * log(t) + log(A)
    valid = msd > 0
    if valid.sum() < 2:
        return {"D": 0.0, "alpha": 1.0, "D_error": 0.0, "fit_type": "insufficient_data"}

    log_t = np.log(times[valid])
    log_msd = np.log(msd[valid])

    # Linear fit in log-log space
    coeffs = np.polyfit(log_t, log_msd, 1)
    alpha = float(coeffs[0])
    log_A = float(coeffs[1])
    float(np.exp(log_A))

    # Linear fit for D (using short lags only)
    if max_fit_lag is None:
        max_fit_lag = max(2, len(lags) // 3)
    max_fit_lag = min(max_fit_lag, len(lags))

    t_fit = times[:max_fit_lag]
    msd_fit = msd[:max_fit_lag]

    # MSD = 2*n*D*t → D = slope / (2*n)
    if len(t_fit) >= 2:
        slope, intercept = np.polyfit(t_fit, msd_fit, 1)
        D = float(slope / (2 * n_dims))
        # Residuals for error estimate
        residuals = msd_fit - (slope * t_fit + intercept)
        if len(residuals) > 2:
            rmse = float(np.sqrt(np.mean(residuals**2)))
            D_error = rmse / (2 * n_dims * np.sqrt(len(t_fit)))
        else:
            D_error = 0.0
    else:
        D = 0.0
        D_error = 0.0

    # Classify
    if alpha < 0.8:
        fit_type = "confined"
    elif alpha > 1.2:
        fit_type = "directed"
    elif 0.8 <= alpha <= 1.2:
        fit_type = "normal"
    else:
        fit_type = "anomalous"

    return {
        "D": max(D, 0.0),
        "alpha": alpha,
        "D_error": D_error,
        "fit_type": fit_type,
    }


def classify_motion(positions, dt=1.0, n_dims=2):
    """Classify trajectory motion type from positions.

    Computes MSD, fits diffusion, and classifies as confined/normal/directed.

    Args:
        positions: (N, 2) array of positions.
        dt: Time between frames.
        n_dims: Number of spatial dimensions.

    Returns:
        dict with:
            motion_type: 'confined', 'normal', 'directed', or 'stationary'.
            D: Diffusion coefficient.
            alpha: Anomalous exponent.
            total_displacement: Net displacement from start to end.
            path_length: Total distance traveled.
            straightness: total_displacement / path_length.
    """
    pos = np.asarray(positions, dtype=np.float64)
    N = len(pos)

    if N < 3:
        return {
            "motion_type": "stationary",
            "D": 0.0,
            "alpha": 1.0,
            "total_displacement": 0.0,
            "path_length": 0.0,
            "straightness": 0.0,
        }

    # MSD analysis
    msd_result = compute_msd(pos)
    diff_result = fit_diffusion(msd_result["lags"], msd_result["msd"], dt=dt, n_dims=n_dims)

    # Displacement metrics
    steps = np.diff(pos, axis=0)
    step_lengths = np.sqrt(np.sum(steps**2, axis=1))
    path_length = float(step_lengths.sum())
    total_disp = float(np.sqrt(np.sum((pos[-1] - pos[0]) ** 2)))
    straightness = total_disp / path_length if path_length > 0 else 0.0

    # Override classification if effectively stationary
    if path_length < 1e-6:
        motion_type = "stationary"
    else:
        motion_type = diff_result["fit_type"]

    return {
        "motion_type": motion_type,
        "D": diff_result["D"],
        "alpha": diff_result["alpha"],
        "total_displacement": total_disp,
        "path_length": path_length,
        "straightness": straightness,
    }


def ensemble_msd(trajectories, max_lag=None, dt=1.0):
    """Compute ensemble-averaged MSD over multiple trajectories.

    Args:
        trajectories: List of (N_i, 2) position arrays.
        max_lag: Maximum lag. Default: min trajectory length // 4.
        dt: Time between frames.

    Returns:
        dict with:
            lags: 1D array of lag values.
            msd: 1D array of ensemble-averaged MSD.
            msd_sem: Standard error of the mean at each lag.
            n_trajectories: Number of trajectories used.
            D: Ensemble diffusion coefficient.
            alpha: Ensemble anomalous exponent.
    """
    if not trajectories:
        return {
            "lags": np.array([]),
            "msd": np.array([]),
            "msd_sem": np.array([]),
            "n_trajectories": 0,
            "D": 0.0,
            "alpha": 1.0,
        }

    lengths = [len(t) for t in trajectories]
    min_len = min(lengths)

    if max_lag is None:
        max_lag = max(1, min_len // 4)
    max_lag = min(max_lag, min_len - 1)

    # Compute MSD for each trajectory
    all_msd = []
    for traj in trajectories:
        result = compute_msd(traj, max_lag=max_lag)
        if len(result["msd"]) == max_lag:
            all_msd.append(result["msd"])

    if not all_msd:
        return {
            "lags": np.arange(1, max_lag + 1),
            "msd": np.zeros(max_lag),
            "msd_sem": np.zeros(max_lag),
            "n_trajectories": 0,
            "D": 0.0,
            "alpha": 1.0,
        }

    all_msd = np.array(all_msd)
    lags = np.arange(1, max_lag + 1)
    mean_msd = all_msd.mean(axis=0)
    sem_msd = all_msd.std(axis=0) / np.sqrt(len(all_msd))

    # Fit ensemble MSD
    diff = fit_diffusion(lags, mean_msd, dt=dt)

    return {
        "lags": lags,
        "msd": mean_msd,
        "msd_sem": sem_msd,
        "n_trajectories": len(all_msd),
        "D": diff["D"],
        "alpha": diff["alpha"],
    }
