"""Fluorescence Recovery After Photobleaching (FRAP) analysis.

Quantifies molecular diffusion and binding dynamics from FRAP
experiments. Measures bleach depth, recovery kinetics, mobile
fraction, and effective diffusion coefficient.

Functions:
    measure_frap_curve    -- Extract intensity vs time from bleach ROI
    fit_frap_recovery     -- Fit exponential recovery to FRAP curve
    mobile_fraction       -- Calculate mobile and immobile fractions
    diffusion_coefficient -- Estimate D from recovery half-time and ROI geometry
    normalize_frap        -- Double-normalize FRAP curve
"""

import numpy as np
from scipy import optimize


def measure_frap_curve(stack, roi_mask, reference_mask=None):
    """Extract FRAP intensity curve from a timelapse stack.

    Measures mean intensity in the bleached ROI at each timepoint.
    Optionally corrects for overall photobleaching using a reference
    region.

    Args:
        stack: 3D array (T, H, W), fluorescence timelapse.
        roi_mask: 2D bool array, bleached region of interest.
        reference_mask: 2D bool array or None, unbleached reference region
            for photobleaching correction.

    Returns:
        dict with:
            raw_intensity: 1D array, mean ROI intensity per frame.
            corrected_intensity: 1D array (corrected if reference given).
            reference_intensity: 1D array or None.
            n_frames: int.
    """
    stack = np.asarray(stack, dtype=float)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    n_frames = stack.shape[0]
    raw = np.array([stack[t][roi_mask].mean() for t in range(n_frames)])

    if reference_mask is not None:
        reference_mask = np.asarray(reference_mask, dtype=bool)
        ref = np.array([stack[t][reference_mask].mean() for t in range(n_frames)])
        # Correct: multiply by (ref_initial / ref_current)
        ref_initial = ref[0] if ref[0] > 0 else 1.0
        correction = ref_initial / np.where(ref > 0, ref, 1.0)
        corrected = raw * correction
    else:
        ref = None
        corrected = raw.copy()

    return {
        "raw_intensity": raw,
        "corrected_intensity": corrected,
        "reference_intensity": ref,
        "n_frames": n_frames,
    }


def normalize_frap(intensity, bleach_frame, pre_bleach_frames=None):
    """Double-normalize a FRAP intensity curve.

    Normalizes so that pre-bleach = 1.0 and bleach minimum = 0.0.
    This standard normalization allows comparison between experiments.

    Args:
        intensity: 1D array, corrected intensity values.
        bleach_frame: int, index of the bleach event (first post-bleach frame).
        pre_bleach_frames: int or None, number of pre-bleach frames to
            average. If None, uses all frames before bleach_frame.

    Returns:
        dict with:
            normalized: 1D array, normalized FRAP curve.
            pre_bleach_intensity: float, mean pre-bleach intensity.
            bleach_intensity: float, intensity right after bleach.
            bleach_depth: float, fraction of signal bleached (0-1).
    """
    intensity = np.asarray(intensity, dtype=float)

    if pre_bleach_frames is None:
        pre_region = intensity[:bleach_frame]
    else:
        start = max(0, bleach_frame - pre_bleach_frames)
        pre_region = intensity[start:bleach_frame]

    pre_mean = float(pre_region.mean()) if len(pre_region) > 0 else float(intensity[0])
    bleach_val = float(intensity[bleach_frame]) if bleach_frame < len(intensity) else pre_mean

    # Normalize: pre-bleach=1, bleach=0
    denom = pre_mean - bleach_val
    if abs(denom) < 1e-10:
        normalized = np.ones_like(intensity)
    else:
        normalized = (intensity - bleach_val) / denom

    bleach_depth = (pre_mean - bleach_val) / pre_mean if pre_mean > 0 else 0.0

    return {
        "normalized": normalized,
        "pre_bleach_intensity": round(pre_mean, 4),
        "bleach_intensity": round(bleach_val, 4),
        "bleach_depth": round(max(bleach_depth, 0.0), 4),
    }


def fit_frap_recovery(timepoints, intensity, bleach_frame):
    """Fit exponential recovery curve to FRAP data.

    Fits: I(t) = I_inf * (1 - exp(-t/tau)) + I_0
    where tau is the recovery time constant.

    Args:
        timepoints: 1D array of time values.
        intensity: 1D array of (normalized) intensity values.
        bleach_frame: int, index of bleach event.

    Returns:
        dict with:
            tau: float, recovery time constant.
            half_time: float, recovery half-time (t_1/2 = tau * ln(2)).
            plateau: float, asymptotic recovery level.
            r_squared: float, goodness of fit.
            fitted: 1D array, fitted curve values.
    """
    timepoints = np.asarray(timepoints, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    # Use only post-bleach data
    post_mask = np.arange(len(timepoints)) >= bleach_frame
    t_post = timepoints[post_mask] - timepoints[bleach_frame]
    i_post = intensity[post_mask]

    if len(t_post) < 3:
        return {
            "tau": 0.0,
            "half_time": 0.0,
            "plateau": 0.0,
            "r_squared": 0.0,
            "fitted": intensity.copy(),
        }

    i0 = float(i_post[0])
    i_inf_est = float(i_post[-1])

    def model(t, i_inf, tau):
        return i_inf * (1 - np.exp(-t / max(tau, 1e-10))) + i0 * np.exp(-t / max(tau, 1e-10))

    try:
        popt, _ = optimize.curve_fit(
            model,
            t_post,
            i_post,
            p0=[i_inf_est, float(t_post[-1]) / 3],
            bounds=([0, 1e-6], [np.inf, np.inf]),
            maxfev=5000,
        )
        i_inf, tau = popt

        # R² calculation
        fitted_post = model(t_post, *popt)
        ss_res = np.sum((i_post - fitted_post) ** 2)
        ss_tot = np.sum((i_post - i_post.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Full fitted curve
        fitted = intensity.copy()
        fitted[post_mask] = fitted_post

    except (RuntimeError, ValueError):
        # Fallback: linear estimate
        tau = float(t_post[-1]) / 2
        i_inf = float(i_post[-1])
        r2 = 0.0
        fitted = intensity.copy()

    half_time = tau * np.log(2)

    return {
        "tau": round(float(tau), 4),
        "half_time": round(float(half_time), 4),
        "plateau": round(float(i_inf), 4),
        "r_squared": round(float(max(r2, 0.0)), 4),
        "fitted": fitted,
    }


def mobile_fraction(normalized_curve, bleach_frame):
    """Calculate mobile and immobile fractions from FRAP recovery.

    Mobile fraction = (I_recovery - I_bleach) / (I_pre - I_bleach)
    where I_recovery is the asymptotic plateau.

    Args:
        normalized_curve: 1D array, double-normalized FRAP curve
            (pre=1, bleach=0).
        bleach_frame: int, index of bleach event.

    Returns:
        dict with:
            mobile_fraction: float (0-1).
            immobile_fraction: float (0-1).
            plateau: float, recovery plateau.
    """
    curve = np.asarray(normalized_curve, dtype=float)

    # Pre-bleach level (should be ~1 if normalized)
    pre = float(curve[:bleach_frame].mean()) if bleach_frame > 0 else 1.0

    # Bleach level (should be ~0 if normalized)
    bleach = float(curve[bleach_frame])

    # Recovery plateau: average of last 20% of frames
    n_post = len(curve) - bleach_frame
    last_start = max(bleach_frame + int(n_post * 0.8), bleach_frame + 1)
    plateau = float(curve[last_start:].mean()) if last_start < len(curve) else float(curve[-1])

    denom = pre - bleach
    if abs(denom) < 1e-10:
        mf = 1.0
    else:
        mf = (plateau - bleach) / denom

    mf = max(min(mf, 1.0), 0.0)

    return {
        "mobile_fraction": round(mf, 4),
        "immobile_fraction": round(1.0 - mf, 4),
        "plateau": round(plateau, 4),
    }


def diffusion_coefficient(half_time, roi_radius, geometry="circle"):
    """Estimate effective diffusion coefficient from FRAP recovery.

    Uses the Soumpasis (1983) model for circular bleach spots:
    D = 0.224 * r² / t_1/2

    Args:
        half_time: float, recovery half-time (same units as D output).
        roi_radius: float, radius of bleach spot (in physical units).
        geometry: str, 'circle' or 'strip'.

    Returns:
        dict with:
            D_eff: float, effective diffusion coefficient (unit²/time).
            roi_radius: float.
            half_time: float.
            model: str.
    """
    if half_time <= 0 or roi_radius <= 0:
        return {
            "D_eff": 0.0,
            "roi_radius": float(roi_radius),
            "half_time": float(half_time),
            "model": geometry,
        }

    if geometry == "circle":
        # Soumpasis model
        D = 0.224 * roi_radius**2 / half_time
    elif geometry == "strip":
        # Strip geometry: D = w² / (4 * t_1/2)
        D = roi_radius**2 / (4 * half_time)
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    return {
        "D_eff": round(float(D), 6),
        "roi_radius": float(roi_radius),
        "half_time": float(half_time),
        "model": geometry,
    }
