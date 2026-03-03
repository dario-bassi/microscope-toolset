"""Kinetics analysis: exponential decay/recovery fitting.

Patterns extracted from Ch144, Ch192 (photobleaching) for reuse in
FRAP, bleaching kinetics, and any exponential dynamics.

Key lesson (Ch192): Signal-only pixel masking gives brightest-pixel-biased
baselines (~5% high). Region-mean or moderate-radius circular masks give
more representative values. For per-cell ranking, consistency matters more
than absolute accuracy.
"""

import numpy as np
from scipy.optimize import curve_fit


def exp_decay(t, I0, k):
    """Exponential decay: I(t) = I0 * exp(-k * t)."""
    return I0 * np.exp(-k * t)


def exp_recovery(t, I0, I_inf, k):
    """Exponential recovery (FRAP): I(t) = I_inf - (I_inf - I0) * exp(-k * t)."""
    return I_inf - (I_inf - I0) * np.exp(-k * t)


def fit_exponential_decay(times, intensities, p0=None):
    """Fit I(t) = I0 * exp(-k*t) to intensity time series.

    Args:
        times: array of time points (frames or seconds)
        intensities: array of mean intensities at each time
        p0: initial guess [I0, k] (default: auto from data)

    Returns:
        dict with I0, k, half_life, total_loss_pct, residual_std, fit_values
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(intensities, dtype=float)

    if p0 is None:
        p0 = [y[0], 0.05]

    try:
        popt, pcov = curve_fit(exp_decay, t, y, p0=p0, maxfev=10000)
        I0, k = float(popt[0]), float(popt[1])
        fit_vals = exp_decay(t, I0, k)
        resid = float(np.std(y - fit_vals))
    except Exception:
        # Fallback: log-linear regression
        pos = y > 0
        if np.sum(pos) < 2:
            return {'I0': float(y[0]), 'k': 0.0, 'half_life': float('inf'),
                    'total_loss_pct': 0.0, 'residual_std': 0.0,
                    'fit_values': y.tolist()}
        log_y = np.log(y[pos])
        coeffs = np.polyfit(t[pos], log_y, 1)
        k = float(-coeffs[0])
        I0 = float(np.exp(coeffs[1]))
        fit_vals = exp_decay(t, I0, k)
        resid = float(np.std(y - fit_vals))

    half_life = float(np.log(2) / k) if k > 0 else float('inf')
    loss = (1 - y[-1] / y[0]) * 100 if y[0] > 0 else 0.0

    return {
        'I0': round(I0, 3),
        'k': round(k, 6),
        'half_life': round(half_life, 2),
        'total_loss_pct': round(float(loss), 2),
        'residual_std': round(resid, 3),
        'fit_values': [round(float(v), 3) for v in fit_vals],
    }


def fit_exponential_recovery(times, intensities, p0=None, max_plateau=None):
    """Fit I(t) = I_inf - (I_inf - I0) * exp(-k*t) for FRAP recovery.

    Args:
        times: array of time points
        intensities: array of mean intensities
        p0: initial guess [I0, I_inf, k] (default: auto)
        max_plateau: if set, constrain I_inf <= this value (e.g., pre-bleach mean).
            Prevents extrapolation above the physical maximum.

    Returns:
        dict with I0, I_inf, k, half_life, recovery_pct, residual_std, fit_values
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(intensities, dtype=float)

    if p0 is None:
        p0 = [y[0], y[-1], 0.1]

    try:
        if max_plateau is not None:
            # Constrained fit: plateau cannot exceed max_plateau
            bounds = (
                [0, y[0], 0.001],  # lower bounds: I0, I_inf, k
                [max_plateau * 1.1, max_plateau * 1.05, 10.0],  # upper bounds
            )
            p0_clamped = [
                min(p0[0], max_plateau),
                min(p0[1], max_plateau),
                p0[2],
            ]
            popt, pcov = curve_fit(exp_recovery, t, y, p0=p0_clamped,
                                   bounds=bounds, maxfev=10000)
        else:
            popt, pcov = curve_fit(exp_recovery, t, y, p0=p0, maxfev=10000)
        I0, I_inf, k = [float(v) for v in popt]
        fit_vals = exp_recovery(t, I0, I_inf, k)
        resid = float(np.std(y - fit_vals))
    except Exception:
        I0, I_inf, k = float(y[0]), float(y[-1]), 0.0
        fit_vals = y.copy()
        resid = 0.0

    half_life = float(np.log(2) / k) if k > 0 else float('inf')
    recovery_pct = (y[-1] - y[0]) / (I_inf - I0) * 100 if I_inf != I0 else 100.0

    return {
        'I0': round(I0, 3),
        'I_inf': round(I_inf, 3),
        'k': round(k, 6),
        'half_life': round(half_life, 2),
        'recovery_pct': round(float(recovery_pct), 2),
        'residual_std': round(resid, 3),
        'fit_values': [round(float(v), 3) for v in fit_vals],
    }


def michaelis_menten(s, Vmax, Km):
    """Michaelis-Menten equation: v = Vmax * [S] / (Km + [S])."""
    return Vmax * s / (Km + s)


def fit_michaelis_menten(substrate_conc, rates, p0=None):
    """Fit Michaelis-Menten kinetics: v = Vmax * [S] / (Km + [S]).

    Args:
        substrate_conc: array of substrate concentrations.
        rates: array of initial reaction rates (same length as substrate_conc).
        p0: initial guess [Vmax, Km]. Default: auto from data.

    Returns:
        dict with Vmax, Km, Vmax_half (rate at Km), residual_std, fit_values.
    """
    s = np.asarray(substrate_conc, dtype=float)
    v = np.asarray(rates, dtype=float)

    if p0 is None:
        Vmax_guess = float(np.max(v) * 1.1)
        # Km ~ concentration at half-max rate
        half_v = Vmax_guess / 2
        idx = np.argmin(np.abs(v - half_v))
        Km_guess = max(float(s[idx]), float(s[s > 0].min()) if np.any(s > 0) else 1.0)
        p0 = [Vmax_guess, Km_guess]

    try:
        popt, pcov = curve_fit(
            michaelis_menten, s, v, p0=p0,
            bounds=([0, 0], [np.inf, np.inf]), maxfev=10000,
        )
        Vmax, Km = float(popt[0]), float(popt[1])
        fit_vals = michaelis_menten(s, Vmax, Km)
        resid = float(np.std(v - fit_vals))
    except Exception:
        # Fallback: Lineweaver-Burk (1/v vs 1/s)
        pos = (s > 0) & (v > 0)
        if np.sum(pos) < 2:
            return {'Vmax': float(np.max(v)), 'Km': 0.0, 'Vmax_half': float(np.max(v)) / 2,
                    'residual_std': 0.0, 'fit_values': v.tolist()}
        inv_s = 1.0 / s[pos]
        inv_v = 1.0 / v[pos]
        coeffs = np.polyfit(inv_s, inv_v, 1)
        # 1/v = (Km/Vmax)(1/s) + 1/Vmax
        Vmax = 1.0 / coeffs[1] if coeffs[1] > 0 else float(np.max(v))
        Km = coeffs[0] * Vmax if coeffs[1] > 0 else 0.0
        fit_vals = michaelis_menten(s, Vmax, Km)
        resid = float(np.std(v - fit_vals))

    return {
        'Vmax': round(Vmax, 4),
        'Km': round(Km, 4),
        'Vmax_half': round(Vmax / 2, 4),
        'residual_std': round(resid, 4),
        'fit_values': [round(float(fv), 4) for fv in fit_vals],
    }


def fit_beer_lambert(concentrations, absorbances):
    """Fit Beer-Lambert law: A = epsilon * l * c (linear in concentration).

    Fits A = slope * c + intercept. The slope = epsilon * pathlength.
    Returns slope for calculating unknown concentrations: c = (A - intercept) / slope.

    Args:
        concentrations: array of known concentrations (standard curve).
        absorbances: array of measured absorbances.

    Returns:
        dict with slope, intercept, r_squared, predict (callable),
        fit_values.
    """
    c = np.asarray(concentrations, dtype=float)
    a = np.asarray(absorbances, dtype=float)

    coeffs = np.polyfit(c, a, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    fit_vals = np.polyval(coeffs, c)
    ss_res = float(np.sum((a - fit_vals) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(absorbance):
        """Convert absorbance(s) to concentration."""
        abs_arr = np.asarray(absorbance, dtype=float)
        return (abs_arr - intercept) / slope if slope != 0 else abs_arr * 0

    return {
        'slope': round(slope, 6),
        'intercept': round(intercept, 6),
        'r_squared': round(r_squared, 6),
        'predict': predict,
        'fit_values': [round(float(v), 6) for v in fit_vals],
    }


def fit_hill(doses, responses, p0=None, top=None, bottom=None):
    """Fit Hill equation to dose-response data.

    4-parameter: y = bottom + (top - bottom) / (1 + (x / IC50)^n)
    2-parameter: y = bottom + (top - bottom) / (1 + (x / IC50)^n) with fixed top/bottom

    Standard for drug screening (MTT, viability assays). For viability data:
    top=1 (untreated), bottom=0 (fully dead), IC50=half-maximal dose.

    Args:
        doses: array of drug concentrations (positive values).
        responses: array of normalized responses (0-1 for viability).
        p0: initial parameter guess. For 4-param: [top, bottom, IC50, n].
            For 2-param (when top/bottom fixed): [IC50, n]. Auto-estimates if None.
        top: if given, fix the top parameter (maximum response). Reduces to 2-param fit.
        bottom: if given, fix the bottom parameter (minimum response).

    Returns:
        dict with:
            IC50: half-maximal inhibitory concentration
            hill_n: Hill coefficient (steepness)
            top: maximum response
            bottom: minimum response
            max_effect_pct: maximum effect as percentage (100 * (1 - bottom/top))
            r_squared: goodness of fit
            params: (top, bottom, IC50, n) tuple
            predict: callable, predict(doses) -> responses
    """
    from scipy.optimize import curve_fit

    doses = np.asarray(doses, dtype=float)
    responses = np.asarray(responses, dtype=float)

    # Determine fit mode based on fixed parameters
    fixed_top = top is not None
    fixed_bottom = bottom is not None

    def hill_4p(x, top_p, bottom_p, ic50, n):
        return bottom_p + (top_p - bottom_p) / (1 + (x / ic50) ** n)

    if fixed_top and fixed_bottom:
        # 2-parameter fit: only IC50 and n
        top_val = float(top)
        bottom_val = float(bottom)

        def hill_2p(x, ic50, n):
            return bottom_val + (top_val - bottom_val) / (1 + (x / ic50) ** n)

        if p0 is None:
            p0 = [float(np.median(doses)), 1.5]
        bounds_2p = ([1e-10, 0.1], [doses.max() * 100, 10.0])
        popt, pcov = curve_fit(hill_2p, doses, responses, p0=p0,
                               bounds=bounds_2p, maxfev=10000)
        ic50_val, n_val = popt
        popt_full = (top_val, bottom_val, ic50_val, n_val)
        predict_fn = lambda x, _t=top_val, _b=bottom_val, _ic=ic50_val, _n=n_val: (
            _b + (_t - _b) / (1 + (np.asarray(x, dtype=float) / _ic) ** _n)
        )
    elif fixed_top:
        # 3-parameter fit: bottom, IC50, n
        top_val = float(top)

        def hill_3p_top(x, bottom_p, ic50, n):
            return bottom_p + (top_val - bottom_p) / (1 + (x / ic50) ** n)

        if p0 is None:
            p0 = [float(np.min(responses)), float(np.median(doses)), 1.5]
        bounds_3p = ([-0.5, 1e-10, 0.1], [1.5, doses.max() * 100, 10.0])
        popt, pcov = curve_fit(hill_3p_top, doses, responses, p0=p0,
                               bounds=bounds_3p, maxfev=10000)
        bottom_val, ic50_val, n_val = popt
        popt_full = (top_val, bottom_val, ic50_val, n_val)
        predict_fn = lambda x, _p=popt_full: hill_4p(np.asarray(x, dtype=float), *_p)
    elif fixed_bottom:
        # 3-parameter fit: top, IC50, n
        bottom_val = float(bottom)

        def hill_3p_bottom(x, top_p, ic50, n):
            return bottom_val + (top_p - bottom_val) / (1 + (x / ic50) ** n)

        if p0 is None:
            p0 = [float(np.max(responses)), float(np.median(doses)), 1.5]
        bounds_3p = ([0.0, 1e-10, 0.1], [2.0, doses.max() * 100, 10.0])
        popt, pcov = curve_fit(hill_3p_bottom, doses, responses, p0=p0,
                               bounds=bounds_3p, maxfev=10000)
        top_val, ic50_val, n_val = popt
        popt_full = (top_val, bottom_val, ic50_val, n_val)
        predict_fn = lambda x, _p=popt_full: hill_4p(np.asarray(x, dtype=float), *_p)
    else:
        # Full 4-parameter fit (original behavior)
        if p0 is None:
            p0 = [
                float(np.max(responses)),
                float(np.min(responses)),
                float(np.median(doses)),
                1.5,
            ]
        bounds = ([0.0, -0.5, 1e-10, 0.1], [2.0, 1.5, doses.max() * 100, 10.0])
        popt, pcov = curve_fit(hill_4p, doses, responses, p0=p0,
                               bounds=bounds, maxfev=10000)
        top_val, bottom_val, ic50_val, n_val = popt
        popt_full = tuple(float(p) for p in popt)
        predict_fn = lambda x, _p=popt_full: hill_4p(np.asarray(x, dtype=float), *_p)

    y_pred = predict_fn(doses)
    ss_res = float(np.sum((responses - y_pred) ** 2))
    ss_tot = float(np.sum((responses - np.mean(responses)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    max_effect = 100.0 * (1.0 - float(bottom_val) / float(top_val)) if float(top_val) > 0 else 0.0

    return {
        'IC50': round(float(ic50_val), 6),
        'hill_n': round(float(n_val), 4),
        'top': round(float(top_val), 4),
        'bottom': round(float(bottom_val), 4),
        'max_effect_pct': round(max_effect, 1),
        'r_squared': round(r_squared, 6),
        'params': tuple(float(p) for p in popt_full),
        'predict': predict_fn,
    }


def fit_q10(temperatures, rates, ref_temp=None):
    """Fit Q10 temperature coefficient from rate measurements at multiple temperatures.

    Q10 model: rate(T) = rate_ref × Q10^((T - T_ref) / 10)

    Only uses temperatures where rate > 0 (excludes cold arrest / heat shock).
    For the ascending portion, Q10 is typically 2-3 for biological processes.

    Args:
        temperatures: array of temperatures (°C).
        rates: array of growth rates or reaction rates at each temperature.
        ref_temp: reference temperature (°C). Default: temperature with max rate.

    Returns:
        dict with:
            Q10: temperature coefficient.
            ref_temp: reference temperature used.
            ref_rate: rate at reference temperature.
            optimal_temp: temperature with highest rate.
            r_squared: goodness of fit.
            predicted: array of predicted rates.
    """
    temps = np.asarray(temperatures, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)

    if len(temps) < 2:
        return {'Q10': 1.0, 'ref_temp': float(temps[0]) if len(temps) else 0.0,
                'ref_rate': float(rates_arr[0]) if len(rates_arr) else 0.0,
                'optimal_temp': float(temps[0]) if len(temps) else 0.0,
                'r_squared': 0.0, 'predicted': rates_arr.tolist()}

    optimal_idx = int(np.argmax(rates_arr))
    optimal_temp = float(temps[optimal_idx])

    if ref_temp is None:
        ref_temp = optimal_temp

    # Only fit positive rates (exclude cold arrest / heat shock)
    positive = rates_arr > 0
    if positive.sum() < 2:
        return {'Q10': 1.0, 'ref_temp': ref_temp,
                'ref_rate': float(rates_arr[optimal_idx]),
                'optimal_temp': optimal_temp,
                'r_squared': 0.0, 'predicted': rates_arr.tolist()}

    t_pos = temps[positive]
    r_pos = rates_arr[positive]
    log_r = np.log(r_pos)

    # Linear regression: ln(rate) = ln(Q10)/10 * T + const
    coeffs = np.polyfit(t_pos, log_r, 1)
    slope = float(coeffs[0])  # ln(Q10) / 10
    q10 = float(np.exp(slope * 10))

    # R-squared on the positive-rate subset
    predicted_log = np.polyval(coeffs, t_pos)
    ss_res = float(np.sum((log_r - predicted_log) ** 2))
    ss_tot = float(np.sum((log_r - np.mean(log_r)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Predicted rates for all temperatures
    ref_rate_pred = float(np.exp(np.polyval(coeffs, ref_temp)))
    predicted_all = np.where(
        rates_arr > 0,
        np.exp(np.polyval(coeffs, temps)),
        0.0,
    )

    return {
        'Q10': round(q10, 4),
        'ref_temp': round(ref_temp, 1),
        'ref_rate': round(ref_rate_pred, 6),
        'optimal_temp': round(optimal_temp, 1),
        'r_squared': round(r_squared, 4),
        'predicted': [round(float(v), 6) for v in predicted_all],
    }


def measure_bleaching_trajectory(frames, cells, labels, method='region'):
    """Measure mean intensity trajectory across frames for a cell population.

    Args:
        frames: list of 2D arrays (nucleus/fluorescence images per timepoint)
        cells: list of cell dicts with 'label' key (from segment_tissue)
        labels: label image from segment_tissue
        method: 'region' (whole membrane-defined region) or
                'signal' (signal-only pixels from frame 0, mean+std threshold)

    Returns:
        dict with trajectory (list of mean intensities), per_cell_trajectories
    """
    n_frames = len(frames)
    h, w = frames[0].shape

    # Build masks
    if method == 'signal':
        f0 = frames[0].astype(float)
        masks = {}
        for c in cells:
            region = (labels == c['label'])
            pix = f0[region]
            thresh = float(np.mean(pix) + np.std(pix))
            sig = region & (f0 > thresh)
            if np.count_nonzero(sig) < 10:
                sig = region
            masks[c['label']] = sig
    else:
        masks = {c['label']: (labels == c['label']) for c in cells}

    # Measure trajectories
    global_traj = []
    per_cell = {c['label']: [] for c in cells}

    for f_idx in range(n_frames):
        frame_f = frames[f_idx].astype(float)
        vals = []
        for c in cells:
            m = masks[c['label']]
            v = float(np.mean(frame_f[m]))
            per_cell[c['label']].append(v)
            vals.append(v)
        global_traj.append(float(np.mean(vals)))

    return {
        'trajectory': global_traj,
        'per_cell': per_cell,
    }


def rank_by_bleach_rate(per_cell_trajectories, times=None, descending=True):
    """Rank cells by their exponential decay rate k.

    Args:
        per_cell_trajectories: dict {label: [intensity_per_frame]}
        times: array of time points (default: 0, 1, 2, ...)
        descending: if True, fastest bleaching first

    Returns:
        list of (label, k) tuples sorted by rate
    """
    results = []
    for label, traj in per_cell_trajectories.items():
        y = np.array(traj, dtype=float)
        if times is None:
            t = np.arange(len(y), dtype=float)
        else:
            t = np.asarray(times, dtype=float)

        fit = fit_exponential_decay(t, y)
        results.append((label, fit['k']))

    results.sort(key=lambda x: x[1], reverse=descending)
    return results


def fit_q10_with_ci(temperatures, rates, ref_temp=None, n_bootstrap=500, ci=0.95):
    """Fit Q10 with bootstrap confidence intervals.

    Extends fit_q10() with uncertainty estimation via bootstrap resampling.
    Essential for reporting measurement reliability when rate estimates vary.

    Args:
        temperatures: array of temperatures (°C).
        rates: array of growth rates at each temperature.
        ref_temp: reference temperature (°C). Default: temperature with max rate.
        n_bootstrap: int, number of bootstrap samples (default 500).
        ci: float, confidence interval level (default 0.95 = 95% CI).

    Returns:
        dict with all fit_q10 fields PLUS:
            Q10_ci_lo: float, lower bound of Q10 confidence interval.
            Q10_ci_hi: float, upper bound of Q10 confidence interval.
            Q10_std: float, standard deviation of bootstrapped Q10 estimates.
            n_bootstrap: int.
            ci_level: float.

    Notes:
        - Bootstrap resamples (temperature, rate) pairs with replacement.
        - For n<4 data points, CI will be wide (high uncertainty).
        - Use this over fit_q10() when reporting Q10 in publications or
          when comparing conditions (does CI overlap?).
    """
    temps = np.asarray(temperatures, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)

    # Get point estimate from fit_q10
    base_result = fit_q10(temps, rates_arr, ref_temp=ref_temp)

    # Bootstrap over (temp, rate) pairs
    n = len(temps)
    rng = np.random.default_rng(42)
    boot_q10s = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t_boot = temps[idx]
        r_boot = rates_arr[idx]
        # Only fit positive rates
        pos = r_boot > 0
        if pos.sum() < 2:
            continue
        t_pos = t_boot[pos]
        r_pos = r_boot[pos]
        log_r = np.log(r_pos)
        try:
            coeffs = np.polyfit(t_pos, log_r, 1)
            q10_boot = float(np.exp(coeffs[0] * 10))
            if 0.1 <= q10_boot <= 100:  # sanity bounds
                boot_q10s.append(q10_boot)
        except Exception:
            continue

    alpha = 1.0 - ci
    if len(boot_q10s) >= 10:
        q10_arr = np.array(boot_q10s)
        ci_lo = float(np.percentile(q10_arr, 100 * alpha / 2))
        ci_hi = float(np.percentile(q10_arr, 100 * (1 - alpha / 2)))
        q10_std = float(np.std(q10_arr))
    else:
        ci_lo = ci_hi = base_result['Q10']
        q10_std = 0.0

    return {
        **base_result,
        'Q10_ci_lo': round(ci_lo, 4),
        'Q10_ci_hi': round(ci_hi, 4),
        'Q10_std': round(q10_std, 4),
        'n_bootstrap': n_bootstrap,
        'ci_level': ci,
    }


def measure_growth_rate_series(counts, frames=None):
    """Estimate per-frame growth rate from a count time series.

    More robust than (n_end - n_start) / (n_snaps * n_start) because:
    1. Uses all data points (log-linear regression), not just endpoints
    2. Handles counting noise better (noise cancels in regression)
    3. Returns R-squared to assess measurement quality

    Designed for Q10 bacterial/yeast growth experiments where you snap
    N_SNAPS images at constant temperature and measure per-frame rate.

    Args:
        counts: list/array of cell counts per snap (all at same temperature).
        frames: list/array of frame numbers (default: 0, 1, ..., N-1).

    Returns:
        dict with:
            rate_per_frame: float, per-cell per-frame growth rate.
                = regression slope of log(count) vs frame
                = ln(2) / doubling_time_frames
            doubling_time_frames: float, estimated doubling time in frames.
            r_squared: float, regression quality (close to 1 = exponential growth).
            n_start: float, estimated count at frame 0.
            n_end: float, estimated count at final frame.
            method: str, description of method used.

    Notes:
        - Returns rate=0 if fewer than 3 data points or no growth detected.
        - Filters out zero/negative counts before fitting.
        - If r_squared < 0.5, growth may not be exponential (use with caution).
    """
    counts_arr = np.asarray(counts, dtype=float)
    if frames is None:
        frames_arr = np.arange(len(counts_arr), dtype=float)
    else:
        frames_arr = np.asarray(frames, dtype=float)

    # Filter valid (positive) counts
    valid = counts_arr > 0
    if valid.sum() < 3:
        return {
            'rate_per_frame': 0.0,
            'doubling_time_frames': float('inf'),
            'r_squared': 0.0,
            'n_start': float(counts_arr[0]) if len(counts_arr) > 0 else 0.0,
            'n_end': float(counts_arr[-1]) if len(counts_arr) > 0 else 0.0,
            'method': 'insufficient_data',
        }

    t = frames_arr[valid]
    log_c = np.log(counts_arr[valid])

    coeffs = np.polyfit(t, log_c, 1)
    slope = float(coeffs[0])  # = log(growth_factor) per frame

    # R-squared
    predicted = np.polyval(coeffs, t)
    ss_res = float(np.sum((log_c - predicted) ** 2))
    ss_tot = float(np.sum((log_c - np.mean(log_c)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rate = max(slope, 0.0)  # growth rate cannot be negative in a growth experiment
    dt = float(np.log(2) / rate) if rate > 0 else float('inf')

    n_start = float(np.exp(np.polyval(coeffs, frames_arr[0])))
    n_end = float(np.exp(np.polyval(coeffs, frames_arr[-1])))

    return {
        'rate_per_frame': round(rate, 6),
        'doubling_time_frames': round(dt, 2) if dt != float('inf') else float('inf'),
        'r_squared': round(r_squared, 4),
        'n_start': round(n_start, 1),
        'n_end': round(n_end, 1),
        'method': 'log_linear_regression',
    }
