"""Temperature-controlled experiment workflows for growth kinetics analysis.

Provides structured protocols for measuring biological growth or reaction rates
across multiple temperature setpoints using pymmcore-plus microscope control.

Functions:
    measure_growth_rate_at_temp  -- Single-temperature rate measurement
    temperature_response_curve   -- Multi-temperature dose-response experiment
    estimate_q10                 -- Q10 from temperature-rate pairs
    yeast_optimal_temp_state     -- Known optimal for S. cerevisiae (state 3 = 30°C)

Real microscope note:
    Measurement protocol is solid, but assumes INSTANT thermal equilibration
    when changing objective turrets or stage temperatures. Real hardware drifts:
    - Objective turret changes → 5–10°C temporary drift in focal plane
    - Stage heater activation → 30–60s settling time for temperature stability
    - Thermal gradients across sample chamber (center vs edges: 2–3°C difference)
    For accurate multi-temperature experiments:
    - Add pre-stabilization waits: wait ~60s after temperature setState()
    - Log actual temperature readings (not just setpoint) if thermistor available
    - Run baseline frames at each temperature before measurement to discard drift
    - Consider measuring thermal response curve (T vs time) on your hardware
    - Account for thermal lag when computing Q10 from fast temperature sweeps
    Current n_equil=3 may be insufficient on real hardware; increase to 10–30 frames.
"""

from typing import Any

import numpy as np

# ---- Biological constants ------------------------------------------------

# S. cerevisiae temperature controller states → °C
YEAST_TEMP_MAP = {0: 20, 1: 4, 2: 25, 3: 30, 4: 37, 5: 42}

# Known optimal for S. cerevisiae growth
YEAST_OPTIMAL_STATE = 3  # 30°C
YEAST_OPTIMAL_TEMP = 30.0

# Expected Q10 for yeast growth (below optimum)
YEAST_Q10_EXPECTED = 2.0

# Temperature states that are sub-optimal (T ≤ 30°C)
YEAST_SUB_OPTIMAL_STATES = [1, 0, 2, 3]  # 4, 20, 25, 30°C


# ---- Cell counting utilities ---------------------------------------------


def count_yeast_blobs(image, threshold=0.15, min_sigma=2.0, max_sigma=8.0):
    """Count yeast cells via Laplacian of Gaussian blob detection.

    Works on both brightfield and fluorescence (nucleus-channel) images.
    Normalized before detection for illumination invariance.

    Args:
        image:      2D numpy array (grayscale).
        threshold:  blob_log threshold on normalized image.
                    0.15 is a good default for yeast at 10x.
        min_sigma:  Minimum cell radius estimate (pixels).
        max_sigma:  Maximum cell radius estimate (pixels).

    Returns:
        int: Number of detected cells (minimum 1 to avoid division by zero).
    """
    from skimage.feature import blob_log

    img = np.asarray(image, dtype=float)
    img_max = img.max()
    if img_max <= 0:
        return 1
    img_norm = img / img_max
    try:
        blobs = blob_log(img_norm, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        return max(int(len(blobs)), 1)
    except Exception:
        return 1


def count_yeast_watershed(image, min_area=50, max_area=5000):
    """Count yeast cells via Otsu threshold + watershed splitting.

    More robust than blob_log for dense cultures. Handles touching cells.

    Args:
        image:    2D numpy array (grayscale, fluorescence preferred).
        min_area: Minimum cell area (pixels²) to count.
        max_area: Maximum cell area (pixels²) to count.

    Returns:
        int: Number of detected cells.
    """
    from scipy.ndimage import distance_transform_edt, label
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    img = np.asarray(image, dtype=float)
    img_max = img.max()
    if img_max <= 0:
        return 1

    # Threshold
    thresh = threshold_otsu(img)
    binary = img > thresh
    binary = remove_small_objects(binary, min_size=min_area)

    if not binary.any():
        return 1

    # Watershed for splitting touching cells
    dist = distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=5, labels=binary)
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    labeled = watershed(-dist, markers, mask=binary)
    n_labels = labeled.max()

    # Filter by area
    count = 0
    for lbl in range(1, n_labels + 1):
        area = int((labeled == lbl).sum())
        if min_area <= area <= max_area:
            count += 1

    return max(count, 1)


def estimate_budding_rate(counts: list[int], n_snaps: int | None = None) -> float:
    """Estimate budding rate per cell per snap from count timeseries.

    Uses linear regression over all counts (more robust than start-end).
    Rate = slope / estimated_baseline_count

    Args:
        counts:  List of cell counts over time (one per snap).
        n_snaps: Optional denominator override. Default: len(counts) - 1.

    Returns:
        float: Budding rate (divisions per cell per snap). Non-negative.
    """
    if len(counts) < 2:
        return 0.0
    x = np.arange(len(counts), dtype=float)
    try:
        slope, intercept = np.polyfit(x, counts, 1)
        baseline = max(float(intercept), 1.0)
        rate = max(float(slope) / baseline, 0.0)
    except Exception:
        # Fallback: simple start-end
        n = n_snaps if n_snaps else max(len(counts) - 1, 1)
        start = max(counts[0], 1)
        rate = max((counts[-1] - counts[0]) / (n * start), 0.0)
    return float(rate)


# ---- Experiment protocol ---------------------------------------------------


def measure_growth_rate_at_temp(
    core,
    state: int,
    group: str,
    channel: str,
    n_equil: int = 3,
    n_measure: int = 30,
    temp_device: str = "Temperature",
    count_fn=None,
) -> dict[str, Any]:
    """Measure growth rate at a single temperature setpoint.

    Protocol:
        1. Set temperature device to state
        2. Wait n_equil snaps for equilibration
        3. Acquire n_measure snaps for growth measurement
        4. Return counts + estimated budding rate

    Args:
        core:        pymmcore-plus core instance.
        state:       Temperature controller state (int).
        group:       Config group name (e.g., 'Fake').
        channel:     Channel config for cell imaging.
        n_equil:     Number of equilibration snaps (discarded).
        n_measure:   Number of measurement snaps.
        temp_device: Device name for setState().
        count_fn:    Optional custom cell counting function.
                     Must accept a 2D numpy array and return int.
                     Default: count_yeast_blobs.

    Returns:
        dict with keys:
            state, temp_C, counts, rate, n_start, n_end
    """
    from useq import MDASequence

    from src.hardware.core import run_events

    if count_fn is None:
        count_fn = count_yeast_blobs

    # Set temperature
    core.setState(temp_device, state)

    # Equilibration — discard frames while temperature stabilises
    if n_equil > 0:
        equil_seq = MDASequence(
            time_plan={"loops": n_equil, "interval": 0.0},
            channels=[{"config": channel, "group": group}],
        )
        run_events(core, list(equil_seq))

    # Measurement via MDA
    seq = MDASequence(
        time_plan={"loops": n_measure, "interval": 0.0},
        channels=[{"config": channel, "group": group}],
    )
    frames = []

    def _on_frame(img, event):
        if img.ndim == 3:
            img = img[:, :, 0]
        frames.append(img.astype(float))

    run_events(core, list(seq), on_frame=_on_frame)

    # Count cells in each frame
    counts = [count_fn(f) for f in frames]
    rate = estimate_budding_rate(counts)

    temp_map = getattr(core, "_temp_map", YEAST_TEMP_MAP)
    temp_C = temp_map.get(state, -1)

    return {
        "state": int(state),
        "temp_C": int(temp_C),
        "counts": counts,
        "n_start": counts[0],
        "n_end": counts[-1],
        "rate": float(rate),
    }


# ---- Q10 Fitting ---------------------------------------------------------


def estimate_q10(
    temps_C: list[float],
    rates: list[float],
    t_optimal: float = YEAST_OPTIMAL_TEMP,
    sub_optimal_only: bool = True,
) -> dict[str, float]:
    """Estimate Q10 temperature coefficient from rate measurements.

    Uses log-linear regression: log(rate) = (T - T_opt)/10 × log(Q10)

    Args:
        temps_C:         Temperature values (°C).
        rates:           Corresponding growth rates.
        t_optimal:       Optimal temperature (°C). Default: 30 for yeast.
        sub_optimal_only: If True, only fit points where T ≤ t_optimal.

    Returns:
        dict with:
            Q10:          Temperature coefficient (clipped to [1, 10]).
            slope:        Log-linear slope.
            intercept:    Log-linear intercept.
            r_squared:    R² of fit.
            n_points:     Number of data points used.
    """
    temps = np.asarray(temps_C, dtype=float)
    rates = np.asarray(rates, dtype=float)

    # Filter valid rates
    valid = rates > 0
    if sub_optimal_only:
        valid = valid & (temps <= t_optimal)

    n_points = int(valid.sum())
    if n_points < 2:
        return {
            "Q10": YEAST_Q10_EXPECTED,
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "n_points": n_points,
        }

    T_fit = temps[valid]
    R_fit = rates[valid]
    x = (T_fit - t_optimal) / 10.0
    y = np.log(R_fit)

    try:
        slope, intercept = np.polyfit(x, y, 1)
        Q10 = float(np.exp(slope))

        # R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = float(1.0 - ss_res / max(ss_tot, 1e-10))
    except Exception:
        Q10 = YEAST_Q10_EXPECTED
        slope = intercept = 0.0
        r_sq = 0.0

    Q10 = float(np.clip(Q10, 1.0, 10.0))

    return {
        "Q10": round(Q10, 4),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(r_sq, 4),
        "n_points": n_points,
    }


def yeast_optimal_temp_state() -> tuple[int, float]:
    """Return the known optimal temperature state for S. cerevisiae.

    Biology: yeast grows optimally at 30°C. Never trust measurement alone
    for T_optimal — counting noise can make 37°C appear faster.

    Returns:
        (state, temp_C): (3, 30.0)
    """
    return YEAST_OPTIMAL_STATE, float(YEAST_OPTIMAL_TEMP)


def select_counting_channel(core, group: str, channels: list[str]) -> str:
    """Select the best channel for yeast cell counting.

    Prefers nucleus-channel (Calcofluor-White) which gives clean bright
    blobs on dark background. Falls back to brightfield if CW not present.

    Args:
        core:     pymmcore-plus core instance.
        group:    Config group.
        channels: Available channel names.

    Returns:
        Name of best channel for counting.
    """
    from src.hardware.core import snap as _snap

    def snap_count(ch):
        img = _snap(core, channel=ch)
        if img.ndim == 3:
            img = img[:, :, 0]
        return count_yeast_blobs(img.astype(float))

    if "nucleus-channel" in channels:
        nc = snap_count("nucleus-channel")
        snap_count("brightfield") if "brightfield" in channels else 0
        if nc >= 5:
            return "nucleus-channel"

    if "brightfield" in channels:
        return "brightfield"

    return channels[0] if channels else "brightfield"


# ---- Sprint 56: Regression-based rate measurement ------------------------


def measure_growth_rate_at_temp_v2(
    core,
    state: int,
    group: str,
    channel: str,
    n_equil: int = 2,
    n_measure: int = 12,
    temp_device: str = "Temperature",
    count_fn=None,
) -> dict[str, Any]:
    """Measure growth rate at a single temperature using log-linear regression.

    Improvement over measure_growth_rate_at_temp():
    - Uses measure_growth_rate_series() from kinetics.py for regression over
      all N snaps (not just start vs end) — much more noise-resistant.
    - Returns per-frame rate, doubling time, and R² quality metric.
    - count_fn default: count_yeast_blobs with min_sigma=2 (catches smaller cells).

    Protocol:
        1. setState to temperature
        2. n_equil equilibration snaps (discarded)
        3. n_measure snaps recorded and counted
        4. Log-linear regression over count series → rate_per_frame

    Args:
        core:        pymmcore-plus core instance.
        state:       Temperature controller state (int).
        group:       Config group name (e.g., 'Fake').
        channel:     Channel config name for cell imaging.
        n_equil:     Number of equilibration snaps to discard.
        n_measure:   Number of measurement snaps.
        temp_device: Temperature device name for core.setState().
        count_fn:    Optional cell counting function (img → int).
                     Default: count_yeast_blobs with min_sigma=2.

    Returns:
        dict with keys:
            state:              int, temperature state
            temp_C:             int, temperature in °C
            counts:             list[int], raw count per snap
            rate_per_frame:     float, log-linear growth rate per snap
            doubling_time_frames: float, doubling time in frames (inf if rate=0)
            r_squared:          float, regression quality (1.0 = perfect exponential)
            n_start:            float, regression-estimated start count
            n_end:              float, regression-estimated end count
            method:             str, 'log_linear_regression'
    """
    from useq import MDASequence

    from src.analysis.kinetics import measure_growth_rate_series
    from src.hardware.core import run_events

    if count_fn is None:

        def count_fn(img):
            return count_yeast_blobs(img, threshold=0.12, min_sigma=2.0, max_sigma=10.0)

    core.setState(temp_device, state)

    # Equilibration — discard frames while temperature stabilises
    if n_equil > 0:
        equil_seq = MDASequence(
            time_plan={"loops": n_equil, "interval": 0.0},
            channels=[{"config": channel, "group": group}],
        )
        run_events(core, list(equil_seq))

    # Measurement via MDA
    meas_seq = MDASequence(
        time_plan={"loops": n_measure, "interval": 0.0},
        channels=[{"config": channel, "group": group}],
    )
    counts = []

    def _on_frame(img, event):
        if img.ndim == 3:
            img = img[:, :, 0]
        counts.append(count_fn(img.astype(float)))

    run_events(core, list(meas_seq), on_frame=_on_frame)

    # Regression-based rate
    reg = measure_growth_rate_series(counts)

    temp_map = getattr(core, "_temp_map", YEAST_TEMP_MAP)
    temp_C = temp_map.get(state, -1)

    return {
        "state": int(state),
        "temp_C": int(temp_C),
        "counts": counts,
        "rate_per_frame": reg["rate_per_frame"],
        "doubling_time_frames": reg["doubling_time_frames"],
        "r_squared": reg["r_squared"],
        "n_start": reg["n_start"],
        "n_end": reg["n_end"],
        "method": reg["method"],
    }


def temperature_response_curve_v2(
    core,
    states: list[int],
    group: str,
    channel: str,
    n_equil: int = 2,
    n_measure: int = 12,
    temp_device: str = "Temperature",
    t_optimal: float = YEAST_OPTIMAL_TEMP,
    count_fn=None,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Multi-temperature growth rate experiment with regression + CI reporting.

    Runs measure_growth_rate_at_temp_v2() for each state, then fits Q10 with
    bootstrap confidence intervals via fit_q10_with_ci().

    Best practice:
        - Order states cold-to-warm (prevents overcrowding at high density)
        - Use membrane-channel or nucleus-channel (fluorescence), NOT brightfield
        - Use n_measure >= 12 for reliable regression

    Args:
        core:        pymmcore-plus core instance.
        states:      List of temperature states to visit (e.g., [1,0,2,3,4,5]).
        group:       Config group name.
        channel:     Channel config for cell imaging.
        n_equil:     Equilibration snaps per temperature.
        n_measure:   Measurement snaps per temperature.
        temp_device: Temperature device name.
        t_optimal:   Known biological optimum (°C). Default 30 for yeast.
        count_fn:    Optional custom counting function.
        n_bootstrap: Bootstrap replicates for CI estimation.

    Returns:
        dict with keys:
            results:      list of per-temperature dicts from measure_growth_rate_at_temp_v2
            Q10:          float, fitted Q10 temperature coefficient
            Q10_ci_lo:    float, lower 95% CI bound
            Q10_ci_hi:    float, upper 95% CI bound
            Q10_std:      float, bootstrap std of Q10
            T_optimal_C:  float, temperature with highest growth rate
            T_optimal_state: int, state with highest rate
            temps_C:      list[int], temperatures tested
            rates:        list[float], regression rates per temperature
            r_squared_values: list[float], per-temperature fit quality
    """
    from src.analysis.kinetics import fit_q10_with_ci

    results = []
    for state in states:
        r = measure_growth_rate_at_temp_v2(
            core,
            state,
            group,
            channel,
            n_equil=n_equil,
            n_measure=n_measure,
            temp_device=temp_device,
            count_fn=count_fn,
        )
        results.append(r)
        print(
            f"  {r['temp_C']}°C: rate={r['rate_per_frame']:.5f}, "
            f"R²={r['r_squared']:.3f}, counts={r['counts'][:3]}..."
        )

    temps_C = [r["temp_C"] for r in results]
    rates = [r["rate_per_frame"] for r in results]
    r_sq_vals = [r["r_squared"] for r in results]

    # Find empirical optimum
    best_idx = int(np.argmax(rates))
    T_opt_measured = float(temps_C[best_idx])
    T_opt_state = int(results[best_idx]["state"])

    # Q10 fit with CI (sub-optimal only: T <= t_optimal)
    ci_result = fit_q10_with_ci(
        temps_C,
        rates,
        ref_temp=t_optimal,
        n_bootstrap=n_bootstrap,
    )

    return {
        "results": results,
        "Q10": ci_result["Q10"],
        "Q10_ci_lo": ci_result.get("Q10_ci_lo", ci_result["Q10"]),
        "Q10_ci_hi": ci_result.get("Q10_ci_hi", ci_result["Q10"]),
        "Q10_std": ci_result.get("Q10_std", 0.0),
        "T_optimal_C": T_opt_measured,
        "T_optimal_state": T_opt_state,
        "temps_C": temps_C,
        "rates": rates,
        "r_squared_values": r_sq_vals,
    }
