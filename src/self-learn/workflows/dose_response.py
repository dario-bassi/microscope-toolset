"""Dose-response workflow: plate layout + measurement + Hill fitting.

Provides a complete pipeline for drug screening experiments:
1. Define plate layout (concentrations, controls, replicates)
2. Acquire measurements at each well
3. Background-correct and normalize
4. Fit Hill curve to extract IC50, EC50, etc.

Works with both microscopy (cell-based) and plate reader data.
"""

import numpy as np

from ..analysis.kinetics import fit_hill


def make_plate_layout(doses, n_replicates=3, control_wells=2, n_rows=8, n_cols=12):
    """Generate a standard dose-response plate layout.

    Creates a column-based layout where each dose occupies one column
    with n_replicates rows. Control wells (no drug) are placed in the
    last column(s).

    Args:
        doses: List of drug concentrations (e.g., [0.01, 0.1, 1, 10, 100]).
        n_replicates: Number of replicate wells per dose.
        control_wells: Number of columns reserved for untreated controls.
        n_rows: Plate rows (default 8 for 96-well).
        n_cols: Plate columns (default 12 for 96-well).

    Returns:
        dict with:
            layout: 2D array (n_rows, n_cols) of concentrations. 0 = control,
                -1 = empty, >0 = drug concentration.
            dose_columns: Dict mapping dose -> list of (row, col) positions.
            control_positions: List of (row, col) for control wells.
            doses: Sorted list of unique doses used.
    """
    layout = np.full((n_rows, n_cols), -1.0)  # -1 = empty
    dose_columns = {}
    control_positions = []

    # Place controls in last column(s)
    for c in range(n_cols - control_wells, n_cols):
        for r in range(min(n_replicates, n_rows)):
            layout[r, c] = 0.0
            control_positions.append((r, c))

    # Place doses starting from column 0
    sorted_doses = sorted(doses)
    col = 0
    for dose in sorted_doses:
        if col >= n_cols - control_wells:
            break
        positions = []
        for r in range(min(n_replicates, n_rows)):
            layout[r, col] = dose
            positions.append((r, col))
        dose_columns[dose] = positions
        col += 1

    return {
        "layout": layout,
        "dose_columns": dose_columns,
        "control_positions": control_positions,
        "doses": sorted_doses,
    }


def measure_plate(measurements, layout, metric="mean_intensity"):
    """Extract per-well measurements from a plate layout.

    Args:
        measurements: 2D array (n_rows, n_cols) or dict mapping (row, col)
            to measurement value. If 2D array, each element is the
            measurement for that well.
        layout: Layout dict from make_plate_layout().
        metric: Ignored if measurements is numeric; used as key if
            measurements is a dict of dicts.

    Returns:
        dict with:
            doses: Array of unique doses (sorted).
            means: Mean response at each dose.
            stds: Std of response at each dose.
            raw: Dict mapping dose -> list of raw values.
            control_mean: Mean of control wells.
            control_std: Std of control wells.
    """
    layout["layout"]

    def _get_value(r, c):
        if isinstance(measurements, np.ndarray):
            return float(measurements[r, c])
        elif isinstance(measurements, dict):
            val = measurements.get((r, c), None)
            if val is None:
                return None
            if isinstance(val, dict):
                return float(val.get(metric, 0))
            return float(val)
        return None

    # Collect values per dose
    raw = {}
    for dose, positions in layout["dose_columns"].items():
        values = []
        for r, c in positions:
            v = _get_value(r, c)
            if v is not None:
                values.append(v)
        raw[dose] = values

    # Control values
    ctrl_values = []
    for r, c in layout["control_positions"]:
        v = _get_value(r, c)
        if v is not None:
            ctrl_values.append(v)
    raw[0.0] = ctrl_values

    # Compute stats
    all_doses = sorted(set([0.0] + list(layout["doses"])))
    means = []
    stds = []
    for d in all_doses:
        vals = raw.get(d, [])
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    ctrl_mean = float(np.mean(ctrl_values)) if ctrl_values else 0.0
    ctrl_std = float(np.std(ctrl_values)) if ctrl_values else 0.0

    return {
        "doses": np.array(all_doses),
        "means": np.array(means),
        "stds": np.array(stds),
        "raw": raw,
        "control_mean": ctrl_mean,
        "control_std": ctrl_std,
    }


def normalize_responses(plate_data, method="control", background=None):
    """Normalize dose-response data for Hill fitting.

    Args:
        plate_data: Dict from measure_plate().
        method: 'control' (divide by control mean) or 'range' (min-max).
        background: Optional background value to subtract from all measurements.

    Returns:
        dict with:
            doses: Array of doses (excluding control dose 0).
            responses: Normalized response at each dose.
            stds: Normalized standard deviations.
            control_response: Normalized control value (should be ~1.0).
    """
    doses = plate_data["doses"]
    means = plate_data["means"].copy()
    stds = plate_data["stds"].copy()
    ctrl_mean = plate_data["control_mean"]

    # Background subtraction
    if background is not None:
        means = means - background
        ctrl_mean = ctrl_mean - background

    if method == "control":
        if ctrl_mean > 0:
            norm_means = means / ctrl_mean
            norm_stds = stds / ctrl_mean
        else:
            norm_means = means
            norm_stds = stds
    elif method == "range":
        vmin = float(np.nanmin(means))
        vmax = float(np.nanmax(means))
        rng = vmax - vmin
        if rng > 0:
            norm_means = (means - vmin) / rng
            norm_stds = stds / rng
        else:
            norm_means = np.ones_like(means)
            norm_stds = np.zeros_like(stds)
    else:
        raise ValueError(f"method must be 'control' or 'range', got '{method}'")

    # Exclude dose=0 (control) from fitting data
    mask = doses > 0
    return {
        "doses": doses[mask],
        "responses": norm_means[mask],
        "stds": norm_stds[mask],
        "control_response": float(norm_means[0]) if len(norm_means) > 0 else 1.0,
    }


def auto_ec50(doses, responses, top=None, bottom=None):
    """Complete EC50/IC50 estimation with automatic parameter selection.

    Wrapper around fit_hill() that auto-detects inhibition vs stimulation
    and chooses appropriate constraints.

    Args:
        doses: Array of drug concentrations (positive, non-zero).
        responses: Normalized responses (0-1 typical for viability).
        top: Fixed top parameter. If None, auto-detected.
        bottom: Fixed bottom parameter. If None, auto-detected.

    Returns:
        dict with:
            IC50: Half-maximal concentration.
            hill_n: Hill coefficient.
            top: Maximum response.
            bottom: Minimum response.
            effect_type: 'inhibition' or 'stimulation'.
            max_effect_pct: Maximum effect percentage.
            r_squared: Fit quality.
            predict: Callable for predictions.
            quality: 'good' (R²>0.9), 'acceptable' (R²>0.7), or 'poor'.
    """
    doses = np.asarray(doses, dtype=float)
    responses = np.asarray(responses, dtype=float)

    # Auto-detect response direction
    low_dose_resp = float(np.mean(responses[doses <= np.percentile(doses, 30)]))
    high_dose_resp = float(np.mean(responses[doses >= np.percentile(doses, 70)]))

    if high_dose_resp < low_dose_resp:
        effect_type = "inhibition"
        if top is None:
            top = max(1.0, float(np.max(responses) * 1.05))
        if bottom is None:
            bottom = max(0.0, float(np.min(responses) * 0.95))
    else:
        effect_type = "stimulation"
        if top is None:
            top = float(np.max(responses) * 1.05)
        if bottom is None:
            bottom = max(0.0, float(np.min(responses) * 0.95))

    result = fit_hill(doses, responses, top=top, bottom=bottom)
    result["effect_type"] = effect_type

    r2 = result.get("r_squared", 0)
    if r2 >= 0.9:
        result["quality"] = "good"
    elif r2 >= 0.7:
        result["quality"] = "acceptable"
    else:
        result["quality"] = "poor"

    return result


def dose_response_pipeline(
    measurements, doses, n_replicates=3, background=None, top=None, bottom=None
):
    """End-to-end dose-response analysis from raw measurements.

    Combines plate layout, measurement extraction, normalization,
    and Hill fitting in a single call.

    Args:
        measurements: 2D array (n_rows, n_cols) of raw well measurements,
            OR list of values ordered by [dose0_rep0, dose0_rep1, ..., dose1_rep0, ...].
        doses: List of drug concentrations.
        n_replicates: Replicates per dose.
        background: Optional background value to subtract.
        top: Fixed Hill top (None = auto).
        bottom: Fixed Hill bottom (None = auto).

    Returns:
        dict with:
            layout: Plate layout info.
            raw: Raw measurement data per dose.
            normalized: Normalized responses.
            fit: Hill curve fit results (IC50, hill_n, etc.).
            summary: Text summary of results.
    """
    # Build layout
    layout = make_plate_layout(doses, n_replicates=n_replicates)

    # Handle list input: reshape to plate
    if isinstance(measurements, (list, np.ndarray)):
        measurements = np.asarray(measurements, dtype=float)
        if measurements.ndim == 1:
            # Flat list: reshape to (n_replicates, n_doses + control_cols)
            n_doses = len(doses)
            n_doses * n_replicates + n_replicates  # drug + control
            plate = np.full((layout["layout"].shape), np.nan)
            idx = 0
            for ci, _dose in enumerate(sorted(doses)):
                for r in range(n_replicates):
                    if idx < len(measurements):
                        plate[r, ci] = measurements[idx]
                        idx += 1
            # Controls
            ctrl_col = layout["layout"].shape[1] - 1
            for r in range(n_replicates):
                if idx < len(measurements):
                    plate[r, ctrl_col] = measurements[idx]
                    idx += 1
            measurements = plate

    # Measure
    plate_data = measure_plate(measurements, layout)

    # Normalize
    norm = normalize_responses(plate_data, background=background)

    # Fit
    if len(norm["doses"]) >= 3:
        fit = auto_ec50(norm["doses"], norm["responses"], top=top, bottom=bottom)
    else:
        fit = {"IC50": np.nan, "hill_n": np.nan, "quality": "insufficient_data"}

    # Summary
    ic50 = fit.get("IC50", np.nan)
    quality = fit.get("quality", "unknown")
    effect = fit.get("effect_type", "unknown")
    summary = (
        f"Dose-response analysis: {effect}, "
        f"IC50={ic50:.4g}, "
        f"Hill n={fit.get('hill_n', 'N/A')}, "
        f"fit quality={quality}"
    )

    return {
        "layout": layout,
        "raw": plate_data,
        "normalized": norm,
        "fit": fit,
        "summary": summary,
    }
