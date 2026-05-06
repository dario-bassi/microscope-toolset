"""Plate reader data processing and analysis.

Functions for processing multi-well plate reader data: background
correction, outlier detection, edge effect correction, and
well-level statistics.

Functions:
    read_plate       -- Parse raw plate reader data into structured format
    background_correct -- Subtract reference wavelength or blank wells
    detect_outliers  -- Flag outlier wells using neighbor comparison
    edge_correction  -- Correct systematic edge effects
    well_statistics  -- Compute per-condition statistics with replicates
"""

import numpy as np


def read_plate(data, n_rows=8, n_cols=12, wavelengths=None):
    """Parse raw plate data into structured format.

    Handles multiple input formats: 2D array (single wavelength),
    3D array (multiple wavelengths), or dict of 2D arrays.

    Args:
        data: Plate reader measurements. Can be:
            - 2D array (n_rows, n_cols) for single wavelength.
            - 3D array (n_wavelengths, n_rows, n_cols).
            - dict mapping wavelength -> 2D array.
        n_rows: Expected number of rows (default 8).
        n_cols: Expected number of columns (default 12).
        wavelengths: List of wavelength labels. Auto-numbered if None.

    Returns:
        dict with:
            data: dict mapping wavelength -> 2D array.
            n_rows: int.
            n_cols: int.
            wavelengths: list of str, wavelength labels.
    """
    if isinstance(data, dict):
        parsed = {}
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            if arr.shape != (n_rows, n_cols):
                raise ValueError(
                    f"Wavelength {k}: expected shape ({n_rows}, {n_cols}), " f"got {arr.shape}"
                )
            parsed[str(k)] = arr
        wl_labels = list(parsed.keys())
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2:
            if arr.shape != (n_rows, n_cols):
                raise ValueError(f"Expected shape ({n_rows}, {n_cols}), got {arr.shape}")
            wl_labels = [wavelengths[0] if wavelengths else "1"]
            parsed = {wl_labels[0]: arr}
        elif arr.ndim == 3:
            n_wl = arr.shape[0]
            if arr.shape[1:] != (n_rows, n_cols):
                raise ValueError(f"Expected shape (N, {n_rows}, {n_cols}), got {arr.shape}")
            if wavelengths is None:
                wl_labels = [str(i + 1) for i in range(n_wl)]
            else:
                wl_labels = [str(w) for w in wavelengths]
            parsed = {wl_labels[i]: arr[i] for i in range(n_wl)}
        else:
            raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")

    return {
        "data": parsed,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "wavelengths": wl_labels,
    }


def background_correct(
    plate_data, method="blank", blank_wells=None, reference_wavelength=None, scale=1.0
):
    """Apply background correction to plate reader data.

    Args:
        plate_data: dict from read_plate() with 'data' key, or 2D array.
        method: 'blank' (subtract blank well mean), 'reference' (subtract
            reference wavelength), or 'median' (subtract plate median).
        blank_wells: list of (row, col) tuples for blank wells.
            Required if method='blank'.
        reference_wavelength: str, wavelength key to use as reference.
            Required if method='reference'.
        scale: float, divide result by this value after correction.

    Returns:
        dict with:
            corrected: dict mapping wavelength -> corrected 2D array.
            background: dict mapping wavelength -> background value used.
    """
    if isinstance(plate_data, np.ndarray):
        plate_data = read_plate(plate_data)

    data = plate_data["data"]
    corrected = {}
    backgrounds = {}

    if method == "blank":
        if blank_wells is None:
            raise ValueError("blank_wells required for method='blank'")
        for wl, arr in data.items():
            bg_values = [
                arr[r, c] for r, c in blank_wells if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]
            ]
            bg = np.mean(bg_values) if bg_values else 0
            corrected[wl] = (arr - bg) / scale
            backgrounds[wl] = bg

    elif method == "reference":
        if reference_wavelength is None:
            raise ValueError("reference_wavelength required for method='reference'")
        ref_key = str(reference_wavelength)
        if ref_key not in data:
            raise ValueError(f"Reference wavelength '{ref_key}' not found")
        ref = data[ref_key]
        for wl, arr in data.items():
            if wl == ref_key:
                continue
            corrected[wl] = (arr - ref) / scale
            backgrounds[wl] = float(ref.mean())

    elif method == "median":
        for wl, arr in data.items():
            bg = float(np.median(arr))
            corrected[wl] = (arr - bg) / scale
            backgrounds[wl] = bg

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "corrected": corrected,
        "background": backgrounds,
    }


def detect_outliers(plate, layout=None, z_threshold=2.0):
    """Flag outlier wells by comparing to condition replicates.

    For each group of replicate wells (same condition), compute
    z-scores and flag values beyond the threshold.

    Args:
        plate: 2D array (n_rows, n_cols) of corrected values.
        layout: 2D array (n_rows, n_cols) of condition labels.
            Wells with the same label are compared. If None, uses
            row-based grouping (each row is a replicate group).
        z_threshold: float, z-score cutoff for outlier detection.

    Returns:
        dict with:
            outlier_mask: 2D bool array, True = outlier.
            n_outliers: int.
            outlier_positions: list of (row, col) tuples.
            z_scores: 2D array of z-scores within each group.
    """
    plate = np.asarray(plate, dtype=float)
    n_rows, n_cols = plate.shape

    if layout is None:
        # Default: row-based groups
        layout = np.zeros_like(plate)
        for r in range(n_rows):
            layout[r, :] = r

    layout = np.asarray(layout, dtype=float)
    outlier_mask = np.zeros_like(plate, dtype=bool)
    z_scores = np.zeros_like(plate, dtype=float)

    # Group by unique layout values (excluding -1 = empty)
    unique_groups = np.unique(layout)
    for g in unique_groups:
        if g < 0:
            continue  # skip empty wells
        mask = layout == g
        values = plate[mask]
        if len(values) < 2:
            continue

        mean = values.mean()
        std = values.std()
        if std < 1e-10:
            continue

        group_z = (plate - mean) / std
        z_scores[mask] = group_z[mask]
        outlier_mask[mask] = np.abs(group_z[mask]) > z_threshold

    positions = list(zip(*np.where(outlier_mask), strict=False)) if outlier_mask.any() else []

    return {
        "outlier_mask": outlier_mask,
        "n_outliers": int(outlier_mask.sum()),
        "outlier_positions": positions,
        "z_scores": z_scores,
    }


def edge_correction(plate, method="multiplicative"):
    """Correct systematic edge effects in plate reader data.

    Edge wells often show higher/lower signal due to evaporation
    or temperature gradients. This function estimates and corrects
    the systematic spatial bias.

    Args:
        plate: 2D array (n_rows, n_cols).
        method: 'multiplicative' (divide by spatial pattern) or
            'additive' (subtract spatial pattern).

    Returns:
        dict with:
            corrected: 2D array, edge-corrected values.
            correction_map: 2D array, the estimated spatial bias.
            edge_ratio: float, mean(edge) / mean(interior).
    """
    plate = np.asarray(plate, dtype=float)
    n_rows, n_cols = plate.shape

    # Identify edge vs interior
    edge_mask = np.zeros_like(plate, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    interior_mask = ~edge_mask

    edge_mean = plate[edge_mask].mean() if edge_mask.any() else 1.0
    interior_mean = plate[interior_mask].mean() if interior_mask.any() else 1.0
    edge_ratio = edge_mean / max(interior_mean, 1e-10)

    # Estimate smooth spatial pattern using row and column medians
    row_medians = np.median(plate, axis=1, keepdims=True)
    col_medians = np.median(plate, axis=0, keepdims=True)
    global_median = np.median(plate)

    # Additive decomposition: pattern = row_effect + col_effect - global
    correction_map = row_medians + col_medians - global_median

    if method == "multiplicative":
        safe_map = np.where(np.abs(correction_map) > 1e-10, correction_map, 1e-10)
        corrected = plate * (global_median / safe_map)
    elif method == "additive":
        corrected = plate - correction_map + global_median
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "corrected": corrected,
        "correction_map": correction_map,
        "edge_ratio": float(edge_ratio),
    }


def well_statistics(plate, layout, exclude_outliers=True, z_threshold=2.0):
    """Compute per-condition statistics from replicate wells.

    Groups wells by condition (from layout), optionally excludes
    outliers, and computes mean, std, SEM, and CV for each condition.

    Args:
        plate: 2D array of measured values.
        layout: 2D array of condition labels. Same label = replicates.
            Negative values are ignored (empty wells).
        exclude_outliers: bool, whether to remove outliers before stats.
        z_threshold: float, z-score threshold for outlier removal.

    Returns:
        dict mapping condition_label -> {
            mean: float, std: float, sem: float, cv: float,
            n: int, values: list, positions: list of (r,c).
        }
    """
    plate = np.asarray(plate, dtype=float)
    layout = np.asarray(layout, dtype=float)

    if exclude_outliers:
        outlier_info = detect_outliers(plate, layout, z_threshold)
        outlier_mask = outlier_info["outlier_mask"]
    else:
        outlier_mask = np.zeros_like(plate, dtype=bool)

    stats_dict = {}
    unique_groups = np.unique(layout)

    for g in unique_groups:
        if g < 0:
            continue  # skip empty

        mask = (layout == g) & (~outlier_mask)
        positions = list(zip(*np.where(layout == g), strict=False))
        values = plate[mask].tolist()
        n = len(values)

        if n == 0:
            stats_dict[float(g)] = {
                "mean": 0.0,
                "std": 0.0,
                "sem": 0.0,
                "cv": 0.0,
                "n": 0,
                "values": [],
                "positions": positions,
            }
            continue

        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
        sem_val = std_val / np.sqrt(n) if n > 0 else 0.0
        cv_val = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else 0.0

        stats_dict[float(g)] = {
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "sem": round(sem_val, 6),
            "cv": round(cv_val, 4),
            "n": n,
            "values": values,
            "positions": positions,
        }

    return stats_dict
