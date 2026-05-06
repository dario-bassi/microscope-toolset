"""Morphological dynamics — track shape changes over time.

Measures how cell/object morphology evolves across a timelapse,
detecting shape transitions, rounding, spreading, elongation,
and morphological heterogeneity. Critical for drug response,
differentiation, and wound healing assays.

Functions:
    track_morphology      -- Measure object properties across frames
    detect_shape_change   -- Find when objects undergo shape transitions
    morphology_timecourse -- Population morphology statistics over time
    spreading_index       -- Track cell spreading/rounding over time
    shape_heterogeneity   -- Measure within-population morphological variability
"""

import numpy as np
from scipy import ndimage


def track_morphology(label_stack, properties=None):
    """Measure object properties across frames.

    Args:
        label_stack: 3D int array (T, H, W) of labeled objects.
            Label 0 = background. Same label across frames = same object.
        properties: list of str or None. Properties to measure.
            Default: ['area', 'eccentricity', 'solidity', 'perimeter'].

    Returns:
        dict with:
            objects: dict mapping label -> dict of property time series.
                e.g. objects[3]['area'] = [100, 105, 110, ...].
            n_objects: int.
            n_frames: int.
            properties: list of str, measured properties.
    """
    label_stack = np.asarray(label_stack, dtype=int)
    n_frames = label_stack.shape[0]

    if properties is None:
        properties = ["area", "eccentricity", "solidity", "perimeter"]

    # Find all unique labels across all frames
    all_labels = set()
    for t in range(n_frames):
        all_labels.update(np.unique(label_stack[t]))
    all_labels.discard(0)  # remove background

    objects = {}
    for lab in sorted(all_labels):
        obj = {prop: [] for prop in properties}
        for t in range(n_frames):
            mask = label_stack[t] == lab
            if mask.sum() == 0:
                for prop in properties:
                    obj[prop].append(np.nan)
                continue

            vals = _measure_single_object(mask, properties)
            for prop in properties:
                obj[prop].append(vals[prop])
        objects[int(lab)] = obj

    return {
        "objects": objects,
        "n_objects": len(objects),
        "n_frames": n_frames,
        "properties": properties,
    }


def detect_shape_change(property_series, window=3, threshold=2.0):
    """Find when an object undergoes a shape transition.

    Detects abrupt changes in a morphological property time series
    using rolling statistics and z-score anomaly detection.

    Args:
        property_series: 1D array of property values over time.
        window: int, rolling window size for baseline statistics.
        threshold: float, z-score threshold for detecting a change.

    Returns:
        dict with:
            change_frames: list of int, frames where changes detected.
            change_magnitudes: list of float, z-score at each change.
            n_changes: int.
            baseline_mean: float, mean of stable period.
            baseline_std: float, std of stable period.
    """
    values = np.asarray(property_series, dtype=float)
    valid = ~np.isnan(values)
    n = len(values)

    if valid.sum() < window + 1:
        return {
            "change_frames": [],
            "change_magnitudes": [],
            "n_changes": 0,
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
        }

    # Use first window frames as baseline
    valid_vals = values[valid]
    baseline = valid_vals[:window]
    bl_mean = float(np.mean(baseline))
    bl_std = float(np.std(baseline))
    if bl_std < 1e-10:
        bl_std = float(np.mean(np.abs(baseline - bl_mean)) + 1e-10)

    change_frames = []
    change_mags = []

    for i in range(window, n):
        if np.isnan(values[i]):
            continue
        z = abs(values[i] - bl_mean) / bl_std
        if z > threshold:
            change_frames.append(int(i))
            change_mags.append(round(float(z), 3))

    return {
        "change_frames": change_frames,
        "change_magnitudes": change_mags,
        "n_changes": len(change_frames),
        "baseline_mean": round(bl_mean, 4),
        "baseline_std": round(bl_std, 4),
    }


def morphology_timecourse(label_stack, properties=None):
    """Population-level morphology statistics over time.

    Args:
        label_stack: 3D int array (T, H, W).
        properties: list of str or None.

    Returns:
        dict with:
            frames: list of int (0..T-1).
            stats: dict mapping property -> dict with 'mean', 'std', 'median'
                arrays (one value per frame).
            n_objects_per_frame: list of int.
    """
    label_stack = np.asarray(label_stack, dtype=int)
    n_frames = label_stack.shape[0]

    if properties is None:
        properties = ["area", "eccentricity", "solidity"]

    stats = {prop: {"mean": [], "std": [], "median": []} for prop in properties}
    n_obj = []

    for t in range(n_frames):
        labels_t = label_stack[t]
        unique_labels = [lbl for lbl in np.unique(labels_t) if lbl > 0]
        n_obj.append(len(unique_labels))

        if not unique_labels:
            for prop in properties:
                stats[prop]["mean"].append(np.nan)
                stats[prop]["std"].append(np.nan)
                stats[prop]["median"].append(np.nan)
            continue

        prop_vals = {prop: [] for prop in properties}
        for lab in unique_labels:
            mask = labels_t == lab
            vals = _measure_single_object(mask, properties)
            for prop in properties:
                prop_vals[prop].append(vals[prop])

        for prop in properties:
            arr = np.array(prop_vals[prop])
            stats[prop]["mean"].append(round(float(np.mean(arr)), 4))
            stats[prop]["std"].append(round(float(np.std(arr)), 4))
            stats[prop]["median"].append(round(float(np.median(arr)), 4))

    return {
        "frames": list(range(n_frames)),
        "stats": stats,
        "n_objects_per_frame": n_obj,
    }


def spreading_index(label_stack):
    """Track cell spreading/rounding over time.

    Spreading index = area / (perimeter^2 / (4*pi)).
    High SI = round/spread, low SI = elongated/retracted.

    Args:
        label_stack: 3D int array (T, H, W).

    Returns:
        dict with:
            mean_si: list of float, mean spreading index per frame.
            population_trend: str, 'spreading'/'rounding'/'stable'.
            initial_si: float, mean SI at first frame.
            final_si: float, mean SI at last frame.
            change_rate: float, SI change per frame.
    """
    label_stack = np.asarray(label_stack, dtype=int)
    n_frames = label_stack.shape[0]

    si_means = []
    for t in range(n_frames):
        labels_t = label_stack[t]
        unique_labels = [lbl for lbl in np.unique(labels_t) if lbl > 0]
        if not unique_labels:
            si_means.append(np.nan)
            continue

        sis = []
        for lab in unique_labels:
            mask = labels_t == lab
            area = float(mask.sum())
            eroded = ndimage.binary_erosion(mask)
            perimeter = float((mask & ~eroded).sum())
            if perimeter > 0:
                si = 4 * np.pi * area / (perimeter * perimeter)
            else:
                si = 1.0
            sis.append(si)
        si_means.append(round(float(np.mean(sis)), 4))

    si_arr = np.array(si_means)
    valid = ~np.isnan(si_arr)

    if valid.sum() < 2:
        return {
            "mean_si": si_means,
            "population_trend": "stable",
            "initial_si": si_means[0] if si_means else 0.0,
            "final_si": si_means[-1] if si_means else 0.0,
            "change_rate": 0.0,
        }

    valid_si = si_arr[valid]
    valid_idx = np.where(valid)[0]
    slope = float(np.polyfit(valid_idx, valid_si, 1)[0])

    if abs(slope) < 0.005:
        trend = "stable"
    elif slope > 0:
        trend = "spreading"
    else:
        trend = "rounding"

    return {
        "mean_si": si_means,
        "population_trend": trend,
        "initial_si": round(float(valid_si[0]), 4),
        "final_si": round(float(valid_si[-1]), 4),
        "change_rate": round(slope, 6),
    }


def shape_heterogeneity(label_stack, frame_index=-1):
    """Measure within-population morphological variability.

    High heterogeneity may indicate mixed cell types, drug response
    variability, or ongoing differentiation.

    Args:
        label_stack: 3D int array (T, H, W).
        frame_index: int, which frame to analyze (-1 = last).

    Returns:
        dict with:
            area_cv: float, coefficient of variation of areas.
            eccentricity_cv: float, CV of eccentricities.
            solidity_cv: float, CV of solidities.
            heterogeneity_score: float, mean of CVs (0=homogeneous).
            n_objects: int.
            outlier_labels: list of int, objects with extreme morphology.
    """
    label_stack = np.asarray(label_stack, dtype=int)
    frame = label_stack[frame_index]
    unique_labels = [lbl for lbl in np.unique(frame) if lbl > 0]
    n = len(unique_labels)

    if n < 3:
        return {
            "area_cv": 0.0,
            "eccentricity_cv": 0.0,
            "solidity_cv": 0.0,
            "heterogeneity_score": 0.0,
            "n_objects": n,
            "outlier_labels": [],
        }

    areas, eccs, sols = [], [], []
    for lab in unique_labels:
        mask = frame == lab
        vals = _measure_single_object(mask, ["area", "eccentricity", "solidity"])
        areas.append(vals["area"])
        eccs.append(vals["eccentricity"])
        sols.append(vals["solidity"])

    areas, eccs, sols = np.array(areas), np.array(eccs), np.array(sols)

    def _cv(arr):
        m = np.mean(arr)
        return float(np.std(arr) / m) if m > 0 else 0.0

    area_cv = _cv(areas)
    ecc_cv = _cv(eccs) if np.mean(eccs) > 0.01 else 0.0
    sol_cv = _cv(sols)
    hetero = (area_cv + ecc_cv + sol_cv) / 3

    # Outliers: >= 2σ from mean in any property
    outliers = set()
    for arr, label_list in [(areas, unique_labels), (eccs, unique_labels)]:
        m, s = np.mean(arr), np.std(arr)
        if s > 0:
            for j, lab in enumerate(label_list):
                if abs(arr[j] - m) >= 2 * s:
                    outliers.add(lab)

    return {
        "area_cv": round(area_cv, 4),
        "eccentricity_cv": round(ecc_cv, 4),
        "solidity_cv": round(sol_cv, 4),
        "heterogeneity_score": round(hetero, 4),
        "n_objects": n,
        "outlier_labels": sorted(outliers),
    }


# ── Private helpers ──────────────────────────────────────────────────


def _measure_single_object(mask, properties):
    """Measure properties of a single binary mask."""
    vals = {}
    area = float(mask.sum())

    for prop in properties:
        if prop == "area":
            vals["area"] = area
        elif prop == "perimeter":
            eroded = ndimage.binary_erosion(mask)
            vals["perimeter"] = float((mask & ~eroded).sum())
        elif prop == "eccentricity":
            ys, xs = np.where(mask)
            if len(ys) < 5:
                vals["eccentricity"] = 0.0
            else:
                cov = np.cov(xs.astype(float), ys.astype(float))
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.maximum(eigvals, 0)
                if eigvals[-1] > 0:
                    vals["eccentricity"] = float(np.sqrt(1 - eigvals[0] / eigvals[-1]))
                else:
                    vals["eccentricity"] = 0.0
        elif prop == "solidity":
            from scipy.spatial import ConvexHull

            ys, xs = np.where(mask)
            if len(ys) < 4:
                vals["solidity"] = 1.0
            else:
                pts = np.column_stack([xs, ys])
                try:
                    hull = ConvexHull(pts)
                    vals["solidity"] = float(area / hull.volume)
                except Exception:
                    vals["solidity"] = 1.0
        elif prop == "centroid":
            cy, cx = ndimage.center_of_mass(mask)
            vals["centroid"] = (float(cy), float(cx))
        elif prop == "compactness":
            eroded = ndimage.binary_erosion(mask)
            perimeter = float((mask & ~eroded).sum())
            if perimeter > 0:
                vals["compactness"] = 4 * np.pi * area / (perimeter**2)
            else:
                vals["compactness"] = 1.0

    return vals
