"""Detection quality assurance and sanity checks.

Validates detection results against physical expectations to catch
systematic errors BEFORE submission. Addresses the #1 failure pattern:
correct detection with wrong methodology or unreasonable results.

Key functions:
    check_density     -- Validate cell count vs expected density for FOV
    check_size_range  -- Flag objects outside expected size range
    check_edge_bias   -- Detect systematic edge exclusion bias
    check_bimodal     -- Verify a distribution is bimodal (expected for classification)
    validate_count    -- Combined sanity check for cell counting results
"""

import numpy as np


def check_density(
    n_cells, fov_area_um2, expected_density=None, min_density=0.0001, max_density=0.01
):
    """Check if detected cell count is physically reasonable.

    Args:
        n_cells: Number of detected cells.
        fov_area_um2: Field of view area in µm².
        expected_density: Expected density (cells/µm²). If None, uses
            min/max range.
        min_density: Minimum plausible cell density (cells/µm²).
            Default 0.0001 = ~1 cell per 100x100 µm.
        max_density: Maximum plausible cell density (cells/µm²).
            Default 0.01 = ~1 cell per 10x10 µm (very dense).

    Returns:
        dict with:
            density: Measured density (cells/µm²).
            status: 'ok', 'too_sparse', 'too_dense', or 'suspicious'.
            message: Human-readable assessment.
    """
    if fov_area_um2 <= 0:
        return {"density": 0, "status": "error", "message": "Invalid FOV area"}

    density = n_cells / fov_area_um2

    if expected_density is not None:
        ratio = density / expected_density if expected_density > 0 else float("inf")
        if 0.5 <= ratio <= 2.0:
            return {
                "density": density,
                "status": "ok",
                "message": f"{n_cells} cells in {fov_area_um2:.0f} µm² "
                f"(density {density:.4f}/µm², expected {expected_density:.4f})",
            }
        elif ratio < 0.5:
            return {
                "density": density,
                "status": "too_sparse",
                "message": f"Detected {n_cells} but expected ~{int(expected_density * fov_area_um2)}. "
                f"May be undercounting.",
            }
        else:
            return {
                "density": density,
                "status": "too_dense",
                "message": f"Detected {n_cells} but expected ~{int(expected_density * fov_area_um2)}. "
                f"May be overcounting or detecting noise.",
            }

    if density < min_density:
        return {
            "density": density,
            "status": "too_sparse",
            "message": f"{n_cells} cells in {fov_area_um2:.0f} µm² "
            f"seems very sparse (density {density:.6f}/µm²)",
        }
    elif density > max_density:
        return {
            "density": density,
            "status": "too_dense",
            "message": f"{n_cells} cells in {fov_area_um2:.0f} µm² "
            f"seems overcrowded (density {density:.4f}/µm²). Check for noise.",
        }
    else:
        return {
            "density": density,
            "status": "ok",
            "message": f"{n_cells} cells at density {density:.5f}/µm²",
        }


def check_size_range(areas, pixel_size_um=1.0, min_cell_um2=5, max_cell_um2=5000):
    """Check if detected object sizes are physically reasonable.

    Args:
        areas: List or array of object areas in pixels.
        pixel_size_um: Pixel size in µm.
        min_cell_um2: Minimum plausible cell area (µm²).
            Default 5 = tiny bacteria.
        max_cell_um2: Maximum plausible cell area (µm²).
            Default 5000 = very large cell or small cluster.

    Returns:
        dict with:
            n_total: Total objects.
            n_valid: Objects within size range.
            n_too_small: Objects below minimum.
            n_too_large: Objects above maximum.
            median_area_um2: Median area in µm².
            status: 'ok', 'many_small', 'many_large', or 'mixed'.
            message: Human-readable assessment.
    """
    areas = np.asarray(areas, dtype=float)
    px_area = pixel_size_um**2
    areas_um2 = areas * px_area

    n_total = len(areas)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_valid": 0,
            "n_too_small": 0,
            "n_too_large": 0,
            "median_area_um2": 0,
            "status": "empty",
            "message": "No objects detected",
        }

    too_small = int((areas_um2 < min_cell_um2).sum())
    too_large = int((areas_um2 > max_cell_um2).sum())
    n_valid = n_total - too_small - too_large

    median_um2 = float(np.median(areas_um2))

    if too_small > n_total * 0.3:
        status = "many_small"
        message = f"{too_small}/{n_total} objects below {min_cell_um2} µm² — likely noise"
    elif too_large > n_total * 0.3:
        status = "many_large"
        message = f"{too_large}/{n_total} objects above {max_cell_um2} µm² — clusters or tissue"
    elif too_small + too_large > n_total * 0.3:
        status = "mixed"
        message = f"{too_small} too small, {too_large} too large out of {n_total}"
    else:
        status = "ok"
        message = f"{n_valid}/{n_total} objects in valid size range. Median: {median_um2:.1f} µm²"

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_too_small": too_small,
        "n_too_large": too_large,
        "median_area_um2": median_um2,
        "status": status,
        "message": message,
    }


def check_edge_bias(centroids, fov_size, margin=10):
    """Check for systematic edge exclusion bias.

    Too aggressive edge exclusion can cause significant undercounting
    on dense fields.

    Args:
        centroids: (N, 2) array of (y, x) positions.
        fov_size: (height, width) of the FOV.
        margin: Edge margin being used for exclusion.

    Returns:
        dict with:
            n_total: Cells detected (after exclusion).
            n_excluded_estimate: Estimated cells excluded by margin.
            exclusion_fraction: Fraction of FOV area excluded.
            message: Assessment and recommendation.
    """
    h, w = fov_size
    fov_area = h * w
    interior_area = max(1, (h - 2 * margin) * (w - 2 * margin))
    exclusion_frac = 1.0 - interior_area / fov_area

    centroids = np.atleast_2d(centroids)
    n_total = len(centroids)

    if n_total == 0:
        return {
            "n_total": 0,
            "n_excluded_estimate": 0,
            "exclusion_fraction": exclusion_frac,
            "message": "No cells detected",
        }

    # Estimate excluded cells assuming uniform distribution
    n_excluded_estimate = int(round(n_total * exclusion_frac / (1 - exclusion_frac)))

    if exclusion_frac > 0.15:
        message = (
            f"Edge margin={margin}px excludes {exclusion_frac:.1%} of FOV area. "
            f"Estimated {n_excluded_estimate} cells lost. "
            f"Consider reducing margin for this sample."
        )
    else:
        message = f"Edge margin={margin}px excludes {exclusion_frac:.1%} of area — acceptable."

    return {
        "n_total": n_total,
        "n_excluded_estimate": n_excluded_estimate,
        "exclusion_fraction": exclusion_frac,
        "message": message,
    }


def check_bimodal(values, min_separation=1.0):
    """Check if a distribution is bimodal (expected for binary classification).

    Uses the dip statistic approximation: if the histogram has a clear
    valley between two peaks, the distribution is bimodal.

    Args:
        values: 1D array of measurements.
        min_separation: Minimum required separation between modes in
            units of standard deviation.

    Returns:
        dict with:
            is_bimodal: True if distribution appears bimodal.
            n_modes: Estimated number of modes (1 or 2).
            valley: Value at the valley between modes (if bimodal).
            mode1_mean: Mean of lower mode.
            mode2_mean: Mean of upper mode.
            separation_sigma: Separation between modes in units of overall std.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 10:
        return {
            "is_bimodal": False,
            "n_modes": 1,
            "valley": None,
            "mode1_mean": float(values.mean()) if n > 0 else 0,
            "mode2_mean": None,
            "separation_sigma": 0,
        }

    # Histogram-based approach
    n_bins = min(50, max(10, n // 5))
    counts, edges = np.histogram(values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth histogram
    if len(counts) >= 5:
        kernel = np.ones(3) / 3
        counts_smooth = np.convolve(counts, kernel, mode="same")
    else:
        counts_smooth = counts.astype(float)

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(counts_smooth) - 1):
        if counts_smooth[i] > counts_smooth[i - 1] and counts_smooth[i] > counts_smooth[i + 1]:
            peaks.append(i)

    if len(peaks) >= 2:
        # Find the deepest valley between the two highest peaks
        peak_heights = [(counts_smooth[p], p) for p in peaks]
        peak_heights.sort(reverse=True)
        p1, p2 = sorted([peak_heights[0][1], peak_heights[1][1]])

        valley_idx = p1 + np.argmin(counts_smooth[p1 : p2 + 1])
        valley_val = float(centers[valley_idx])

        mode1_vals = values[values < valley_val]
        mode2_vals = values[values >= valley_val]

        mode1_mean = float(mode1_vals.mean()) if len(mode1_vals) > 0 else 0
        mode2_mean = float(mode2_vals.mean()) if len(mode2_vals) > 0 else 0

        overall_std = float(np.std(values))
        separation = abs(mode2_mean - mode1_mean) / max(overall_std, 1e-10)

        # Valley must be significantly lower than both peaks for true bimodality
        valley_depth = counts_smooth[valley_idx]
        peak_min_height = min(counts_smooth[p1], counts_smooth[p2])
        valley_ratio = valley_depth / max(peak_min_height, 1e-10)

        is_bimodal = (
            separation >= min_separation
            and valley_ratio < 0.75
            and len(mode1_vals) >= n * 0.1
            and len(mode2_vals) >= n * 0.1
        )

        return {
            "is_bimodal": bool(is_bimodal),
            "n_modes": 2 if is_bimodal else 1,
            "valley": valley_val,
            "mode1_mean": mode1_mean,
            "mode2_mean": mode2_mean,
            "separation_sigma": round(separation, 2),
        }

    return {
        "is_bimodal": False,
        "n_modes": 1,
        "valley": None,
        "mode1_mean": float(values.mean()),
        "mode2_mean": None,
        "separation_sigma": 0,
    }


def validate_count(
    n_cells,
    fov_area_um2,
    areas_px=None,
    pixel_size_um=1.0,
    expected_density=None,
    sample_type="generic",
):
    """Combined sanity check for cell counting results.

    Runs density check, size check, and provides overall assessment.

    Args:
        n_cells: Number of detected cells.
        fov_area_um2: Field of view area in µm².
        areas_px: Optional array of cell areas in pixels.
        pixel_size_um: Pixel size in µm.
        expected_density: Expected density if known.
        sample_type: 'bacteria', 'mammalian', 'tissue', or 'generic'.
            Sets size range expectations.

    Returns:
        dict with:
            status: 'ok', 'warning', or 'error'.
            checks: Dict of individual check results.
            message: Overall assessment.
            suggestions: List of suggested fixes if issues found.
    """
    # Size range by sample type
    size_ranges = {
        "bacteria": (0.5, 20),
        "mammalian": (50, 5000),
        "tissue": (20, 2000),
        "generic": (5, 5000),
    }
    min_sz, max_sz = size_ranges.get(sample_type, (5, 5000))

    checks = {}
    suggestions = []

    # Density check
    checks["density"] = check_density(n_cells, fov_area_um2, expected_density=expected_density)
    if checks["density"]["status"] != "ok":
        if checks["density"]["status"] == "too_dense":
            suggestions.append("Verify at higher magnification — may be detecting noise")
            suggestions.append("Try cross-validating with a second channel")
        elif checks["density"]["status"] == "too_sparse":
            suggestions.append("Check if sample is in the FOV — try 10x overview first")
            suggestions.append("Lower detection threshold or check edge exclusion")

    # Size check
    if areas_px is not None and len(areas_px) > 0:
        checks["size"] = check_size_range(areas_px, pixel_size_um, min_sz, max_sz)
        if checks["size"]["status"] == "many_small":
            suggestions.append(
                f"Many objects below {min_sz} µm² — likely noise. "
                f"Increase min_area or use morphological filtering"
            )
        elif checks["size"]["status"] == "many_large":
            suggestions.append(
                f"Many objects above {max_sz} µm² — clusters? " f"Try watershed splitting"
            )

    # Overall status
    statuses = [c["status"] for c in checks.values()]
    if all(s == "ok" for s in statuses):
        status = "ok"
        message = f"{n_cells} cells detected — all checks passed"
    elif any(s in ("too_dense", "too_sparse", "many_small") for s in statuses):
        status = "warning"
        messages = [c["message"] for c in checks.values() if c["status"] != "ok"]
        message = "; ".join(messages)
    else:
        status = "ok"
        message = f"{n_cells} cells detected — minor issues noted"

    return {
        "status": status,
        "checks": checks,
        "message": message,
        "suggestions": suggestions,
    }
