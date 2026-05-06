"""Measurement validation and sanity checking.

Cross-checks quantitative measurements against biological priors,
physical constraints, and internal consistency. Catches common errors
like wrong magnification, inverted channels, or miscalibrated thresholds.

Functions:
    validate_cell_count     -- Check count is reasonable for FOV
    validate_size           -- Check object sizes are biologically plausible
    validate_speed          -- Check velocities against known ranges
    validate_frequency      -- Check oscillation frequency is realistic
    validate_concentration  -- Check cell density is consistent with count
    run_sanity_checks       -- Run all applicable validators on a measurement set
"""

import numpy as np


def validate_cell_count(count, fov_area_um2, cell_type="generic", magnification=None):
    """Check if cell count is reasonable for the field of view.

    Args:
        count: int, detected cell count.
        fov_area_um2: float, field of view area in µm².
        cell_type: str, one of 'generic', 'bacteria', 'mammalian',
            'yeast', 'rbc', 'wbc', 'neuron'.
        magnification: int or None, objective magnification.

    Returns:
        dict with:
            valid: bool, whether count is within expected range.
            density: float, cells per mm².
            expected_range: tuple (min, max) expected count.
            warnings: list of str.
    """
    density = count / fov_area_um2 * 1e6  # cells per mm²
    warnings = []

    # Expected densities (cells/mm²) by cell type
    density_ranges = {
        "generic": (10, 50000),
        "bacteria": (100, 500000),
        "mammalian": (10, 5000),
        "yeast": (50, 20000),
        "rbc": (1000, 100000),
        "wbc": (1, 500),
        "neuron": (5, 2000),
    }

    min_d, max_d = density_ranges.get(cell_type, (1, 100000))
    expected_count_min = max(0, int(min_d * fov_area_um2 / 1e6))
    expected_count_max = int(max_d * fov_area_um2 / 1e6) + 1

    valid = expected_count_min <= count <= expected_count_max

    if count == 0:
        warnings.append("Zero cells detected — check threshold/channel")
    elif count < expected_count_min:
        warnings.append(
            f"Count {count} below expected minimum " f"{expected_count_min} for {cell_type}"
        )
    elif count > expected_count_max:
        warnings.append(
            f"Count {count} above expected maximum " f"{expected_count_max} for {cell_type}"
        )

    if magnification and magnification >= 40 and count > 200:
        warnings.append(
            f"High count ({count}) at {magnification}x — " "verify not overcounting noise/texture"
        )

    return {
        "valid": valid,
        "density": round(density, 1),
        "expected_range": (expected_count_min, expected_count_max),
        "warnings": warnings,
    }


def validate_size(areas_um2, cell_type="generic"):
    """Check if object sizes are biologically plausible.

    Args:
        areas_um2: array-like, object areas in µm².
        cell_type: str, cell type for size lookup.

    Returns:
        dict with:
            valid: bool, whether median size is in expected range.
            median_area: float.
            expected_range: tuple (min, max) expected area in µm².
            fraction_outliers: float, fraction of objects outside range.
            warnings: list of str.
    """
    areas = np.asarray(areas_um2, dtype=float)
    if len(areas) == 0:
        return {
            "valid": False,
            "median_area": 0,
            "expected_range": (0, 0),
            "fraction_outliers": 1.0,
            "warnings": ["No objects to validate"],
        }

    # Expected area ranges (µm²) by cell type
    area_ranges = {
        "generic": (5, 5000),
        "bacteria": (0.5, 10),
        "mammalian": (100, 5000),
        "yeast": (10, 200),
        "rbc": (30, 100),
        "wbc": (50, 400),
        "neuron_soma": (100, 2000),
        "nucleus": (20, 500),
    }

    min_a, max_a = area_ranges.get(cell_type, (1, 10000))
    median_area = float(np.median(areas))
    valid = min_a <= median_area <= max_a

    outliers = ((areas < min_a * 0.5) | (areas > max_a * 2.0)).sum()
    frac_outliers = outliers / len(areas)

    warnings = []
    if not valid:
        warnings.append(
            f"Median area {median_area:.1f} µm² outside expected "
            f"range [{min_a}, {max_a}] for {cell_type}"
        )
    if frac_outliers > 0.3:
        warnings.append(
            f"{frac_outliers:.0%} of objects are size outliers — " "check threshold or segmentation"
        )

    return {
        "valid": valid,
        "median_area": round(median_area, 1),
        "expected_range": (min_a, max_a),
        "fraction_outliers": round(frac_outliers, 3),
        "warnings": warnings,
    }


def validate_speed(speeds, organism="generic", pixel_size=1.0, dt=1.0):
    """Check if measured speeds are physiologically realistic.

    Args:
        speeds: array-like, speeds in µm/s or px/frame.
        organism: str, type for speed lookup.
        pixel_size: float, µm per pixel (for conversion if needed).
        dt: float, seconds per frame.

    Returns:
        dict with:
            valid: bool.
            median_speed: float, in µm/s.
            expected_range: tuple (min, max) in µm/s.
            warnings: list of str.
    """
    speeds = np.asarray(speeds, dtype=float)
    if len(speeds) == 0:
        return {
            "valid": False,
            "median_speed": 0,
            "expected_range": (0, 0),
            "warnings": ["No speeds"],
        }

    # Convert if in px/frame
    speeds_ums = speeds * pixel_size / dt if dt > 0 else speeds

    # Expected speed ranges (µm/s) by organism
    speed_ranges = {
        "generic": (0, 100),
        "bacteria": (1, 50),
        "mammalian": (0.01, 5),
        "neutrophil": (0.5, 30),
        "c_elegans": (10, 200),
        "sperm": (20, 200),
        "vesicle": (0.01, 5),
    }

    min_s, max_s = speed_ranges.get(organism, (0, 1000))
    median_speed = float(np.median(speeds_ums))
    valid = min_s <= median_speed <= max_s

    warnings = []
    if not valid:
        warnings.append(
            f"Median speed {median_speed:.1f} µm/s outside "
            f"expected [{min_s}, {max_s}] for {organism}"
        )

    return {
        "valid": valid,
        "median_speed": round(median_speed, 2),
        "expected_range": (min_s, max_s),
        "warnings": warnings,
    }


def validate_frequency(freq_hz, signal_type="generic"):
    """Check if oscillation frequency is realistic.

    Args:
        freq_hz: float, measured frequency in Hz.
        signal_type: str, type of oscillating signal.

    Returns:
        dict with:
            valid: bool.
            expected_range: tuple (min, max) in Hz.
            warnings: list of str.
    """
    freq_ranges = {
        "generic": (0.001, 100),
        "cardiac": (0.5, 5.0),
        "calcium_transient": (0.01, 2.0),
        "cell_cycle": (0.00001, 0.001),
        "breathing": (0.1, 1.0),
        "neural_burst": (0.1, 50),
    }

    min_f, max_f = freq_ranges.get(signal_type, (0.001, 100))
    valid = min_f <= freq_hz <= max_f

    warnings = []
    if not valid:
        warnings.append(
            f"Frequency {freq_hz:.3f} Hz outside expected " f"[{min_f}, {max_f}] for {signal_type}"
        )

    return {
        "valid": valid,
        "expected_range": (min_f, max_f),
        "warnings": warnings,
    }


def validate_concentration(count, volume_ul=None, dilution_factor=1.0):
    """Check if cell concentration is reasonable.

    Args:
        count: int, cells in a defined volume.
        volume_ul: float, volume in µL (e.g., hemocytometer chamber).
        dilution_factor: float, dilution applied before counting.

    Returns:
        dict with:
            concentration: float, cells/mL.
            valid: bool.
            warnings: list of str.
    """
    if volume_ul is None or volume_ul <= 0:
        return {"concentration": 0, "valid": False, "warnings": ["No volume specified"]}

    conc = count / (volume_ul * 1e-3) * dilution_factor  # cells/mL
    valid = 1e3 <= conc <= 1e8  # typical range for cell suspensions

    warnings = []
    if conc < 1e3:
        warnings.append(f"Very low concentration ({conc:.0f}/mL) — " "check for under-counting")
    if conc > 1e8:
        warnings.append(
            f"Very high concentration ({conc:.0e}/mL) — "
            "check for over-counting or wrong dilution"
        )

    return {
        "concentration": round(conc),
        "valid": valid,
        "warnings": warnings,
    }


def run_sanity_checks(measurements, fov_um2=None, pixel_size=None, cell_type="generic"):
    """Run all applicable validators on a measurement dictionary.

    Args:
        measurements: dict with keys like 'count', 'areas', 'speeds', etc.
        fov_um2: float, field of view area in µm².
        pixel_size: float, µm per pixel.
        cell_type: str, cell type for validation.

    Returns:
        dict with:
            all_valid: bool, whether all checks passed.
            checks: dict of check_name → result dict.
            warnings: list of all warnings.
    """
    checks = {}
    all_warnings = []

    if "count" in measurements and fov_um2:
        result = validate_cell_count(measurements["count"], fov_um2, cell_type)
        checks["cell_count"] = result
        all_warnings.extend(result["warnings"])

    if "areas" in measurements:
        areas = measurements["areas"]
        if pixel_size:
            areas_um2 = np.asarray(areas) * pixel_size**2
        else:
            areas_um2 = np.asarray(areas)
        result = validate_size(areas_um2, cell_type)
        checks["size"] = result
        all_warnings.extend(result["warnings"])

    if "speeds" in measurements:
        result = validate_speed(
            measurements["speeds"],
            organism=cell_type,
            pixel_size=pixel_size or 1.0,
            dt=measurements.get("dt", 1.0),
        )
        checks["speed"] = result
        all_warnings.extend(result["warnings"])

    if "frequency" in measurements:
        result = validate_frequency(
            measurements["frequency"],
            signal_type=measurements.get("signal_type", "generic"),
        )
        checks["frequency"] = result
        all_warnings.extend(result["warnings"])

    all_valid = all(c.get("valid", True) for c in checks.values())

    return {
        "all_valid": all_valid,
        "checks": checks,
        "warnings": all_warnings,
    }
