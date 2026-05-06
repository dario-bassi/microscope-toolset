"""Colony formation and clonogenic assay analysis.

Detects, counts, and measures cell colonies from clonogenic assays.
Computes plating efficiency, surviving fraction, and colony size
distributions.

Functions:
    detect_colonies       -- Find colonies in brightfield/stained images
    measure_colonies      -- Measure area, diameter, intensity of each colony
    plating_efficiency    -- Compute PE from colony count and cells plated
    surviving_fraction    -- Compute SF relative to control
    colony_size_classes   -- Classify colonies by size
"""

import numpy as np
from scipy import ndimage


def detect_colonies(image, min_area=50, max_area=None, threshold=None, dark_colonies=True):
    """Detect colonies in a plate image.

    Args:
        image: 2D array, brightfield or stained plate image.
        min_area: int, minimum colony area in pixels.
        max_area: int or None, maximum colony area.
        threshold: float or None, intensity threshold.
            If None, uses Otsu's method.
        dark_colonies: bool, True if colonies are darker than background
            (e.g., crystal violet stain on white plate).

    Returns:
        dict with:
            labeled: 2D int array, labeled colony mask.
            n_colonies: int.
            centroids: list of (y, x) tuples.
            areas: list of floats.
    """
    image = np.asarray(image, dtype=float)

    if threshold is None:
        # Simple Otsu approximation
        hist, edges = np.histogram(image.ravel(), bins=256)
        total = image.size
        sum_all = np.sum(np.arange(256) * hist)
        w0, sum0 = 0, 0
        best_thresh, best_var = 0, 0
        for i in range(256):
            w0 += hist[i]
            if w0 == 0:
                continue
            w1 = total - w0
            if w1 == 0:
                break
            sum0 += i * hist[i]
            m0 = sum0 / w0
            m1 = (sum_all - sum0) / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best_var:
                best_var = var
                best_thresh = edges[i]
        threshold = best_thresh

    if dark_colonies:
        mask = image < threshold
    else:
        mask = image > threshold

    # Clean up
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)

    labeled, n = ndimage.label(mask)

    # Filter by size
    areas = []
    centroids = []
    keep_mask = np.zeros_like(labeled)
    new_label = 0

    for i in range(1, n + 1):
        region = labeled == i
        area = region.sum()
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        new_label += 1
        keep_mask[region] = new_label
        areas.append(float(area))
        cy, cx = ndimage.center_of_mass(region)
        centroids.append((round(float(cy), 1), round(float(cx), 1)))

    return {
        "labeled": keep_mask,
        "n_colonies": new_label,
        "centroids": centroids,
        "areas": areas,
    }


def measure_colonies(image, labeled, pixel_size=1.0):
    """Measure properties of detected colonies.

    Args:
        image: 2D array, original image.
        labeled: 2D int array, labeled colony mask.
        pixel_size: float, µm per pixel.

    Returns:
        dict with:
            measurements: list of dicts per colony with:
                label, area_px, area_um2, diameter_um,
                mean_intensity, centroid.
            mean_area: float, mean colony area in µm².
            mean_diameter: float, mean equivalent diameter in µm.
            total_area: float, total colony coverage in µm².
    """
    image = np.asarray(image, dtype=float)
    labeled = np.asarray(labeled)

    n = labeled.max()
    measurements = []

    for i in range(1, n + 1):
        region = labeled == i
        area_px = float(region.sum())
        area_um2 = area_px * pixel_size**2
        diameter = 2 * np.sqrt(area_px / np.pi) * pixel_size
        intensity = float(image[region].mean())
        cy, cx = ndimage.center_of_mass(region)

        measurements.append(
            {
                "label": i,
                "area_px": round(area_px, 1),
                "area_um2": round(area_um2, 1),
                "diameter_um": round(diameter, 1),
                "mean_intensity": round(intensity, 1),
                "centroid": (round(float(cy), 1), round(float(cx), 1)),
            }
        )

    areas = [m["area_um2"] for m in measurements]
    diameters = [m["diameter_um"] for m in measurements]

    return {
        "measurements": measurements,
        "mean_area": round(float(np.mean(areas)), 1) if areas else 0.0,
        "mean_diameter": round(float(np.mean(diameters)), 1) if diameters else 0.0,
        "total_area": round(float(np.sum(areas)), 1),
    }


def plating_efficiency(n_colonies, n_cells_plated):
    """Compute plating efficiency.

    PE = (number of colonies / number of cells plated) × 100

    Args:
        n_colonies: int, number of colonies formed.
        n_cells_plated: int, number of cells initially plated.

    Returns:
        dict with:
            plating_efficiency: float, PE as percentage.
            n_colonies: int.
            n_cells_plated: int.
    """
    if n_cells_plated <= 0:
        return {
            "plating_efficiency": 0.0,
            "n_colonies": n_colonies,
            "n_cells_plated": n_cells_plated,
        }

    pe = (n_colonies / n_cells_plated) * 100
    return {
        "plating_efficiency": round(pe, 2),
        "n_colonies": n_colonies,
        "n_cells_plated": n_cells_plated,
    }


def surviving_fraction(n_colonies_treated, n_cells_treated, pe_control):
    """Compute surviving fraction relative to control.

    SF = (colonies_treated / cells_treated) / (PE_control / 100)

    Args:
        n_colonies_treated: int, colonies in treated condition.
        n_cells_treated: int, cells plated in treated condition.
        pe_control: float, plating efficiency of untreated control (%).

    Returns:
        dict with:
            surviving_fraction: float (0-1).
            log_sf: float, log10(SF) for survival curves.
    """
    if pe_control <= 0 or n_cells_treated <= 0:
        return {
            "surviving_fraction": 0.0,
            "log_sf": float("-inf"),
        }

    sf = (n_colonies_treated / n_cells_treated) / (pe_control / 100)
    sf = min(sf, 1.0)  # Can't survive more than 100%

    log_sf = float(np.log10(sf)) if sf > 0 else float("-inf")

    return {
        "surviving_fraction": round(sf, 4),
        "log_sf": round(log_sf, 4) if np.isfinite(log_sf) else float("-inf"),
    }


def colony_size_classes(areas, thresholds=None):
    """Classify colonies by size into categories.

    Default classes: small (<1000 µm²), medium (1000-5000), large (>5000).

    Args:
        areas: list or array of colony areas (µm²).
        thresholds: list of 2 values [small_max, medium_max] or None.

    Returns:
        dict with:
            classifications: list of 'small'/'medium'/'large' per colony.
            counts: dict of category counts.
            fractions: dict of category fractions.
            mean_per_class: dict of mean area per class.
    """
    areas = np.asarray(areas, dtype=float)

    if thresholds is None:
        thresholds = [1000, 5000]

    small_max, medium_max = thresholds

    classifications = []
    for a in areas:
        if a < small_max:
            classifications.append("small")
        elif a < medium_max:
            classifications.append("medium")
        else:
            classifications.append("large")

    counts = {"small": 0, "medium": 0, "large": 0}
    area_sums = {"small": [], "medium": [], "large": []}
    for cls, a in zip(classifications, areas, strict=False):
        counts[cls] += 1
        area_sums[cls].append(a)

    total = max(len(areas), 1)
    fractions = {k: round(v / total, 4) for k, v in counts.items()}
    mean_per_class = {k: round(float(np.mean(v)), 1) if v else 0.0 for k, v in area_sums.items()}

    return {
        "classifications": classifications,
        "counts": counts,
        "fractions": fractions,
        "mean_per_class": mean_per_class,
    }
