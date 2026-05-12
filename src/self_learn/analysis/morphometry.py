"""Morphometry: object segmentation, size measurement, and population analysis.

Provides reusable functions for the segment → measure → classify workflow
common in tissue pathology and cell biology.
"""

import numpy as np
from skimage import measure, morphology, filters


def segment_nuclei(image, method='otsu', min_area=20, max_area=None,
                   fill_holes=True, pixel_size_um=None,
                   min_area_um2=None, max_area_um2=None):
    """Segment nuclei from a fluorescence image.

    Physical-unit defaults (when ``pixel_size_um`` is provided):
    ``min_area_um2=20`` (tiny cytoplasm/speckle cutoff) and no upper bound
    unless ``max_area_um2`` is given. Nuclei are typically 20–500 µm²;
    pick `max_area_um2=500` if you want to reject fused/debris blobs.

    The raw pixel defaults (``min_area=20``) only make sense at ~1 µm/px
    (10x). At 40x (0.25 µm/px) the same 20-px² floor corresponds to 1.25 µm²
    which admits noise speckles; at 4x (2 µm/px) it rejects real small
    nuclei. Pass ``pixel_size_um`` from ``get_config(core).pixel_size_um``
    whenever magnification can vary.

    Args:
        image: 2D grayscale image (bright nuclei on dark background).
        method: Thresholding method ('otsu', 'li', 'triangle', or int/float
            for manual threshold).
        min_area: Minimum object area in pixels. Ignored if ``pixel_size_um``
            is given.
        max_area: Maximum object area in pixels. Ignored if ``pixel_size_um``
            is given. None = no upper limit.
        fill_holes: Whether to fill holes in segmented objects.
        pixel_size_um: Optional float from ``get_config(core).pixel_size_um``.
            **Recommended** — triggers physical-unit sizing.
        min_area_um2: Optional float, overrides physical default (20 µm²).
        max_area_um2: Optional float. If given, overrides ``max_area``.

    Returns:
        dict with:
            labeled: 2D labeled image (0 = background).
            props: list of regionprops objects.
            binary: 2D boolean mask.
            threshold: threshold value used.
            areas_um2: (N,) np.ndarray of object areas in µm², present only
                when ``pixel_size_um`` is provided.
    """
    if pixel_size_um is not None:
        min_a_um2 = 20.0 if min_area_um2 is None else float(min_area_um2)
        ps = float(pixel_size_um)
        min_area = max(1, int(round(min_a_um2 / (ps * ps))))
        if max_area_um2 is not None:
            max_area = max(min_area + 1, int(round(float(max_area_um2) / (ps * ps))))

    img = np.asarray(image, dtype=np.float64)

    if isinstance(method, (int, float)):
        thresh = float(method)
    elif method == 'otsu':
        thresh = float(filters.threshold_otsu(img))
    elif method == 'li':
        thresh = float(filters.threshold_li(img))
    elif method == 'triangle':
        thresh = float(filters.threshold_triangle(img))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'otsu', 'li', "
                         "'triangle', or a numeric threshold.")

    binary = img > thresh

    if fill_holes:
        binary = morphology.remove_small_holes(binary, max_size=499)

    binary = morphology.remove_small_objects(binary, max_size=min_area)

    if max_area is not None:
        labeled_tmp = measure.label(binary)
        for p in measure.regionprops(labeled_tmp):
            if p.area > max_area:
                binary[labeled_tmp == p.label] = False

    labeled = measure.label(binary)
    props = measure.regionprops(labeled, intensity_image=image)

    result = {
        'labeled': labeled,
        'props': props,
        'binary': binary,
        'threshold': thresh,
    }
    if pixel_size_um is not None:
        ps = float(pixel_size_um)
        result['areas_um2'] = np.array([p.area * ps * ps for p in props], dtype=float)
    return result


def measure_objects(props, pixel_size=1.0, edge_margin=10,
                    image_shape=None, intensity_image=None):
    """Extract morphometric measurements from regionprops.

    Args:
        props: List of regionprops objects.
        pixel_size: Micrometers per pixel (same as config.pixel_size_um).
        edge_margin: Exclude objects whose centroid is within this many pixels
            of the image edge.
        image_shape: (height, width) for edge filtering. If None, no edge
            filtering is applied.
        intensity_image: Optional 2D array for intensity features. When
            provided, adds mean_intensity, std_intensity, min_intensity,
            max_intensity, and intensity_range per object.

    Returns:
        list of dicts, each containing:
            centroid_px: (row, col) pixel centroid.
            centroid_um: (y, x) centroid in micrometers relative to image origin.
            area_px: Area in pixels.
            area_um2: Area in µm².
            diameter_um: Equivalent circular diameter in µm.
            major_um: Major axis length in µm.
            minor_um: Minor axis length in µm.
            eccentricity: Eccentricity (0 = circle, 1 = line).
            solidity: Area / convex hull area.
            circularity: 4*pi*area/perimeter^2 (1 = perfect circle).
            aspect_ratio: major/minor axis ratio (1 = circular).
            label: Label from the labeled image.
            (if intensity_image provided):
            mean_intensity: float.
            std_intensity: float.
            min_intensity: float.
            max_intensity: float.
            intensity_range: float (max - min).
    """
    results = []
    for p in props:
        cy, cx = p.centroid

        if image_shape is not None:
            h, w = image_shape
            if (cy < edge_margin or cy > h - edge_margin or
                    cx < edge_margin or cx > w - edge_margin):
                continue

        area_px = p.area
        area_um2 = area_px * pixel_size ** 2
        d_um = 2 * np.sqrt(area_px / np.pi) * pixel_size
        major = p.axis_major_length * pixel_size
        minor = p.axis_minor_length * pixel_size

        # Circularity: 4*pi*area / perimeter^2
        perim = p.perimeter if p.perimeter > 0 else 1e-10
        circularity = 4 * np.pi * area_px / (perim ** 2)
        circularity = min(circularity, 1.0)  # Cap at 1.0

        # Aspect ratio
        aspect = major / minor if minor > 0 else 1.0

        obj = {
            'centroid_px': (cy, cx),
            'centroid_um': (cy * pixel_size, cx * pixel_size),
            'area_px': int(area_px),
            'area_um2': float(area_um2),
            'diameter_um': float(d_um),
            'major_um': float(major),
            'minor_um': float(minor),
            'eccentricity': float(p.eccentricity),
            'solidity': float(p.solidity),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect),
            'label': p.label,
        }

        # Intensity features
        if intensity_image is not None:
            coords = p.coords  # (N, 2) array of (row, col)
            intensities = intensity_image[coords[:, 0], coords[:, 1]].astype(float)
            obj['mean_intensity'] = float(np.mean(intensities))
            obj['std_intensity'] = float(np.std(intensities))
            obj['min_intensity'] = float(np.min(intensities))
            obj['max_intensity'] = float(np.max(intensities))
            obj['intensity_range'] = float(np.max(intensities) - np.min(intensities))

        results.append(obj)

    return results


def population_stats(measurements, key='diameter_um'):
    """Compute population statistics for a morphometric measurement.

    Args:
        measurements: List of dicts from measure_objects().
        key: Which measurement to analyze.

    Returns:
        dict with: count, mean, std, median, min, max, percentiles (dict),
            cv (coefficient of variation).
    """
    values = np.array([m[key] for m in measurements])
    n = len(values)
    if n == 0:
        return {'count': 0, 'mean': 0, 'std': 0, 'median': 0,
                'min': 0, 'max': 0, 'cv': 0, 'percentiles': {}}

    pcts = {}
    for p in [10, 25, 50, 75, 90, 95]:
        pcts[f'P{p}'] = float(np.percentile(values, p))

    mean = float(values.mean())
    std = float(values.std())

    return {
        'count': n,
        'mean': mean,
        'std': std,
        'median': float(np.median(values)),
        'min': float(values.min()),
        'max': float(values.max()),
        'cv': std / mean if mean > 0 else 0.0,
        'percentiles': pcts,
    }


def identify_outliers(measurements, key='diameter_um', method='zscore',
                      threshold=2.0):
    """Identify outlier objects in a population.

    Args:
        measurements: List of dicts from measure_objects().
        key: Which measurement to use for outlier detection.
        method: 'zscore' (mean ± threshold*std), 'iqr' (Q75 + threshold*IQR),
            or 'ratio' (> threshold × median).
        threshold: Method-specific threshold parameter.

    Returns:
        dict with:
            outlier_indices: List of indices into measurements.
            outlier_values: List of outlier values.
            threshold_value: The computed threshold.
            method: Method used.
            normal_mean: Mean of non-outlier values.
            normal_std: Std of non-outlier values.
    """
    values = np.array([m[key] for m in measurements])
    n = len(values)

    if n < 3:
        return {
            'outlier_indices': [],
            'outlier_values': [],
            'threshold_value': float('inf'),
            'method': method,
            'normal_mean': float(values.mean()) if n > 0 else 0,
            'normal_std': float(values.std()) if n > 0 else 0,
        }

    if method == 'zscore':
        mean = values.mean()
        std = values.std()
        thresh_val = mean + threshold * std
    elif method == 'iqr':
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        thresh_val = q75 + threshold * iqr
    elif method == 'ratio':
        median = np.median(values)
        thresh_val = threshold * median
    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_idx = [i for i, v in enumerate(values) if v > thresh_val]
    outlier_vals = [float(values[i]) for i in outlier_idx]

    normal_vals = values[values <= thresh_val]

    return {
        'outlier_indices': outlier_idx,
        'outlier_values': outlier_vals,
        'threshold_value': float(thresh_val),
        'method': method,
        'normal_mean': float(normal_vals.mean()) if len(normal_vals) > 0 else 0,
        'normal_std': float(normal_vals.std()) if len(normal_vals) > 0 else 0,
    }


def measure_rois(image, positions, radius=10, names=None):
    """Measure mean intensity in circular ROIs.

    Common pattern for measuring signal at known positions
    (e.g. soma locations, well centers, fiducial marks).

    Args:
        image: 2D grayscale image.
        positions: List of (row, col) pixel coordinates for ROI centers.
        radius: ROI radius in pixels.
        names: Optional list of names for each ROI.

    Returns:
        List of dicts with: name, center, radius, mean, std, max, min,
            n_pixels.
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape[:2]
    yy, xx = np.ogrid[:H, :W]
    results = []
    for i, (row, col) in enumerate(positions):
        dist = np.sqrt((yy - row) ** 2 + (xx - col) ** 2)
        mask = dist <= radius
        pixels = image[mask]
        name = names[i] if names is not None else f"roi_{i}"
        if len(pixels) == 0:
            results.append({
                'name': name, 'center': (row, col), 'radius': radius,
                'mean': 0.0, 'std': 0.0, 'max': 0.0, 'min': 0.0,
                'n_pixels': 0,
            })
        else:
            results.append({
                'name': name,
                'center': (row, col),
                'radius': radius,
                'mean': float(pixels.mean()),
                'std': float(pixels.std()),
                'max': float(pixels.max()),
                'min': float(pixels.min()),
                'n_pixels': int(len(pixels)),
            })
    return results


def to_world_coords(measurements, stage_x, stage_y, pixel_size,
                     image_center=(256, 256)):
    """Add world coordinates to measurement dicts.

    Args:
        measurements: List of dicts from measure_objects().
        stage_x, stage_y: Stage position (world center of FOV).
        pixel_size: µm per pixel.
        image_center: (cy, cx) pixel center of the image.

    Returns:
        Same list with 'world_x' and 'world_y' added to each dict.
    """
    cy_center, cx_center = image_center
    for m in measurements:
        py, px = m['centroid_px']
        m['world_x'] = stage_x + (px - cx_center) * pixel_size
        m['world_y'] = stage_y + (py - cy_center) * pixel_size
    return measurements
