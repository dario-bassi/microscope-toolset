"""Apoptosis and cell death detection from microscopy images.

Detects and classifies cell death morphology: membrane blebbing,
nuclear fragmentation, cell shrinkage, and chromatin condensation.
Goes beyond simple live/dead binary classification.

Functions:
    detect_apoptotic      -- Classify cells as healthy/early_apoptotic/late_apoptotic/necrotic
    apoptotic_index       -- Fraction of apoptotic cells in population
    morphology_score      -- Score individual cell health from shape features
    temporal_death_progression -- Track death over time in a population
"""

import numpy as np
from scipy import ndimage


def detect_apoptotic(areas, circularities, intensities=None,
                     eccentricities=None, area_threshold=None):
    """Classify cells by apoptotic morphology.

    Uses nuclear/cell morphology features to classify each cell:
    - healthy: normal size, round, uniform
    - early_apoptotic: shrinking, condensing chromatin (bright, small)
    - late_apoptotic: fragmented, very small pieces
    - necrotic: swollen (large), irregular shape

    Args:
        areas: 1D array, cell/nuclear area in pixels.
        circularities: 1D array, circularity (4π*A/P²), 0-1.
        intensities: optional 1D array, mean intensity per cell.
        eccentricities: optional 1D array, shape eccentricity (0=circle, 1=line).
        area_threshold: float or None. Median area used if None.

    Returns:
        dict with:
            classifications: list of str per cell.
            counts: dict mapping class -> count.
            apoptotic_fraction: float, (early + late) / total.
    """
    areas = np.asarray(areas, dtype=float)
    circularities = np.asarray(circularities, dtype=float)
    n = len(areas)

    if n == 0:
        return {
            'classifications': [],
            'counts': {'healthy': 0, 'early_apoptotic': 0,
                       'late_apoptotic': 0, 'necrotic': 0},
            'apoptotic_fraction': 0.0,
        }

    if intensities is not None:
        intensities = np.asarray(intensities, dtype=float)
    if eccentricities is not None:
        eccentricities = np.asarray(eccentricities, dtype=float)

    median_area = area_threshold if area_threshold else float(np.median(areas))
    median_circ = float(np.median(circularities))

    classifications = []
    for i in range(n):
        area = areas[i]
        circ = circularities[i]

        # Late apoptotic: very small fragments
        if area < median_area * 0.3:
            classifications.append('late_apoptotic')
        # Necrotic: swollen cells, low circularity
        elif area > median_area * 2.0 and circ < median_circ * 0.7:
            classifications.append('necrotic')
        # Early apoptotic: small, condensed (bright if intensity available)
        elif area < median_area * 0.6:
            if intensities is not None and intensities[i] > np.median(intensities) * 1.3:
                classifications.append('early_apoptotic')
            elif circ > 0.8:  # very round = condensed
                classifications.append('early_apoptotic')
            else:
                classifications.append('healthy')
        # Necrotic: irregular shape
        elif circ < 0.4:
            classifications.append('necrotic')
        else:
            classifications.append('healthy')

    counts = {c: classifications.count(c) for c in
              ['healthy', 'early_apoptotic', 'late_apoptotic', 'necrotic']}
    apop = counts['early_apoptotic'] + counts['late_apoptotic']

    return {
        'classifications': classifications,
        'counts': counts,
        'apoptotic_fraction': round(apop / n, 4) if n > 0 else 0.0,
    }


def apoptotic_index(classifications=None, n_apoptotic=None, n_total=None):
    """Compute apoptotic index.

    Args:
        classifications: list of str from detect_apoptotic().
        n_apoptotic: int, direct count (alternative).
        n_total: int, total count (required with n_apoptotic).

    Returns:
        dict with:
            apoptotic_index: float.
            n_apoptotic: int.
            n_total: int.
            percentage: float.
    """
    if classifications is not None:
        n_a = sum(1 for c in classifications
                  if c in ('early_apoptotic', 'late_apoptotic'))
        n_t = len(classifications)
    elif n_apoptotic is not None and n_total is not None:
        n_a = n_apoptotic
        n_t = n_total
    else:
        raise ValueError("Provide classifications or both n_apoptotic and n_total")

    ai = n_a / n_t if n_t > 0 else 0.0

    return {
        'apoptotic_index': round(ai, 4),
        'n_apoptotic': n_a,
        'n_total': n_t,
        'percentage': round(ai * 100, 2),
    }


def morphology_score(area, circularity, intensity=None,
                     ref_area=None, ref_circularity=0.85):
    """Score individual cell health from morphology (0=dead, 1=healthy).

    Args:
        area: float, cell area.
        circularity: float, cell circularity.
        intensity: float or None, cell intensity.
        ref_area: float, reference healthy cell area. Required.
        ref_circularity: float, reference healthy circularity.

    Returns:
        float, health score 0-1.
    """
    if ref_area is None or ref_area <= 0:
        return 0.5  # unknown reference

    # Area score: penalize very small (apoptotic) or very large (necrotic)
    area_ratio = area / ref_area
    if area_ratio < 0.3:
        area_score = 0.0  # fragmented
    elif area_ratio < 0.6:
        area_score = (area_ratio - 0.3) / 0.3  # shrinking
    elif area_ratio < 1.5:
        area_score = 1.0  # normal
    elif area_ratio < 3.0:
        area_score = 1.0 - (area_ratio - 1.5) / 1.5  # swelling
    else:
        area_score = 0.0  # severely swollen

    # Circularity score
    circ_score = min(circularity / ref_circularity, 1.0)

    # Combine
    score = 0.6 * area_score + 0.4 * circ_score
    return round(max(min(score, 1.0), 0.0), 4)


def temporal_death_progression(timepoints, alive_counts, dead_counts=None,
                               total_counts=None):
    """Track cell death progression over time.

    Args:
        timepoints: 1D array of time values.
        alive_counts: 1D array, live cells at each timepoint.
        dead_counts: optional 1D array, dead cells at each timepoint.
        total_counts: optional 1D array, total at each timepoint.

    Returns:
        dict with:
            timepoints: array.
            viability: array, fraction alive at each timepoint.
            death_rate: float, rate of death (change per time unit).
            half_death_time: float, time for 50% death (if applicable).
            pattern: str, 'stable', 'gradual_decline', 'rapid_decline', 'recovery'.
    """
    timepoints = np.asarray(timepoints, dtype=float)
    alive = np.asarray(alive_counts, dtype=float)
    n = len(timepoints)

    if dead_counts is not None:
        dead = np.asarray(dead_counts, dtype=float)
        total = alive + dead
    elif total_counts is not None:
        total = np.asarray(total_counts, dtype=float)
    else:
        total = alive.copy()  # assume alive = total

    viability = np.where(total > 0, alive / total, 1.0)

    # Death rate: slope of viability vs time
    if n >= 2 and (timepoints[-1] - timepoints[0]) > 0:
        slope = np.polyfit(timepoints, viability, 1)[0]
        death_rate = -slope  # positive = dying
    else:
        death_rate = 0.0
        slope = 0.0

    # Half-death time: when viability crosses 0.5
    half_death_time = 0.0
    if viability[0] > 0.5:
        crossings = np.where(viability < 0.5)[0]
        if len(crossings) > 0:
            idx = crossings[0]
            if idx > 0:
                # Linear interpolation
                v0, v1 = viability[idx - 1], viability[idx]
                t0, t1 = timepoints[idx - 1], timepoints[idx]
                if v0 != v1:
                    half_death_time = t0 + (0.5 - v0) / (v1 - v0) * (t1 - t0)
                else:
                    half_death_time = t0

    # Classify pattern
    if n < 3:
        pattern = 'unknown'
    elif abs(slope) < 0.01:
        pattern = 'stable'
    elif slope < -0.1:
        pattern = 'rapid_decline'
    elif slope < 0:
        pattern = 'gradual_decline'
    elif slope > 0.01:
        pattern = 'recovery'
    else:
        pattern = 'stable'

    return {
        'timepoints': timepoints,
        'viability': viability,
        'death_rate': round(float(death_rate), 6),
        'half_death_time': round(float(half_death_time), 4),
        'pattern': pattern,
    }
