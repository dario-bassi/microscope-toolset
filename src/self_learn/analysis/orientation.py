"""Orientation and circular statistics for axial data (angles mod 180°).

Extracted from intensity.py for modularity. All functions remain importable
from intensity.py and analysis/__init__.py for backward compatibility.
"""

import numpy as np


def circular_mean(angles_deg, axial=True):
    """Compute circular mean of angles in degrees.

    For axial data (orientations without head/tail distinction, e.g. cell
    elongation), doubles the angles before averaging. This correctly handles
    the 0°/180° wraparound.

    Args:
        angles_deg: Array of angles in degrees.
        axial: If True, treats data as axial (mod 180°). If False, directional (mod 360°).

    Returns:
        Mean angle in degrees, in [0, 180) for axial or [0, 360) for directional.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    factor = 2.0 if axial else 1.0
    rad = np.radians(angles * factor)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    mean_rad = np.arctan2(mean_sin, mean_cos)
    mean_deg = np.degrees(mean_rad) / factor
    modulo = 180.0 if axial else 360.0
    result = mean_deg % modulo
    # Guard against floating-point edge: -1e-15 % 180 == 180.0
    if result >= modulo:
        result = 0.0
    return float(result)


def circular_std(angles_deg, axial=True):
    """Compute circular standard deviation of angles in degrees.

    Args:
        angles_deg: Array of angles in degrees.
        axial: If True, axial data (mod 180°).

    Returns:
        Circular standard deviation in degrees.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    factor = 2.0 if axial else 1.0
    rad = np.radians(angles * factor)
    R = np.sqrt(np.mean(np.cos(rad))**2 + np.mean(np.sin(rad))**2)
    R = min(R, 1.0)
    # Circular variance = 1 - R; circular std = sqrt(-2 * ln(R))
    if R > 1e-10:
        csd_rad = np.sqrt(-2 * np.log(R))
    else:
        csd_rad = np.pi / factor  # maximum dispersion
    return float(np.degrees(csd_rad) / factor)


def nematic_order_parameter(angles_deg, reference_deg=90.0):
    """Compute nematic order parameter S for orientation alignment.

    S = <cos(2 * (θ - reference))>

    S = 1: all aligned at reference angle
    S = -1: all perpendicular to reference
    S = 0: random/isotropic

    Common use: reference=90° for perpendicular alignment to horizontal stretch.

    Args:
        angles_deg: Array of orientation angles in degrees.
        reference_deg: Reference direction in degrees.

    Returns:
        Float in [-1, 1].
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    return float(np.mean(np.cos(2 * np.radians(angles - reference_deg))))
