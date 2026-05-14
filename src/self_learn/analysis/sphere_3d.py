"""Sphere 3D estimation from 2D equatorial observations.

For imaging a sphere (e.g., Volvox colony) at its equatorial plane,
only a thin ring of the surface is in focus. This module estimates
total surface features from equatorial observations using DOF-based
geometry.
"""

import numpy as np


def dof_visible_fraction(radius_px, dof_px=6.0):
    """Fraction of sphere surface visible at equatorial focal plane.

    At the equatorial plane, the depth of field (DOF) captures a ring
    of width ~DOF around the equator. The visible fraction is:
        fraction = DOF / R  (front and back rings combined)

    This is much less than 0.5 (hemisphere) because only the equatorial
    ring is in focus, not the entire front hemisphere.

    Parameters
    ----------
    radius_px : float
        Sphere radius in pixels.
    dof_px : float
        Depth of field in pixels (at 10x ≈ 6px, at 40x ≈ 1.5px).

    Returns
    -------
    float
        Fraction of sphere surface visible (typically 0.02–0.15).
    """
    if radius_px <= 0:
        return 1.0
    # DOF captures ±dof/2 from equator, front + back = 2 rings
    # Each ring area ≈ 2πR × dof/2, total sphere = 4πR²
    # fraction = 2 × (2πR × dof/2) / (4πR²) = dof / (2R) × 2 = dof/R
    # But front and back overlap for thin DOF, so:
    fraction = min(dof_px / radius_px, 1.0)
    return fraction


def extrapolate_sphere_surface(visible_count, radius_px, dof_px=6.0):
    """Estimate total surface features from equatorial visible count.

    Parameters
    ----------
    visible_count : int
        Number of features visible at equatorial focal plane.
    radius_px : float
        Sphere radius in pixels.
    dof_px : float
        Depth of field in pixels.

    Returns
    -------
    int
        Estimated total features on sphere surface.
    """
    fraction = dof_visible_fraction(radius_px, dof_px)
    if fraction <= 0:
        return visible_count
    return round(visible_count / fraction)


def dof_for_objective(magnification, na=None):
    """Estimate depth of field in microns for given objective.

    Uses the approximation: DOF ≈ λ/(2·NA²) + n·e/(NA·M)
    where λ=0.55μm (green), n=1 (air), e=sensor_pixel.

    For quick estimates:
        10x (NA 0.25): ~6 μm
        20x (NA 0.45): ~2 μm
        40x (NA 0.65): ~1.3 μm
        100x (NA 1.25): ~0.5 μm

    Parameters
    ----------
    magnification : int
        Objective magnification (10, 20, 40, 100).
    na : float, optional
        Numerical aperture. If None, uses typical values.

    Returns
    -------
    float
        Depth of field in microns.
    """
    if na is None:
        na_map = {4: 0.10, 10: 0.25, 20: 0.45, 40: 0.65, 60: 0.85, 100: 1.25}
        na = na_map.get(magnification, 0.25)

    wavelength = 0.55  # green light, microns
    pixel_size = 6.5  # typical sensor pixel, microns
    dof_wave = wavelength / (2 * na**2)
    dof_geom = pixel_size / (na * magnification)
    return dof_wave + dof_geom


def dof_in_pixels(magnification, pixel_size_um=None, na=None):
    """Depth of field converted to image pixels.

    Parameters
    ----------
    magnification : int
        Objective magnification.
    pixel_size_um : float, optional
        Pixel size in microns. If None, uses 10/magnification.
    na : float, optional
        Numerical aperture.

    Returns
    -------
    float
        Depth of field in pixels.
    """
    if pixel_size_um is None:
        pixel_size_um = 10.0 / magnification
    dof_um = dof_for_objective(magnification, na)
    return dof_um / pixel_size_um


def sphere_surface_area(radius_px):
    """Total surface area of a sphere in pixels².

    Parameters
    ----------
    radius_px : float
        Sphere radius in pixels.

    Returns
    -------
    float
        Surface area in pixels².
    """
    return 4 * np.pi * radius_px**2


def estimate_cell_density(visible_count, radius_px, dof_px=6.0):
    """Estimate cell density on sphere surface (cells per pixel²).

    Parameters
    ----------
    visible_count : int
        Visible cells at equatorial plane.
    radius_px : float
        Sphere radius in pixels.
    dof_px : float
        Depth of field in pixels.

    Returns
    -------
    float
        Estimated cell density (cells/px²).
    """
    total = extrapolate_sphere_surface(visible_count, radius_px, dof_px)
    area = sphere_surface_area(radius_px)
    if area <= 0:
        return 0.0
    return total / area


def equatorial_ring_area(radius_px, dof_px=6.0):
    """Area of the equatorial ring visible in one focal plane.

    The ring spans ±dof/2 from the equator on both front and back,
    projected onto the image plane.

    Parameters
    ----------
    radius_px : float
        Sphere radius in pixels.
    dof_px : float
        Depth of field in pixels.

    Returns
    -------
    float
        Visible ring area in pixels² (projected).
    """
    half_dof = dof_px / 2.0
    if half_dof >= radius_px:
        # Entire sphere visible
        return np.pi * radius_px**2

    # Front ring: annulus from R*cos(asin(h/R)) to R
    # where h = half_dof
    # inner_r = R * cos(asin(h/R)) = sqrt(R² - h²)
    inner_r = np.sqrt(max(radius_px**2 - half_dof**2, 0))
    ring_area = np.pi * (radius_px**2 - inner_r**2)
    # Front + back (roughly 2×, minus small overlap)
    return 2 * ring_area


def interior_visible_fraction(radius_px, dof_px=6.0):
    """Fraction of interior objects visible at equatorial plane.

    For objects distributed inside the sphere (like gonidia), the
    visible fraction depends on DOF differently than surface cells.
    Objects in the front half within ±dof/2 of the focal plane are visible.

    Parameters
    ----------
    radius_px : float
        Sphere radius in pixels.
    dof_px : float
        Depth of field in pixels.

    Returns
    -------
    float
        Fraction of interior volume objects visible.
    """
    if radius_px <= 0:
        return 1.0
    # For interior objects: visible ≈ dof / (2*R) for the front hemisphere
    # Objects on the far side are occluded by the sphere
    # So: fraction ≈ dof / (2*R) (clipped to max 0.5 for front half only)
    fraction = min(dof_px / (2 * radius_px), 0.5)
    return fraction


def extrapolate_interior(visible_count, radius_px, dof_px=6.0):
    """Estimate total interior objects from equatorial visible count.

    Parameters
    ----------
    visible_count : int
        Visible interior objects (e.g., gonidia).
    radius_px : float
        Sphere radius in pixels.
    dof_px : float
        Depth of field in pixels.

    Returns
    -------
    int
        Estimated total interior objects.
    """
    fraction = interior_visible_fraction(radius_px, dof_px)
    if fraction <= 0:
        return visible_count
    return round(visible_count / fraction)
