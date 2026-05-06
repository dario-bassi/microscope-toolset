"""Z-stack analysis for 3D microscopy data.

Functions for projecting, slicing, and measuring volumetric image stacks.
A Z-stack is a list/array of 2D images at different focal planes.

Functions:
    project_mip        -- Maximum intensity projection
    project_mean       -- Mean intensity projection
    project_std        -- Standard deviation projection (focus map)
    slice_orthogonal   -- Extract XZ or YZ cross-section
    measure_volume     -- Volumetric measurement from thresholded stack
    find_focus_plane   -- Find best-focused plane (Laplacian variance)
    z_profile          -- Intensity profile along Z at a point/ROI
"""

import numpy as np
from skimage import filters


def project_mip(stack):
    """Maximum intensity projection along Z axis.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.

    Returns:
        2D array — max value at each (y, x) across all Z planes.
    """
    stack = np.asarray(stack)
    return stack.max(axis=0)


def project_mean(stack):
    """Mean intensity projection along Z axis.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.

    Returns:
        2D array — mean value at each (y, x) across all Z planes.
    """
    stack = np.asarray(stack, dtype=np.float64)
    return stack.mean(axis=0)


def project_std(stack):
    """Standard deviation projection along Z axis.

    High values indicate pixels with strong Z-variation (structures
    that come in and out of focus). Useful as a focus/content map.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.

    Returns:
        2D array — standard deviation at each (y, x) across all Z planes.
    """
    stack = np.asarray(stack, dtype=np.float64)
    return stack.std(axis=0)


def slice_orthogonal(stack, axis="xz", position=None, pixel_size_xy=1.0, pixel_size_z=1.0):
    """Extract an orthogonal cross-section from a Z-stack.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.
        axis: 'xz' for XZ cross-section (fixed Y), 'yz' for YZ (fixed X).
        position: Row/column index for the slice. If None, uses center.
        pixel_size_xy: Lateral pixel size in µm.
        pixel_size_z: Axial step size in µm.

    Returns:
        dict with:
            image: 2D array of the cross-section.
            extent: (left, right, bottom, top) in µm for plotting.
            axis: Which axis was sliced.
    """
    stack = np.asarray(stack)
    nz, ny, nx = stack.shape

    if axis == "xz":
        if position is None:
            position = ny // 2
        section = stack[:, position, :]  # shape (Z, X)
        extent = (0, nx * pixel_size_xy, nz * pixel_size_z, 0)
    elif axis == "yz":
        if position is None:
            position = nx // 2
        section = stack[:, :, position]  # shape (Z, Y)
        extent = (0, ny * pixel_size_xy, nz * pixel_size_z, 0)
    else:
        raise ValueError(f"axis must be 'xz' or 'yz', got '{axis}'")

    return {
        "image": section,
        "extent": extent,
        "axis": axis,
    }


def measure_volume(stack, threshold=None, pixel_size_xy=1.0, pixel_size_z=1.0):
    """Measure volume of structures in a Z-stack by thresholding.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.
        threshold: Intensity threshold. If None, uses Otsu on the
            max-projection.
        pixel_size_xy: Lateral pixel size in µm.
        pixel_size_z: Axial step size in µm.

    Returns:
        dict with:
            volume_um3: Total volume above threshold in µm³.
            volume_voxels: Number of voxels above threshold.
            voxel_size_um3: Size of one voxel in µm³.
            threshold: Threshold value used.
            mask: 3D boolean mask.
    """
    stack = np.asarray(stack, dtype=np.float64)

    if threshold is None:
        mip = stack.max(axis=0)
        threshold = float(filters.threshold_otsu(mip))

    mask = stack > threshold
    n_voxels = int(mask.sum())
    voxel_size = pixel_size_xy**2 * pixel_size_z
    volume = n_voxels * voxel_size

    return {
        "volume_um3": float(volume),
        "volume_voxels": n_voxels,
        "voxel_size_um3": float(voxel_size),
        "threshold": float(threshold),
        "mask": mask,
    }


def find_focus_plane(stack, metric="laplacian"):
    """Find the best-focused plane in a Z-stack.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.
        metric: Focus metric — 'laplacian' (variance of Laplacian),
            'gradient' (mean gradient magnitude), or 'std' (pixel
            standard deviation).

    Returns:
        dict with:
            best_z: Index of the best-focused plane.
            scores: 1D array of focus scores per plane.
            best_score: Focus score of the best plane.
    """
    stack = np.asarray(stack, dtype=np.float64)
    nz = stack.shape[0]

    scores = np.zeros(nz)
    for z in range(nz):
        plane = stack[z]
        if metric == "laplacian":
            lap = filters.laplace(plane)
            scores[z] = lap.var()
        elif metric == "gradient":
            gy, gx = np.gradient(plane)
            scores[z] = np.sqrt(gx**2 + gy**2).mean()
        elif metric == "std":
            scores[z] = plane.std()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    best_z = int(scores.argmax())

    return {
        "best_z": best_z,
        "scores": scores,
        "best_score": float(scores[best_z]),
    }


def z_profile(stack, position=None, roi_mask=None):
    """Extract intensity profile along Z at a point or ROI.

    Args:
        stack: 3D array (Z, Y, X) or list of 2D arrays.
        position: (y, x) pixel coordinates. Used if roi_mask is None.
            If both None, uses image center.
        roi_mask: 2D boolean mask. Mean intensity within mask at each Z.

    Returns:
        dict with:
            z_indices: 1D array of Z indices.
            intensities: 1D array of intensity values.
            peak_z: Z index of maximum intensity.
            fwhm_z: Full width at half maximum in Z slices (or None if
                not computable).
    """
    stack = np.asarray(stack, dtype=np.float64)
    nz = stack.shape[0]

    intensities = np.zeros(nz)
    for z in range(nz):
        if roi_mask is not None:
            intensities[z] = stack[z][roi_mask].mean()
        elif position is not None:
            y, x = position
            intensities[z] = stack[z, y, x]
        else:
            cy, cx = stack.shape[1] // 2, stack.shape[2] // 2
            intensities[z] = stack[z, cy, cx]

    peak_z = int(intensities.argmax())

    # FWHM
    half_max = (intensities.max() + intensities.min()) / 2
    above = intensities >= half_max
    if above.any():
        indices = np.where(above)[0]
        fwhm = float(indices[-1] - indices[0])
    else:
        fwhm = None

    return {
        "z_indices": np.arange(nz),
        "intensities": intensities,
        "peak_z": peak_z,
        "fwhm_z": fwhm,
    }
