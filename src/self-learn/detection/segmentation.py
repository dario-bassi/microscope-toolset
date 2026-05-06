"""Segmentation strategies for diverse microscopy images.

Provides multiple segmentation approaches beyond simple thresholding,
for use when cells are touching, illumination is uneven, or membrane
channels can guide nuclear segmentation.

Functions:
    watershed_split    -- Split touching cells using watershed
    adaptive_threshold -- Local thresholding for uneven illumination
    segment_by_markers -- Marker-controlled watershed (membrane-guided)
    separate_touching  -- Detect and split touching/overlapping cells
"""

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation


def watershed_split(binary_mask, min_distance=7):
    """Split touching cells in a binary mask using watershed.

    Args:
        binary_mask: 2D boolean array where True = cell pixels.
        min_distance: Minimum distance between cell centers for
            peak detection. Larger values merge nearby peaks.

    Returns:
        dict with:
            labeled: 2D labeled image (0 = background, 1..N = cells).
            n_cells: Number of separated cells.
            centroids: List of (y, x) centroid tuples.
    """
    binary = np.asarray(binary_mask, dtype=bool)

    if not binary.any():
        return {
            "labeled": np.zeros_like(binary, dtype=int),
            "n_cells": 0,
            "centroids": [],
        }

    # Distance transform — peaks at cell centers
    dist = ndimage.distance_transform_edt(binary)

    # Find local maxima as markers
    from skimage.feature import peak_local_max

    coords = peak_local_max(dist, min_distance=min_distance, labels=binary)

    if len(coords) == 0:
        # No peaks found — treat as single object
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        centroids = [(p.centroid[0], p.centroid[1]) for p in props]
        return {
            "labeled": labeled,
            "n_cells": len(props),
            "centroids": centroids,
        }

    # Create markers from peaks
    markers = np.zeros_like(binary, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    # Watershed
    labeled = segmentation.watershed(-dist, markers, mask=binary)

    props = measure.regionprops(labeled)
    centroids = [(p.centroid[0], p.centroid[1]) for p in props]

    return {
        "labeled": labeled,
        "n_cells": len(props),
        "centroids": centroids,
    }


def adaptive_threshold(image, block_size=51, offset=0, method="gaussian", min_area=20):
    """Local adaptive thresholding for uneven illumination.

    Args:
        image: 2D grayscale image.
        block_size: Size of local neighborhood (must be odd).
        offset: Constant subtracted from local threshold.
        method: 'gaussian' or 'mean' for local threshold computation.
        min_area: Remove objects smaller than this.

    Returns:
        dict with:
            binary: 2D boolean mask.
            labeled: 2D labeled image.
            n_objects: Number of detected objects.
    """
    img = np.asarray(image, dtype=np.float64)

    if method == "gaussian":
        thresh = filters.threshold_local(img, block_size, method="gaussian", offset=offset)
    elif method == "mean":
        thresh = filters.threshold_local(img, block_size, method="mean", offset=offset)
    else:
        raise ValueError(f"method must be 'gaussian' or 'mean', got '{method}'")

    binary = img > thresh

    if min_area > 0:
        binary = morphology.remove_small_objects(binary, max_size=min_area)

    labeled = measure.label(binary)
    n_objects = labeled.max()

    return {
        "binary": binary,
        "labeled": labeled,
        "n_objects": n_objects,
    }


def segment_by_markers(intensity_image, marker_image, threshold=None, min_area=20):
    """Marker-controlled watershed using a second channel.

    Use membrane/boundary channel to define cell boundaries, and
    nuclear/cell body channel for seeds.

    Args:
        intensity_image: 2D image of cell boundaries (e.g., membrane
            channel). Higher values = stronger boundaries.
        marker_image: 2D image for seed detection (e.g., nuclear channel).
            Bright regions become cell markers.
        threshold: Threshold for marker detection. If None, uses Otsu.
        min_area: Minimum marker area in pixels.

    Returns:
        dict with:
            labeled: 2D labeled image.
            n_cells: Number of segmented cells.
            centroids: List of (y, x) tuples.
    """
    intensity = np.asarray(intensity_image, dtype=np.float64)
    markers_img = np.asarray(marker_image, dtype=np.float64)

    # Detect markers from nuclear channel
    if threshold is None:
        threshold = float(filters.threshold_otsu(markers_img))

    marker_binary = markers_img > threshold
    if min_area > 0:
        marker_binary = morphology.remove_small_objects(marker_binary, max_size=min_area)

    markers = measure.label(marker_binary)

    if markers.max() == 0:
        return {
            "labeled": np.zeros_like(intensity, dtype=int),
            "n_cells": 0,
            "centroids": [],
        }

    # Use intensity image gradient as watershed landscape
    gradient = filters.sobel(intensity)
    labeled = segmentation.watershed(gradient, markers)

    props = measure.regionprops(labeled)
    centroids = [(p.centroid[0], p.centroid[1]) for p in props]

    return {
        "labeled": labeled,
        "n_cells": len(props),
        "centroids": centroids,
    }


def separate_touching(binary_mask, erosion_radius=2, min_area=20):
    """Separate touching cells using erosion → labeling → dilation.

    Simpler than watershed, works well when cells only slightly overlap.

    Args:
        binary_mask: 2D boolean array.
        erosion_radius: Radius of erosion disk. Larger values split
            more aggressively.
        min_area: Minimum area for eroded regions (removes noise).

    Returns:
        dict with:
            labeled: 2D labeled image.
            n_cells: Number of separated cells.
            centroids: List of (y, x) tuples.
    """
    binary = np.asarray(binary_mask, dtype=bool)

    if not binary.any():
        return {
            "labeled": np.zeros_like(binary, dtype=int),
            "n_cells": 0,
            "centroids": [],
        }

    # Erode to separate touching cells
    selem = morphology.disk(erosion_radius)
    eroded = ndimage.binary_erosion(binary, structure=selem)

    # Remove small fragments
    if min_area > 0:
        eroded = morphology.remove_small_objects(eroded, max_size=min_area)

    # Label eroded regions
    markers = measure.label(eroded)

    if markers.max() == 0:
        # No regions survived erosion — fall back to original labeling
        labeled = measure.label(binary)
    else:
        # Watershed expand markers back to original boundary
        dist = ndimage.distance_transform_edt(binary)
        labeled = segmentation.watershed(-dist, markers, mask=binary)

    props = measure.regionprops(labeled)
    centroids = [(p.centroid[0], p.centroid[1]) for p in props]

    return {
        "labeled": labeled,
        "n_cells": len(props),
        "centroids": centroids,
    }
