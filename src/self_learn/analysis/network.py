"""Network and branching structure analysis.

Analyzes tubular/vascular/dendritic network morphology from
binary masks. Used for angiogenesis assays, neuronal dendrites,
fungal hyphae, mitochondrial networks, and other branching structures.

Functions:
    skeletonize_network  -- Thin network mask to 1-pixel skeleton
    detect_junctions     -- Find branch points in skeleton
    detect_endpoints     -- Find terminal points
    measure_network      -- Comprehensive network morphometry
    segment_branches     -- Split skeleton into individual branches
    count_fragments      -- Count disconnected pieces (connected components)
"""

import numpy as np
from scipy import ndimage
from skimage import morphology


def skeletonize_network(mask, min_branch_length=0):
    """Skeletonize a binary mask of a tubular network.

    Args:
        mask: 2D binary array of the network.
        min_branch_length: int, remove branches shorter than this.
            Useful for removing noise spurs.

    Returns:
        dict with:
            skeleton: 2D binary array, thinned to 1 pixel width.
            total_length: float, total skeleton length in pixels.
            n_pixels: int, total number of skeleton pixels.
    """
    mask = np.asarray(mask, dtype=bool)
    skel = morphology.skeletonize(mask)

    if min_branch_length > 0:
        skel = _prune_short_branches(skel, min_branch_length)

    total_length = _skeleton_length(skel)

    return {
        'skeleton': skel,
        'total_length': round(total_length, 2),
        'n_pixels': int(skel.sum()),
    }


def detect_junctions(skeleton):
    """Find branch points (junctions) in a skeleton.

    A junction pixel has 3 or more skeleton neighbors in the
    8-connected sense.

    Args:
        skeleton: 2D binary array (skeletonized).

    Returns:
        dict with:
            positions: list of (row, col) junction coordinates.
            n_junctions: int.
            junction_mask: 2D binary array marking junctions.
    """
    skel = np.asarray(skeleton, dtype=bool)
    # Count neighbors for each skeleton pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    neighbor_count = ndimage.convolve(skel.astype(int), kernel,
                                      mode='constant', cval=0)
    junction_mask = skel & (neighbor_count >= 3)

    # Cluster nearby junction pixels (they often come in groups of 2-3)
    labeled, n = ndimage.label(junction_mask)
    positions = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        positions.append((int(np.mean(ys)), int(np.mean(xs))))

    return {
        'positions': positions,
        'n_junctions': len(positions),
        'junction_mask': junction_mask,
    }


def detect_endpoints(skeleton):
    """Find terminal endpoints in a skeleton.

    An endpoint has exactly 1 skeleton neighbor.

    Args:
        skeleton: 2D binary array.

    Returns:
        dict with:
            positions: list of (row, col) endpoint coordinates.
            n_endpoints: int.
            endpoint_mask: 2D binary array.
    """
    skel = np.asarray(skeleton, dtype=bool)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    neighbor_count = ndimage.convolve(skel.astype(int), kernel,
                                      mode='constant', cval=0)
    endpoint_mask = skel & (neighbor_count == 1)

    positions = list(zip(*np.where(endpoint_mask)))

    return {
        'positions': positions,
        'n_endpoints': len(positions),
        'endpoint_mask': endpoint_mask,
    }


def measure_network(mask, pixel_size=1.0, min_branch_length=5):
    """Comprehensive network morphometry.

    Computes all standard tube formation / angiogenesis metrics:
    total length, branch count, junction count, mesh count, etc.

    Args:
        mask: 2D binary array of the network.
        pixel_size: float, physical size per pixel (µm/px).
        min_branch_length: int, prune branches shorter than this.

    Returns:
        dict with:
            total_length: float, total network length (in physical units).
            n_branches: int, number of distinct branch segments.
            n_junctions: int, number of branch points.
            n_endpoints: int, number of terminal tips.
            mean_branch_length: float.
            n_meshes: int, number of enclosed loops/meshes.
            network_area: float, area covered by the network mask.
            lacunarity: float, ratio of empty to total area in bounding box.
            junction_positions: list of (row, col).
            endpoint_positions: list of (row, col).
    """
    mask = np.asarray(mask, dtype=bool)

    # Skeletonize
    skel_info = skeletonize_network(mask, min_branch_length)
    skeleton = skel_info['skeleton']

    # Junctions and endpoints
    junc = detect_junctions(skeleton)
    endp = detect_endpoints(skeleton)

    # Segment branches
    branches = segment_branches(skeleton)

    # Branch lengths
    branch_lengths = []
    for b in branches['branches']:
        branch_lengths.append(float(b['length']) * pixel_size)

    mean_bl = float(np.mean(branch_lengths)) if branch_lengths else 0.0

    # Mesh count: Euler number approach
    # For a planar graph: V - E + F = 2 (Euler formula)
    # meshes = E - V + 1 for connected graph
    n_v = junc['n_junctions'] + endp['n_endpoints']
    n_e = branches['n_branches']
    n_meshes = max(n_e - n_v + 1, 0)

    # Network area and lacunarity
    network_area = float(mask.sum()) * pixel_size**2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        bbox_area = float((r_max - r_min + 1) * (c_max - c_min + 1)) * pixel_size**2
        lacunarity = 1.0 - (network_area / max(bbox_area, 1e-10))
    else:
        lacunarity = 1.0

    return {
        'total_length': round(skel_info['total_length'] * pixel_size, 2),
        'n_branches': branches['n_branches'],
        'n_junctions': junc['n_junctions'],
        'n_endpoints': endp['n_endpoints'],
        'mean_branch_length': round(mean_bl, 2),
        'n_meshes': n_meshes,
        'network_area': round(network_area, 2),
        'lacunarity': round(lacunarity, 4),
        'junction_positions': junc['positions'],
        'endpoint_positions': endp['positions'],
    }


def segment_branches(skeleton):
    """Split a skeleton into individual branch segments.

    Removes junction pixels, then labels connected components.
    Each component is a single branch segment between junctions
    or between a junction and an endpoint.

    Args:
        skeleton: 2D binary array.

    Returns:
        dict with:
            n_branches: int.
            branches: list of dicts, each with:
                label: int.
                length: float, branch length in pixels.
                pixels: int, number of pixels.
                endpoints: list of (row, col) at branch ends.
    """
    skel = np.asarray(skeleton, dtype=bool)

    # Find junction pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    neighbor_count = ndimage.convolve(skel.astype(int), kernel,
                                      mode='constant', cval=0)
    junction_mask = skel & (neighbor_count >= 3)

    # Dilate junctions slightly to split cleanly
    dilated_junc = ndimage.binary_dilation(junction_mask,
                                           structure=np.ones((3, 3)))

    # Remove junctions from skeleton
    branches_mask = skel & ~dilated_junc
    labeled, n_branches = ndimage.label(branches_mask)

    branches = []
    for i in range(1, n_branches + 1):
        branch_pixels = labeled == i
        n_px = int(branch_pixels.sum())
        length = _skeleton_length(branch_pixels)

        # Find endpoints of this branch
        nc = ndimage.convolve(branch_pixels.astype(int), kernel,
                              mode='constant', cval=0)
        ep_mask = branch_pixels & (nc == 1)
        endpoints = list(zip(*np.where(ep_mask)))

        branches.append({
            'label': i,
            'length': round(length, 2),
            'pixels': n_px,
            'endpoints': endpoints,
        })

    return {
        'n_branches': n_branches,
        'branches': branches,
    }


def count_fragments(image, threshold=None, min_size=5, pixel_size=1.0):
    """Count disconnected fragments in a network image via connected components.

    Uses thresholding + connected component labeling to count the number
    of separate pieces in a network. Unlike foci detection (which counts
    bright spots), this counts disconnected regions — suitable for
    measuring mitochondrial fragmentation, vascular disconnections, etc.

    A healthy network typically has few fragments (1 large connected
    component + a few small ones). After perturbation (e.g., CCCP for
    mitochondria), the network breaks into many small puncta.

    Args:
        image: 2D array (grayscale). Will be thresholded.
        threshold: float or None. Intensity threshold for binarization.
            If None, uses Otsu's method on non-zero pixels.
        min_size: int, minimum object size in pixels. Removes noise.
        pixel_size: float, µm/px for area calculation.

    Returns:
        dict with:
            n_fragments: int, number of connected components.
            labeled: 2D int array, labeled connected components.
            mask: 2D bool array, thresholded binary mask.
            total_area_um2: float, total network area in µm².
            fragmentation_index: float, n_fragments / total_area_um2.
                Higher = more fragmented.
    """
    from skimage import filters as skfilters

    image = np.asarray(image, dtype=float)

    # Determine threshold
    if threshold is None:
        nonzero = image[image > 0]
        if len(nonzero) == 0:
            return {
                'n_fragments': 0,
                'labeled': np.zeros_like(image, dtype=int),
                'mask': np.zeros_like(image, dtype=bool),
                'total_area_um2': 0.0,
                'fragmentation_index': 0.0,
            }
        threshold = float(skfilters.threshold_otsu(nonzero))

    mask = image > threshold
    if min_size > 1:
        mask = morphology.remove_small_objects(mask, max_size=min_size)

    labeled, n_cc = ndimage.label(mask)
    total_area = float(mask.sum()) * pixel_size ** 2
    frag_idx = n_cc / total_area if total_area > 0 else 0.0

    return {
        'n_fragments': n_cc,
        'labeled': labeled,
        'mask': mask,
        'total_area_um2': round(total_area, 2),
        'fragmentation_index': round(frag_idx, 6),
    }


# ── Private helpers ──────────────────────────────────────────────────────

def _skeleton_length(skeleton):
    """Compute skeleton length accounting for diagonal connectivity.

    Diagonal steps count as √2, orthogonal steps as 1.
    """
    skel = np.asarray(skeleton, dtype=bool)
    if not skel.any():
        return 0.0

    # Count orthogonal and diagonal neighbors
    kernel_ortho = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], dtype=int)
    kernel_diag = np.array([[1, 0, 1],
                             [0, 0, 0],
                             [1, 0, 1]], dtype=int)

    n_ortho = ndimage.convolve(skel.astype(int), kernel_ortho,
                                mode='constant', cval=0)
    n_diag = ndimage.convolve(skel.astype(int), kernel_diag,
                               mode='constant', cval=0)

    # Each connection is counted twice (once from each end)
    total_ortho = (skel * n_ortho).sum() / 2.0
    total_diag = (skel * n_diag).sum() / 2.0

    return float(total_ortho * 1.0 + total_diag * np.sqrt(2))


def _prune_short_branches(skeleton, min_length):
    """Remove branches shorter than min_length pixels."""
    skel = skeleton.copy()

    # Find junction and endpoint pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)

    for _ in range(min_length):
        neighbor_count = ndimage.convolve(skel.astype(int), kernel,
                                          mode='constant', cval=0)
        # Endpoints with 1 neighbor
        endpoints = skel & (neighbor_count == 1)
        if not endpoints.any():
            break
        skel = skel & ~endpoints

    return skel
