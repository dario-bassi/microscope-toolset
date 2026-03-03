"""Cell tracking across timepoints using optimal matching.

Key lesson (Ch62, Ch65): Always use Hungarian algorithm (linear_sum_assignment)
instead of greedy nearest-neighbor. Greedy fails when displacements > half
inter-cell distance, causing mismatched trajectories.

Key lesson (Ch342): Raw centroid tracking overestimates speed by ~45% for
organisms with sinusoidal locomotion (C. elegans, sperm). Use smooth_trajectory()
before computing speed.
"""

import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def match_frames(cells_t0, cells_t1):
    """Match cells between two frames using Hungarian algorithm.

    Args:
        cells_t0: list of dicts with 'x', 'y' keys (frame 0 cells)
        cells_t1: list of dicts with 'x', 'y' keys (frame 1 cells)

    Returns:
        list of (idx_t0, idx_t1, displacement) tuples, sorted by idx_t0
    """
    coords_t0 = np.array([[c['x'], c['y']] for c in cells_t0])
    coords_t1 = np.array([[c['x'], c['y']] for c in cells_t1])
    D = cdist(coords_t0, coords_t1)
    row_ind, col_ind = linear_sum_assignment(D)
    matches = []
    for r, c in zip(row_ind, col_ind):
        disp = float(D[r, c])
        matches.append((int(r), int(c), round(disp, 2)))
    return sorted(matches, key=lambda x: x[0])


def track_multiframe(frame_cells):
    """Track cells across multiple frames using Hungarian matching.

    Args:
        frame_cells: list of lists, where each inner list contains cell dicts
                     with 'x', 'y' keys. frame_cells[0] defines trajectory IDs.

    Returns:
        dict with:
            trajectories: list of lists, each inner list has one cell dict per frame
            path_lengths: total path length per trajectory
            step_displacements: all individual step displacements
            mean_path: mean total path length
            max_path: max total path length
            fastest_idx: index of trajectory with max path
    """
    n_cells = len(frame_cells[0])
    trajectories = [[frame_cells[0][i]] for i in range(n_cells)]

    for f_idx in range(1, len(frame_cells)):
        prev_coords = np.array([
            [trajectories[i][-1]['x'], trajectories[i][-1]['y']]
            for i in range(n_cells)
        ])
        curr = frame_cells[f_idx]
        curr_coords = np.array([[c['x'], c['y']] for c in curr])
        D = cdist(prev_coords, curr_coords)
        row_ind, col_ind = linear_sum_assignment(D)
        for r, c in zip(row_ind, col_ind):
            trajectories[r].append(curr[c])

    # Compute path lengths
    path_lengths = []
    all_steps = []
    for traj in trajectories:
        path = 0.0
        for i in range(1, len(traj)):
            dx = traj[i]['x'] - traj[i - 1]['x']
            dy = traj[i]['y'] - traj[i - 1]['y']
            step = math.sqrt(dx ** 2 + dy ** 2)
            path += step
            all_steps.append(step)
        path_lengths.append(path)

    fastest_idx = int(np.argmax(path_lengths))

    return {
        'trajectories': trajectories,
        'path_lengths': [round(p, 2) for p in path_lengths],
        'step_displacements': [round(s, 2) for s in all_steps],
        'mean_path': round(float(np.mean(path_lengths)), 2),
        'max_path': round(float(max(path_lengths)), 2),
        'fastest_idx': fastest_idx,
        'mean_step': round(float(np.mean(all_steps)), 2) if all_steps else 0.0,
    }


def displacement_vectors(cells_t0, cells_t1):
    """Compute displacement vectors between two matched frames.

    Args:
        cells_t0, cells_t1: lists of cell dicts with 'x', 'y'

    Returns:
        dict with:
            matches: list of (idx_t0, idx_t1, displacement)
            vectors: list of (dx, dy) per match
            mean_displacement: mean displacement
            max_displacement: max displacement
            fastest_idx: index in matches with max displacement
    """
    matches = match_frames(cells_t0, cells_t1)
    vectors = []
    displacements = []
    for r, c, d in matches:
        dx = cells_t1[c]['x'] - cells_t0[r]['x']
        dy = cells_t1[c]['y'] - cells_t0[r]['y']
        vectors.append((round(dx, 2), round(dy, 2)))
        displacements.append(d)

    fastest = int(np.argmax(displacements)) if displacements else 0

    return {
        'matches': matches,
        'vectors': vectors,
        'mean_displacement': round(float(np.mean(displacements)), 2) if displacements else 0.0,
        'max_displacement': round(float(max(displacements)), 2) if displacements else 0.0,
        'fastest_idx': fastest,
    }


def smooth_trajectory(positions, window=5, method='rolling'):
    """Smooth a 2D trajectory to remove oscillation noise.

    Critical for organisms with sinusoidal locomotion (C. elegans, sperm)
    where raw centroid tracking overestimates speed by ~45%.

    Args:
        positions: (N, 2) array of (x, y) coordinates
        window: smoothing window size (must be odd for savgol)
        method: 'rolling' for moving average, 'savgol' for Savitzky-Golay

    Returns:
        (N, 2) array of smoothed positions
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 3:
        return positions.copy()

    if method == 'savgol':
        from scipy.signal import savgol_filter
        w = min(window, len(positions))
        if w % 2 == 0:
            w -= 1
        w = max(w, 3)
        order = min(2, w - 1)
        smoothed = np.column_stack([
            savgol_filter(positions[:, 0], w, order),
            savgol_filter(positions[:, 1], w, order),
        ])
    else:  # rolling average
        kernel = np.ones(window) / window
        pad = window // 2
        # Pad with edge values to preserve trajectory length
        x_padded = np.pad(positions[:, 0], pad, mode='edge')
        y_padded = np.pad(positions[:, 1], pad, mode='edge')
        smoothed = np.column_stack([
            np.convolve(x_padded, kernel, mode='valid'),
            np.convolve(y_padded, kernel, mode='valid'),
        ])
        # Trim to original length if convolution produced extra
        smoothed = smoothed[:len(positions)]

    return smoothed


def trajectory_speed(positions, dt=1.0, smooth_window=5):
    """Compute corrected speed from a trajectory with optional smoothing.

    Args:
        positions: (N, 2) array of (x, y) coordinates
        dt: time between frames
        smooth_window: smoothing window (0 or 1 to disable)

    Returns:
        dict with:
            mean_speed: mean speed (distance per dt)
            total_path: total smoothed path length
            total_displacement: straight-line start-to-end distance
            sinuosity: path_length / displacement (1.0 = straight line)
            speeds: per-frame speeds array
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return {
            'mean_speed': 0.0, 'total_path': 0.0,
            'total_displacement': 0.0, 'sinuosity': 1.0,
            'speeds': np.array([]),
        }

    if smooth_window > 1:
        smoothed = smooth_trajectory(positions, window=smooth_window)
    else:
        smoothed = positions

    diffs = np.diff(smoothed, axis=0)
    step_dists = np.sqrt((diffs ** 2).sum(axis=1))
    total_path = float(step_dists.sum())
    speeds = step_dists / dt

    displacement = float(np.sqrt(
        (positions[-1, 0] - positions[0, 0]) ** 2 +
        (positions[-1, 1] - positions[0, 1]) ** 2
    ))
    sinuosity = total_path / displacement if displacement > 0 else float('inf')

    return {
        'mean_speed': round(float(speeds.mean()), 2),
        'total_path': round(total_path, 2),
        'total_displacement': round(displacement, 2),
        'sinuosity': round(sinuosity, 3),
        'speeds': speeds,
    }


def cell_dispersion(image, threshold=None, min_area=20, origin=None):
    """Compute dispersion of cells from a binary/thresholded image.

    Segments the image into individual cells via connected components,
    computes per-cell centroids, and measures spatial dispersion.
    This is the correct approach for photoconversion tracking — using
    cell-level centroids instead of pixel-level intensity weighting.

    Key lesson (Ch354): Pixel-level intensity weighting overestimates
    dispersion by ~40%. Segmenting individual cells as CCs and averaging
    their centroids gives accurate dispersion (GT=65, pixel method=93.7).

    Args:
        image: 2D grayscale image (e.g., photoconverted channel).
        threshold: Intensity threshold for binarization. If None, uses
            Otsu's method.
        min_area: Minimum CC area in pixels to count as a cell.
        origin: (x, y) reference point for dispersion calculation.
            If None, uses the centroid of all detected cells.

    Returns:
        dict with:
            n_cells: number of detected cells
            centroids: list of (x, y) per cell
            origin: (x, y) reference point used
            mean_distance: mean distance from origin
            std_distance: std of distances from origin
            max_distance: max distance from origin
            rms_distance: root-mean-square distance from origin
    """
    import cv2

    img = image.astype(np.float32)

    if threshold is None:
        # Otsu's method on uint8
        img8 = np.clip(img, 0, 255).astype(np.uint8)
        threshold, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(threshold)

    binary = (img > threshold).astype(np.uint8)
    n_labels, labels, stats, centroids_cv = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Filter by area (skip background label 0)
    cell_centroids = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids_cv[i]
            cell_centroids.append((float(cx), float(cy)))

    if not cell_centroids:
        return {
            'n_cells': 0, 'centroids': [],
            'origin': origin or (0.0, 0.0),
            'mean_distance': 0.0, 'std_distance': 0.0,
            'max_distance': 0.0, 'rms_distance': 0.0,
        }

    centroids_arr = np.array(cell_centroids)

    if origin is None:
        origin = (float(centroids_arr[:, 0].mean()),
                  float(centroids_arr[:, 1].mean()))

    origin_arr = np.array(origin)
    distances = np.sqrt(((centroids_arr - origin_arr) ** 2).sum(axis=1))

    return {
        'n_cells': len(cell_centroids),
        'centroids': cell_centroids,
        'origin': origin,
        'mean_distance': round(float(distances.mean()), 2),
        'std_distance': round(float(distances.std()), 2),
        'max_distance': round(float(distances.max()), 2),
        'rms_distance': round(float(np.sqrt((distances ** 2).mean())), 2),
    }


def trajectory_curvature(positions, smooth_window=5):
    """Compute local curvature along a trajectory.

    Curvature = |d(angle)/ds|. High curvature = sharp turns.

    Args:
        positions: (N, 2) array
        smooth_window: smoothing window before curvature computation

    Returns:
        dict with:
            curvatures: per-point curvature array (N-2,)
            mean_curvature: mean absolute curvature
            max_curvature: max curvature (sharpest turn)
            turn_angles: per-step heading change in degrees (N-2,)
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 3:
        return {
            'curvatures': np.array([]),
            'mean_curvature': 0.0,
            'max_curvature': 0.0,
            'turn_angles': np.array([]),
        }

    if smooth_window > 1:
        positions = smooth_trajectory(positions, window=smooth_window)

    # Heading angles
    diffs = np.diff(positions, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    # Angle changes (handle wraparound)
    dangle = np.diff(angles)
    dangle = (dangle + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    # Step lengths for curvature normalization
    step_lens = np.sqrt((diffs ** 2).sum(axis=1))
    mean_steps = (step_lens[:-1] + step_lens[1:]) / 2
    mean_steps = np.maximum(mean_steps, 1e-10)  # avoid division by zero

    curvatures = np.abs(dangle) / mean_steps

    return {
        'curvatures': curvatures,
        'mean_curvature': round(float(curvatures.mean()), 4),
        'max_curvature': round(float(curvatures.max()), 4),
        'turn_angles': np.degrees(dangle),
    }


def match_centroids(coords1, coords2, max_displacement=None):
    """Match cell centroids between two frames using Hungarian algorithm.

    Like match_frames() but takes numpy arrays directly instead of dicts.
    Convenient for MDA workflows where detection returns centroid arrays.

    Args:
        coords1: (N, 2) array of (x, y) centroids in frame 1.
        coords2: (M, 2) array of (x, y) centroids in frame 2.
        max_displacement: Maximum allowed displacement. Matches exceeding
            this are discarded. If None, all matches are kept.

    Returns:
        dict with:
            row_idx: array of matched indices in coords1
            col_idx: array of matched indices in coords2
            displacements: array of displacement magnitudes per match
            vectors: (K, 2) array of (dx, dy) per match
            mean_displacement: mean displacement of valid matches
    """
    coords1 = np.asarray(coords1, dtype=float)
    coords2 = np.asarray(coords2, dtype=float)

    if len(coords1) == 0 or len(coords2) == 0:
        return {
            'row_idx': np.array([], dtype=int),
            'col_idx': np.array([], dtype=int),
            'displacements': np.array([]),
            'vectors': np.empty((0, 2)),
            'mean_displacement': 0.0,
        }

    cost = cdist(coords1, coords2)
    ri, ci = linear_sum_assignment(cost)

    if max_displacement is not None:
        valid = cost[ri, ci] <= max_displacement
        ri = ri[valid]
        ci = ci[valid]

    displacements = cost[ri, ci]
    vectors = coords2[ci] - coords1[ri]

    return {
        'row_idx': ri,
        'col_idx': ci,
        'displacements': displacements,
        'vectors': vectors,
        'mean_displacement': float(displacements.mean()) if len(displacements) > 0 else 0.0,
    }


def population_speeds(centroids_per_frame, timestamps, pixel_size=1.0,
                      max_displacement=None, time_scale=1.0):
    """Compute speed distribution from multi-frame centroid data.

    Matches centroids between consecutive frames using Hungarian algorithm
    and converts displacements to physical speeds. Designed for MDA timelapse
    workflows where you collect frames via on_frame callback.

    Args:
        centroids_per_frame: list of (N_i, 2) arrays, one per frame.
            Each array contains (x, y) centroids in camera pixels.
        timestamps: list of wall-clock timestamps (seconds), one per frame.
        pixel_size: um per camera pixel (e.g., 0.25 at 40x).
        max_displacement: Max displacement in camera pixels to accept a match.
            If None, uses half the FOV diagonal as default.
        time_scale: Simulation time scale multiplier. For real microscopy, use 1.0.
            For sim with --time-scale N, dt_biological = dt_wall * N.

    Returns:
        dict with:
            speeds: 1D array of all per-match speeds in um/s
            mean_speed: mean speed in um/s
            std_speed: std of speeds
            median_speed: median speed
            n_measurements: total number of speed measurements
            n_frames: number of frames used
            speeds_per_pair: list of arrays, speeds from each frame pair
    """
    if len(centroids_per_frame) < 2:
        return {
            'speeds': np.array([]),
            'mean_speed': 0.0, 'std_speed': 0.0, 'median_speed': 0.0,
            'n_measurements': 0, 'n_frames': len(centroids_per_frame),
            'speeds_per_pair': [],
        }

    all_speeds = []
    speeds_per_pair = []

    for i in range(len(centroids_per_frame) - 1):
        c1 = np.asarray(centroids_per_frame[i], dtype=float)
        c2 = np.asarray(centroids_per_frame[i + 1], dtype=float)

        if len(c1) < 2 or len(c2) < 2:
            speeds_per_pair.append(np.array([]))
            continue

        dt_wall = timestamps[i + 1] - timestamps[i]
        if dt_wall <= 0:
            speeds_per_pair.append(np.array([]))
            continue

        dt_bio = dt_wall * time_scale

        result = match_centroids(c1, c2, max_displacement=max_displacement)
        pair_speeds = result['displacements'] * pixel_size / dt_bio
        speeds_per_pair.append(pair_speeds)
        all_speeds.extend(pair_speeds.tolist())

    speeds_arr = np.array(all_speeds) if all_speeds else np.array([])

    return {
        'speeds': speeds_arr,
        'mean_speed': float(speeds_arr.mean()) if len(speeds_arr) > 0 else 0.0,
        'std_speed': float(speeds_arr.std()) if len(speeds_arr) > 0 else 0.0,
        'median_speed': float(np.median(speeds_arr)) if len(speeds_arr) > 0 else 0.0,
        'n_measurements': len(speeds_arr),
        'n_frames': len(centroids_per_frame),
        'speeds_per_pair': speeds_per_pair,
    }


def detect_clusters(positions, neighbor_radius=30, min_neighbors=5, merge_radius=40):
    """Find spatial clusters of cells from centroid positions.

    Uses local density to identify cluster members, then groups
    nearby high-density cells into distinct clusters.

    Useful for: aggregation centers, colony detection, mound formation.

    Args:
        positions: (N, 2) array of (y, x) or (row, col) coordinates.
        neighbor_radius: Radius within which to count neighbors.
        min_neighbors: Minimum neighbor count to be a cluster member.
        merge_radius: Max distance between cluster members to merge.

    Returns:
        dict with:
            n_clusters: Number of clusters found.
            centers: List of (y, x) cluster center coordinates.
            sizes: List of number of high-density cells per cluster.
            labels: Array of cluster labels (-1 = not in cluster).
            cells_per_cluster: List of indices of all cells within
                merge_radius of each cluster center.
            local_density: Array of neighbor counts per cell.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)

    if n < 2:
        return {
            'n_clusters': 0, 'centers': [], 'sizes': [],
            'labels': np.full(n, -1, dtype=int),
            'cells_per_cluster': [],
            'local_density': np.zeros(n),
        }

    D = cdist(positions, positions)
    local_density = np.array([(D[i] < neighbor_radius).sum() - 1 for i in range(n)])

    # Find high-density cells
    hd_mask = local_density >= min_neighbors
    hd_indices = np.where(hd_mask)[0]

    if len(hd_indices) == 0:
        return {
            'n_clusters': 0, 'centers': [], 'sizes': [],
            'labels': np.full(n, -1, dtype=int),
            'cells_per_cluster': [],
            'local_density': local_density,
        }

    # Cluster high-density cells using connected components
    hd_positions = positions[hd_indices]
    hd_D = cdist(hd_positions, hd_positions)
    adjacency = hd_D < merge_radius

    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    n_clusters, component_labels = connected_components(
        csr_matrix(adjacency), directed=False
    )

    # Compute cluster centers and sizes
    labels = np.full(n, -1, dtype=int)
    centers = []
    sizes = []
    cells_per_cluster = []

    for c in range(n_clusters):
        member_mask = component_labels == c
        member_indices = hd_indices[member_mask]
        cluster_pos = positions[member_indices]
        center = cluster_pos.mean(axis=0)
        centers.append((float(center[0]), float(center[1])))
        sizes.append(int(member_mask.sum()))

        # Label high-density members
        for idx in member_indices:
            labels[idx] = c

        # Find ALL cells within merge_radius of center
        dists_to_center = np.sqrt(((positions - center) ** 2).sum(axis=1))
        nearby = np.where(dists_to_center < merge_radius)[0]
        cells_per_cluster.append(nearby.tolist())

    return {
        'n_clusters': n_clusters,
        'centers': centers,
        'sizes': sizes,
        'labels': labels,
        'cells_per_cluster': cells_per_cluster,
        'local_density': local_density,
    }
