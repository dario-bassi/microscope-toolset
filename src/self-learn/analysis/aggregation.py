"""Collective cell migration and aggregation analysis.

Analyzes populations undergoing chemotactic aggregation (e.g. Dictyostelium),
wound healing migration, or other coordinated cell movement. Detects streaming
(coherent directed flow), aggregation centers (mounds), and measures the
fraction of cells participating in collective behavior.

Functions:
    detect_aggregation_centers  -- Find convergence points from displacement vectors
    detect_streaming            -- Identify coherent cell streams
    aggregation_fraction        -- Measure fraction of cells migrating toward centers
    classify_migration_state    -- Classify cells as streaming/mound/isolated
    collective_order            -- Measure population-level migration coherence
    detect_mounds               -- Density-based mound detection from position series
    detect_onset                -- Detect temporal onset of a changing metric
"""

import numpy as np


def detect_aggregation_centers(positions, displacements, min_convergence=3, radius_fraction=0.15):
    """Find points where cells are converging (aggregation centers/mounds).

    Traces displacement vectors backward to find convergence points where
    multiple cells are heading. Uses a grid-based accumulator for robustness.

    Args:
        positions: (N, 2) array of cell (y, x) positions.
        displacements: (N, 2) array of cell (dy, dx) velocities/displacements.
        min_convergence: int, minimum cells converging on a point to call it
            an aggregation center.
        radius_fraction: float, convergence radius as fraction of field size.

    Returns:
        dict with:
            centers: (M, 2) array of aggregation center (y, x) coordinates.
            n_centers: int.
            convergence_counts: list of int, cells converging on each center.
            convergence_map: 2D array, density of convergence points.
    """
    positions = np.asarray(positions, dtype=float)
    displacements = np.asarray(displacements, dtype=float)

    if len(positions) < min_convergence:
        return {
            "centers": np.zeros((0, 2)),
            "n_centers": 0,
            "convergence_counts": [],
            "convergence_map": np.zeros((1, 1)),
        }

    # Project each cell forward along its displacement vector
    # Where multiple projections converge = aggregation center
    targets = positions + displacements * 5  # project 5 steps ahead

    # Cluster targets using simple distance-based grouping
    y_range = float(np.ptp(positions[:, 0]))
    x_range = float(np.ptp(positions[:, 1]))
    field_size = max(y_range, x_range, 1)
    radius = field_size * radius_fraction

    # Build a small convergence map for the output
    grid_size = 32
    y_min = float(min(positions[:, 0].min(), targets[:, 0].min())) - radius
    x_min = float(min(positions[:, 1].min(), targets[:, 1].min())) - radius
    y_max = float(max(positions[:, 0].max(), targets[:, 0].max())) + radius
    x_max = float(max(positions[:, 1].max(), targets[:, 1].max())) + radius
    map_h = max(y_max - y_min, 1)
    map_w = max(x_max - x_min, 1)

    conv_map = np.zeros((grid_size, grid_size))
    for ty, tx in targets:
        gy = int((ty - y_min) / map_h * (grid_size - 1))
        gx = int((tx - x_min) / map_w * (grid_size - 1))
        gy = np.clip(gy, 0, grid_size - 1)
        gx = np.clip(gx, 0, grid_size - 1)
        conv_map[gy, gx] += 1

    # Cluster targets: greedy grouping within radius
    n_targets = len(targets)
    assigned = np.zeros(n_targets, dtype=bool)
    centers = []
    counts = []

    # Sort by density: start with the target that has most neighbors
    neighbor_counts = np.zeros(n_targets)
    for i in range(n_targets):
        dists = np.sqrt(np.sum((targets - targets[i]) ** 2, axis=1))
        neighbor_counts[i] = np.sum(dists < radius)
    order = np.argsort(-neighbor_counts)

    for idx in order:
        if assigned[idx]:
            continue
        dists = np.sqrt(np.sum((targets - targets[idx]) ** 2, axis=1))
        cluster = (~assigned) & (dists < radius)
        n_in_cluster = int(cluster.sum())
        if n_in_cluster >= min_convergence:
            cluster_pts = targets[cluster]
            center = cluster_pts.mean(axis=0)
            centers.append(center.tolist())
            counts.append(n_in_cluster)
            assigned[cluster] = True

    centers = np.array(centers).reshape(-1, 2) if centers else np.zeros((0, 2))

    return {
        "centers": centers,
        "n_centers": len(centers),
        "convergence_counts": counts,
        "convergence_map": conv_map,
    }


def detect_streaming(
    positions, displacements, angle_threshold=30, min_stream_size=3, neighborhood_radius=None
):
    """Identify coherent cell streams (coordinated directional movement).

    A stream is a group of nearby cells moving in a similar direction.
    Critical for Dictyostelium-like chemotactic aggregation.

    Args:
        positions: (N, 2) array of cell (y, x) positions.
        displacements: (N, 2) array of cell (dy, dx) displacements.
        angle_threshold: float, maximum angular deviation (degrees) within
            a stream.
        min_stream_size: int, minimum cells to form a stream.
        neighborhood_radius: float or None. Search radius for neighbors.
            If None, auto-computed from mean inter-cell distance.

    Returns:
        dict with:
            stream_labels: (N,) array, stream ID for each cell (0=isolated).
            n_streams: int.
            stream_sizes: list of int, cells in each stream.
            mean_stream_direction: list of float, mean angle (degrees) per stream.
            streaming_fraction: float, fraction of cells in streams.
    """
    positions = np.asarray(positions, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    n = len(positions)

    if n < min_stream_size:
        return {
            "stream_labels": np.zeros(n, dtype=int),
            "n_streams": 0,
            "stream_sizes": [],
            "mean_stream_direction": [],
            "streaming_fraction": 0.0,
        }

    # Compute displacement angles
    speeds = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
    angles = np.arctan2(displacements[:, 0], displacements[:, 1])

    # Auto neighborhood radius
    if neighborhood_radius is None:
        # Mean nearest-neighbor distance * 3
        from scipy.spatial import distance_matrix

        dmat = distance_matrix(positions, positions)
        np.fill_diagonal(dmat, np.inf)
        nn_dists = dmat.min(axis=1)
        neighborhood_radius = float(np.mean(nn_dists) * 3)

    # Moving cells only (skip stationary)
    speed_thresh = float(np.median(speeds) * 0.2) if np.median(speeds) > 0 else 0
    moving = speeds > speed_thresh

    # Build adjacency by position + angle similarity
    labels = np.zeros(n, dtype=int)
    current_label = 0
    visited = np.zeros(n, dtype=bool)

    angle_rad = np.deg2rad(angle_threshold)

    for seed in range(n):
        if visited[seed] or not moving[seed]:
            continue

        # BFS to find connected stream
        current_label += 1
        queue = [seed]
        visited[seed] = True
        members = [seed]

        while queue:
            cell = queue.pop(0)
            dists = np.sqrt(np.sum((positions - positions[cell]) ** 2, axis=1))
            neighbors = np.where((dists < neighborhood_radius) & (dists > 0) & moving & ~visited)[0]

            for nb in neighbors:
                angle_diff = abs(angles[nb] - angles[cell])
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                if angle_diff < angle_rad:
                    visited[nb] = True
                    queue.append(nb)
                    members.append(nb)

        if len(members) >= min_stream_size:
            for m in members:
                labels[m] = current_label
        else:
            current_label -= 1  # don't count small groups

    # Compute per-stream stats
    n_streams = int(labels.max())
    stream_sizes = []
    stream_dirs = []
    for s in range(1, n_streams + 1):
        mask = labels == s
        stream_sizes.append(int(mask.sum()))
        mean_dy = float(np.mean(displacements[mask, 0]))
        mean_dx = float(np.mean(displacements[mask, 1]))
        stream_dirs.append(round(float(np.degrees(np.arctan2(mean_dy, mean_dx))), 1))

    streaming_frac = float(np.sum(labels > 0)) / n if n > 0 else 0.0

    return {
        "stream_labels": labels,
        "n_streams": n_streams,
        "stream_sizes": stream_sizes,
        "mean_stream_direction": stream_dirs,
        "streaming_fraction": round(streaming_frac, 4),
    }


def aggregation_fraction(positions, displacements, centers):
    """Measure fraction of cells migrating toward aggregation centers.

    A cell is "aggregating" if its displacement vector has a component
    directed toward any center (positive radial inflow).

    Important: streaming cells heading toward a center ARE aggregating,
    even if they haven't reached the mound yet.

    Args:
        positions: (N, 2) array of (y, x).
        displacements: (N, 2) array of (dy, dx).
        centers: (M, 2) array of center (y, x) coordinates.

    Returns:
        dict with:
            fraction: float, fraction of cells moving toward any center.
            per_center: list of float, fraction heading toward each center.
            n_aggregating: int.
            radial_speeds: (N,) array, radial speed toward nearest center
                (negative = approaching, positive = departing).
    """
    positions = np.asarray(positions, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    n = len(positions)

    if n == 0 or len(centers) == 0:
        return {
            "fraction": 0.0,
            "per_center": [],
            "n_aggregating": 0,
            "radial_speeds": np.zeros(0),
        }

    # For each cell, find nearest center
    radial_speeds = np.zeros(n)
    nearest_center = np.zeros(n, dtype=int)

    for i in range(n):
        dists = np.sqrt(np.sum((centers - positions[i]) ** 2, axis=1))
        nc = int(np.argmin(dists))
        nearest_center[i] = nc

        # Radial component of displacement toward center
        to_center = centers[nc] - positions[i]
        dist = np.sqrt(np.sum(to_center**2))
        if dist > 0:
            unit = to_center / dist
            radial_speeds[i] = float(np.dot(displacements[i], unit))
        else:
            radial_speeds[i] = 0.0

    # Aggregating = moving toward center (positive radial speed)
    aggregating = radial_speeds > 0
    n_agg = int(aggregating.sum())
    fraction = float(n_agg) / n if n > 0 else 0.0

    # Per-center fractions
    per_center = []
    for c in range(len(centers)):
        mask = nearest_center == c
        if mask.sum() > 0:
            per_center.append(round(float(aggregating[mask].mean()), 4))
        else:
            per_center.append(0.0)

    return {
        "fraction": round(fraction, 4),
        "per_center": per_center,
        "n_aggregating": n_agg,
        "radial_speeds": radial_speeds,
    }


def classify_migration_state(positions, displacements, centers=None, mound_radius=None):
    """Classify each cell as streaming, in-mound, or isolated.

    Args:
        positions: (N, 2) array.
        displacements: (N, 2) array.
        centers: (M, 2) array or None. If None, auto-detected.
        mound_radius: float or None. Radius to consider "in mound".
            If None, auto-computed.

    Returns:
        dict with:
            states: (N,) array of str, 'streaming'/'mound'/'isolated'.
            n_streaming: int.
            n_mound: int.
            n_isolated: int.
            centers_used: (M, 2) array.
    """
    positions = np.asarray(positions, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    n = len(positions)

    states = np.array(["isolated"] * n, dtype="U12")

    if n < 3:
        return {
            "states": states,
            "n_streaming": 0,
            "n_mound": 0,
            "n_isolated": n,
            "centers_used": np.zeros((0, 2)),
        }

    # Detect centers if not provided
    if centers is None:
        result = detect_aggregation_centers(positions, displacements)
        centers = result["centers"]
    else:
        centers = np.asarray(centers, dtype=float).reshape(-1, 2)

    # Auto mound radius
    if mound_radius is None:
        if len(centers) > 0:
            field_size = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1)
            mound_radius = field_size * 0.1
        else:
            mound_radius = 10.0

    # Mark cells near centers as "mound"
    for i in range(n):
        if len(centers) > 0:
            dists = np.sqrt(np.sum((centers - positions[i]) ** 2, axis=1))
            if np.min(dists) < mound_radius:
                states[i] = "mound"

    # Detect streaming cells
    streaming = detect_streaming(positions, displacements)
    for i in range(n):
        if streaming["stream_labels"][i] > 0 and states[i] != "mound":
            states[i] = "streaming"

    n_str = int(np.sum(states == "streaming"))
    n_mnd = int(np.sum(states == "mound"))
    n_iso = int(np.sum(states == "isolated"))

    return {
        "states": states,
        "n_streaming": n_str,
        "n_mound": n_mnd,
        "n_isolated": n_iso,
        "centers_used": centers,
    }


def collective_order(displacements):
    """Measure population-level migration coherence (order parameter).

    The order parameter ranges from 0 (random movement) to 1 (all cells
    moving in the same direction). Based on the Vicsek model.

    Args:
        displacements: (N, 2) array of (dy, dx) displacements.

    Returns:
        dict with:
            order_parameter: float, 0=disordered, 1=perfectly aligned.
            mean_direction: float, mean heading in degrees.
            angular_spread: float, circular std in degrees.
            speed_cv: float, coefficient of variation of speeds.
    """
    displacements = np.asarray(displacements, dtype=float)
    n = len(displacements)

    if n < 2:
        return {
            "order_parameter": 1.0,
            "mean_direction": 0.0,
            "angular_spread": 0.0,
            "speed_cv": 0.0,
        }

    speeds = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)

    # Filter out stationary cells
    moving = speeds > np.median(speeds) * 0.1
    if moving.sum() < 2:
        return {
            "order_parameter": 0.0,
            "mean_direction": 0.0,
            "angular_spread": 360.0,
            "speed_cv": 0.0,
        }

    active = displacements[moving]
    active_speeds = speeds[moving]

    # Normalize to unit vectors
    norms = np.sqrt(active[:, 0] ** 2 + active[:, 1] ** 2)
    norms = np.maximum(norms, 1e-10)
    unit_vecs = active / norms[:, np.newaxis]

    # Order parameter = |mean unit vector|
    mean_vec = np.mean(unit_vecs, axis=0)
    order = float(np.sqrt(mean_vec[0] ** 2 + mean_vec[1] ** 2))

    # Mean direction
    mean_dir = float(np.degrees(np.arctan2(mean_vec[0], mean_vec[1])))

    # Angular spread (circular std)
    angles = np.arctan2(unit_vecs[:, 0], unit_vecs[:, 1])
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    if R > 0 and R < 1:
        circ_std = float(np.degrees(np.sqrt(-2 * np.log(R))))
    elif R >= 1:
        circ_std = 0.0
    else:
        circ_std = 360.0

    # Speed CV
    speed_cv = (
        float(np.std(active_speeds) / np.mean(active_speeds)) if np.mean(active_speeds) > 0 else 0
    )

    return {
        "order_parameter": round(order, 4),
        "mean_direction": round(mean_dir, 1),
        "angular_spread": round(circ_std, 1),
        "speed_cv": round(speed_cv, 4),
    }


def detect_mounds(
    positions_early,
    positions_late,
    field_size=512,
    sigma=20,
    min_distance=50,
    min_accumulation=None,
):
    """Detect mounds by comparing spatial density between early and late frames.

    Mounds are regions where cell density increased significantly over time.
    This is more robust than velocity-based center detection because it uses
    the actual outcome (where cells accumulated) rather than instantaneous
    movement direction.

    Args:
        positions_early: (N, 2) array of (y, x) in early frames.
        positions_late: (M, 2) array of (y, x) in late frames.
        field_size: int or (h, w), image dimensions.
        sigma: float, Gaussian smoothing radius. Match to expected mound size.
        min_distance: int, minimum distance between mound centers.
        min_accumulation: float or None. Minimum density increase to qualify
            as a mound. If None, auto-computed from the 90th percentile.

    Returns:
        dict with:
            mounds: (K, 2) array of mound center (y, x) coordinates.
            n_mounds: int.
            accumulation_values: list of float, density increase at each mound.
            density_change: 2D array, late_density - early_density.
            density_early: 2D density map of early positions.
            density_late: 2D density map of late positions.
    """
    from .spatial import density_map, detect_density_peaks

    early = density_map(positions_early, field_size=field_size, sigma=sigma)
    late = density_map(positions_late, field_size=field_size, sigma=sigma)

    change = late["map"] - early["map"]

    # Find peaks in density increase
    if min_accumulation is None:
        # Auto: use 90th percentile of positive changes
        positive = change[change > 0]
        if len(positive) > 0:
            min_accumulation = float(np.percentile(positive, 90))
        else:
            min_accumulation = 0.0

    peaks_result = detect_density_peaks(
        change, min_distance=min_distance, min_peak_value=min_accumulation
    )

    # Scale peak coordinates back to original field size
    map_h, map_w = change.shape
    if isinstance(field_size, (int, float)):
        fh = fw = int(field_size)
    else:
        fh, fw = int(field_size[0]), int(field_size[1])

    scale_y = fh / map_h if map_h > 0 else 1
    scale_x = fw / map_w if map_w > 0 else 1

    if peaks_result["n_peaks"] > 0:
        mounds = peaks_result["peaks"].astype(float).copy()
        mounds[:, 0] *= scale_y
        mounds[:, 1] *= scale_x
    else:
        mounds = np.zeros((0, 2))

    return {
        "mounds": mounds,
        "n_mounds": peaks_result["n_peaks"],
        "accumulation_values": peaks_result["peak_values"],
        "density_change": change,
        "density_early": early["map"],
        "density_late": late["map"],
    }


def detect_onset(metric_series, baseline_frames=5, threshold_sigma=2.0, direction="decrease"):
    """Detect when a temporal metric starts changing significantly.

    Compares each value to a baseline established from the first N frames.
    Onset is the first frame where the metric deviates by more than
    threshold_sigma standard deviations from the baseline.

    Use cases: aggregation onset (NN distance decreasing), drug response
    onset (intensity changing), streaming onset (order parameter increasing).

    Args:
        metric_series: 1D array-like of metric values over time.
        baseline_frames: int, number of initial frames to use as baseline.
        threshold_sigma: float, number of std devs for onset detection.
        direction: 'decrease', 'increase', or 'any'. Which direction of
            change to look for.

    Returns:
        dict with:
            onset_index: int or None, first frame index exceeding threshold.
            baseline_mean: float, mean of baseline frames.
            baseline_std: float, std of baseline frames.
            threshold: float, the threshold value used.
            z_scores: 1D array of z-scores relative to baseline.
    """
    values = np.asarray(metric_series, dtype=float)
    n = len(values)

    if n < baseline_frames + 1:
        return {
            "onset_index": None,
            "baseline_mean": float(np.mean(values)) if n > 0 else 0.0,
            "baseline_std": 0.0,
            "threshold": 0.0,
            "z_scores": np.zeros(n),
        }

    bl_mean = float(np.mean(values[:baseline_frames]))
    bl_std = float(np.std(values[:baseline_frames]))
    if bl_std < 1e-10:
        overall_std = float(np.std(values))
        bl_std = overall_std * 0.1 if overall_std > 0 else 1.0

    z_scores = (values - bl_mean) / bl_std

    if direction == "decrease":
        threshold = -threshold_sigma
        onset_idx = None
        for i in range(baseline_frames, n):
            if z_scores[i] < threshold:
                onset_idx = i
                break
    elif direction == "increase":
        threshold = threshold_sigma
        onset_idx = None
        for i in range(baseline_frames, n):
            if z_scores[i] > threshold:
                onset_idx = i
                break
    else:  # 'any'
        threshold = threshold_sigma
        onset_idx = None
        for i in range(baseline_frames, n):
            if abs(z_scores[i]) > threshold:
                onset_idx = i
                break

    return {
        "onset_index": onset_idx,
        "baseline_mean": bl_mean,
        "baseline_std": bl_std,
        "threshold": float(threshold),
        "z_scores": z_scores,
    }
