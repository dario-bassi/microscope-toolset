"""Time-series event detection for microscopy timelapse.

Detects discrete biological events from frame-to-frame changes in cell
populations: division, arrival/departure, morphology transitions, and
statistical changepoints. Complements the temporal module (1D signal
analysis) with cell-level and population-level event detection.

Key functions:
    detect_division     -- Find cell division events (area split + count increase)
    detect_arrival      -- Detect new cells appearing / leaving FOV
    detect_morphology_change -- Track shape transitions over time
    detect_changepoint  -- Statistical change-point detection in a time series
    classify_trajectory -- Label motion states (moving/stopped/turning)
"""

import numpy as np


def detect_division(areas_per_frame, labels_per_frame=None, parent_threshold=0.6):
    """Detect cell division events from per-frame area measurements.

    A division is detected when one object disappears and two smaller
    objects appear nearby whose combined area matches the parent.

    If no labeled images are provided, uses a simpler heuristic: count
    increases accompanied by total-area conservation.

    Args:
        areas_per_frame: List of lists. areas_per_frame[t] = list of cell
            areas at frame t. Sorted order not required.
        labels_per_frame: Optional list of labeled images (2D int arrays).
            If provided, spatial proximity is used for parent assignment.
        parent_threshold: Daughter areas must sum to at least this fraction
            of the parent area to qualify as a division (0-1).

    Returns:
        dict with:
            events: List of dicts, each with 'frame', 'parent_area',
                'daughter_areas', 'confidence'.
            n_divisions: Total division events detected.
            division_frames: Sorted list of frames where divisions occurred.
    """
    n_frames = len(areas_per_frame)
    events = []

    for t in range(1, n_frames):
        prev = sorted(areas_per_frame[t - 1], reverse=True)
        curr = sorted(areas_per_frame[t], reverse=True)

        # Net gain in cell count
        n_new = len(curr) - len(prev)
        if n_new < 1:
            continue

        total_prev = sum(prev)
        total_curr = sum(curr)

        # Total area should be roughly conserved (within 30%)
        if total_prev > 0 and abs(total_curr - total_prev) / total_prev > 0.3:
            continue

        # Find plausible parent-daughter matches:
        # A parent in prev that has no match in curr, but two daughters sum ~= parent
        used_curr = [False] * len(curr)
        for parent_area in prev:
            # Look for pairs of current cells that sum to parent_area
            for i in range(len(curr)):
                if used_curr[i]:
                    continue
                for j in range(i + 1, len(curr)):
                    if used_curr[j]:
                        continue
                    daughter_sum = curr[i] + curr[j]
                    if daughter_sum >= parent_area * parent_threshold:
                        # Each daughter should be meaningfully smaller than parent
                        if curr[i] < parent_area * 0.85 and curr[j] < parent_area * 0.85:
                            # Plausible division
                            ratio = min(daughter_sum, parent_area) / max(daughter_sum, parent_area)
                            events.append({
                                'frame': t,
                                'parent_area': float(parent_area),
                                'daughter_areas': [float(curr[i]), float(curr[j])],
                                'confidence': float(ratio),
                            })
                            used_curr[i] = True
                            used_curr[j] = True
                            break  # move to next parent
                if used_curr[i]:
                    break  # pair found for a parent

    division_frames = sorted(set(e['frame'] for e in events))

    return {
        'events': events,
        'n_divisions': len(events),
        'division_frames': division_frames,
    }


def detect_arrival(centroids_per_frame, fov_size, margin=10, max_match_dist=30):
    """Detect cells arriving at or departing from the field of view.

    Arrivals: cells in current frame near FOV edge with no match in previous.
    Departures: cells in previous frame near FOV edge with no match in current.

    Args:
        centroids_per_frame: List of arrays, each (N, 2) with (y, x) coords.
        fov_size: (height, width) of the field of view.
        margin: Distance from edge (px) to consider a cell as entering/leaving.
        max_match_dist: Maximum distance (px) to match cells between frames.

    Returns:
        dict with:
            arrivals: List of dicts with 'frame', 'position', 'edge'.
            departures: List of dicts with 'frame', 'position', 'edge'.
            n_arrivals: Total arrivals detected.
            n_departures: Total departures detected.
            net_flux: arrivals - departures.
    """
    h, w = fov_size
    arrivals = []
    departures = []

    def near_edge(y, x):
        """Return which edge a point is near, or None."""
        edges = []
        if y < margin:
            edges.append('top')
        if y > h - margin:
            edges.append('bottom')
        if x < margin:
            edges.append('left')
        if x > w - margin:
            edges.append('right')
        return edges

    n_frames = len(centroids_per_frame)
    for t in range(1, n_frames):
        prev = np.atleast_2d(centroids_per_frame[t - 1]) if len(centroids_per_frame[t - 1]) > 0 else np.empty((0, 2))
        curr = np.atleast_2d(centroids_per_frame[t]) if len(centroids_per_frame[t]) > 0 else np.empty((0, 2))

        # Match current to previous by nearest-neighbor
        matched_curr = set()
        matched_prev = set()

        if len(prev) > 0 and len(curr) > 0:
            from scipy.spatial.distance import cdist
            dists = cdist(curr, prev)
            for ci in range(len(curr)):
                pi = int(np.argmin(dists[ci]))
                if dists[ci, pi] < max_match_dist:
                    matched_curr.add(ci)
                    matched_prev.add(pi)

        # Unmatched in current frame near edge = arrival
        for ci in range(len(curr)):
            if ci not in matched_curr:
                y, x = float(curr[ci, 0]), float(curr[ci, 1])
                edges = near_edge(y, x)
                if edges:
                    arrivals.append({
                        'frame': t,
                        'position': (y, x),
                        'edge': edges[0],
                    })

        # Unmatched in previous frame near edge = departure
        for pi in range(len(prev)):
            if pi not in matched_prev:
                y, x = float(prev[pi, 0]), float(prev[pi, 1])
                edges = near_edge(y, x)
                if edges:
                    departures.append({
                        'frame': t,
                        'position': (y, x),
                        'edge': edges[0],
                    })

    return {
        'arrivals': arrivals,
        'departures': departures,
        'n_arrivals': len(arrivals),
        'n_departures': len(departures),
        'net_flux': len(arrivals) - len(departures),
    }


def detect_morphology_change(measurements_per_frame, feature='eccentricity',
                             threshold=0.3, min_duration=2):
    """Detect morphology transitions in tracked cells.

    Monitors a shape feature over time and flags frames where the feature
    changes abruptly. Useful for detecting rounding (pre-division), elongation
    (migration onset), or spreading (adhesion).

    Args:
        measurements_per_frame: List of dicts (one per frame) with the
            feature key. All dicts should describe the SAME tracked cell.
        feature: Which morphological feature to track
            (e.g., 'eccentricity', 'area', 'aspect_ratio').
        threshold: Minimum absolute change in feature to count as a transition.
        min_duration: Minimum frames the new state must persist to be confirmed.

    Returns:
        dict with:
            transitions: List of dicts with 'frame', 'from_value', 'to_value',
                'direction' ('increase' or 'decrease').
            n_transitions: Number of detected transitions.
            time_series: The extracted feature values over time.
    """
    values = []
    for m in measurements_per_frame:
        if isinstance(m, dict):
            values.append(float(m.get(feature, 0)))
        else:
            values.append(float(m))
    values = np.array(values)
    n = len(values)

    transitions = []

    if n < min_duration + 1:
        return {
            'transitions': [],
            'n_transitions': 0,
            'time_series': values.tolist(),
        }

    # Use rolling window comparison: mean of [i-w:i] vs mean of [i:i+w]
    # This detects transitions even when smoothing would blur them
    w = max(min_duration, 2)

    i = w
    while i <= n - w:
        before = values[max(0, i - w):i].mean()
        after = values[i:min(n, i + w)].mean()
        delta = after - before

        if abs(delta) >= threshold:
            transitions.append({
                'frame': i,
                'from_value': float(before),
                'to_value': float(after),
                'direction': 'increase' if delta > 0 else 'decrease',
            })
            i += w  # skip past this transition
            continue
        i += 1

    return {
        'transitions': transitions,
        'n_transitions': len(transitions),
        'time_series': values.tolist(),
    }


def detect_changepoint(signal, method='cusum', threshold=None, min_segment=5):
    """Detect statistical change-points in a 1D time series.

    Finds frames where the statistical properties of the signal change
    abruptly. Useful for drug addition timing, temperature shifts,
    growth phase transitions.

    Args:
        signal: 1D array of measurements over time.
        method: 'cusum' (cumulative sum) or 'variance' (variance ratio).
        threshold: Detection threshold. If None, auto-computed from signal.
        min_segment: Minimum segment length between change-points.

    Returns:
        dict with:
            changepoints: List of frame indices where changes occur.
            n_changepoints: Number of detected change-points.
            segments: List of (start, end) tuples for each segment.
            segment_means: Mean value in each segment.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if n < 2 * min_segment:
        return {
            'changepoints': [],
            'n_changepoints': 0,
            'segments': [(0, n)],
            'segment_means': [float(signal.mean())] if n > 0 else [],
        }

    if method == 'cusum':
        changepoints = _cusum_changepoints(signal, threshold, min_segment)
    elif method == 'variance':
        changepoints = _variance_changepoints(signal, threshold, min_segment)
    else:
        raise ValueError(f"method must be 'cusum' or 'variance', got '{method}'")

    # Build segments
    boundaries = [0] + changepoints + [n]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    segment_means = [float(signal[s:e].mean()) for s, e in segments]

    return {
        'changepoints': changepoints,
        'n_changepoints': len(changepoints),
        'segments': segments,
        'segment_means': segment_means,
    }


def _cusum_changepoints(signal, threshold, min_segment):
    """CUSUM-based change-point detection.

    Uses a sliding comparison: for each candidate split point, compares
    the mean of the left segment to the mean of the right segment.
    """
    n = len(signal)
    if threshold is None:
        threshold = float(np.std(signal) * 0.5)
    if threshold <= 0:
        return []

    changepoints = []

    def _find_best_split(start, end):
        """Find the single best split point in signal[start:end]."""
        length = end - start
        if length < 2 * min_segment:
            return -1

        seg = signal[start:end]
        seg_mean = seg.mean()
        best_score = 0
        best_idx = -1

        for i in range(min_segment, length - min_segment):
            left_mean = seg[:i].mean()
            right_mean = seg[i:].mean()
            score = abs(right_mean - left_mean)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold:
            return start + best_idx
        return -1

    # Recursive splitting
    def _split(start, end, depth=0):
        if depth > 5 or end - start < 2 * min_segment:
            return
        cp = _find_best_split(start, end)
        if cp > 0:
            changepoints.append(cp)
            _split(start, cp, depth + 1)
            _split(cp, end, depth + 1)

    _split(0, n)
    changepoints.sort()
    return changepoints


def _variance_changepoints(signal, threshold, min_segment):
    """Variance-ratio change-point detection.

    Scans through the signal and finds points where the variance
    ratio between left and right segments is maximized.
    """
    n = len(signal)
    if threshold is None:
        threshold = 2.0  # variance ratio threshold

    changepoints = []
    _recursive_variance_split(signal, 0, n, threshold, min_segment, changepoints)
    changepoints.sort()
    return changepoints


def _recursive_variance_split(signal, start, end, threshold, min_segment,
                               changepoints, max_depth=5):
    """Recursively split at maximum variance-ratio points."""
    length = end - start
    if length < 2 * min_segment or max_depth <= 0:
        return

    seg = signal[start:end]
    total_var = float(np.var(seg))
    if total_var < 1e-10:
        return

    best_ratio = 0.0
    best_idx = -1

    for i in range(min_segment, length - min_segment):
        left = seg[:i]
        right = seg[i:]
        left_var = float(np.var(left))
        right_var = float(np.var(right))

        # Variance within segments vs total
        pooled_var = (len(left) * left_var + len(right) * right_var) / length
        if pooled_var < 1e-10:
            continue

        # Ratio: how much variance is explained by the split
        ratio = total_var / pooled_var
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_ratio >= threshold and best_idx > 0:
        cp = start + best_idx
        changepoints.append(cp)
        # Recurse on both sides
        _recursive_variance_split(signal, start, cp, threshold, min_segment,
                                   changepoints, max_depth - 1)
        _recursive_variance_split(signal, cp, end, threshold, min_segment,
                                   changepoints, max_depth - 1)


def classify_trajectory(positions, dt=1.0, speed_threshold=None,
                        turn_threshold=45.0):
    """Classify motion states along a trajectory.

    Labels each point as 'stopped', 'moving', or 'turning' based on
    instantaneous speed and direction changes. Useful for analyzing
    cell migration patterns, worm locomotion, or organelle transport.

    Args:
        positions: (N, 2) array of (y, x) positions over time.
        dt: Time between frames (seconds).
        speed_threshold: Speed below which a cell is 'stopped' (units/sec).
            If None, auto-computed as 10% of median speed.
        turn_threshold: Angle change (degrees) above which motion is a 'turn'.

    Returns:
        dict with:
            states: List of state labels ('stopped', 'moving', 'turning').
            speeds: 1D array of instantaneous speeds.
            angles: 1D array of direction angles (degrees).
            angle_changes: 1D array of direction changes between steps.
            fraction_stopped: Fraction of time spent stopped.
            fraction_moving: Fraction of time spent moving straight.
            fraction_turning: Fraction of time spent turning.
            n_stops: Number of stop events (transitions to stopped).
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    n = len(positions)

    if n < 2:
        return {
            'states': ['stopped'] * n,
            'speeds': np.zeros(n),
            'angles': np.zeros(n),
            'angle_changes': np.zeros(n),
            'fraction_stopped': 1.0,
            'fraction_moving': 0.0,
            'fraction_turning': 0.0,
            'n_stops': 0,
        }

    # Compute displacements and speeds
    displacements = np.diff(positions, axis=0)
    speeds = np.sqrt(np.sum(displacements ** 2, axis=1)) / dt
    # Pad to match length
    speeds_full = np.zeros(n)
    speeds_full[1:] = speeds

    # Compute angles
    angles = np.degrees(np.arctan2(displacements[:, 0], displacements[:, 1]))
    angles_full = np.zeros(n)
    angles_full[1:] = angles

    # Compute angle changes
    angle_changes = np.zeros(n)
    for i in range(2, n):
        delta = abs(angles[i - 1] - angles[i - 2])
        if delta > 180:
            delta = 360 - delta
        angle_changes[i] = delta

    # Auto threshold
    if speed_threshold is None:
        median_speed = float(np.median(speeds[speeds > 0])) if np.any(speeds > 0) else 1.0
        speed_threshold = median_speed * 0.1

    # Classify
    states = []
    for i in range(n):
        if speeds_full[i] < speed_threshold:
            states.append('stopped')
        elif angle_changes[i] > turn_threshold:
            states.append('turning')
        else:
            states.append('moving')

    # First frame is always 'stopped' (no velocity info)
    states[0] = 'stopped'

    # Count stop events (transitions to stopped)
    n_stops = 0
    for i in range(1, n):
        if states[i] == 'stopped' and states[i - 1] != 'stopped':
            n_stops += 1

    total = max(n, 1)
    return {
        'states': states,
        'speeds': speeds_full,
        'angles': angles_full,
        'angle_changes': angle_changes,
        'fraction_stopped': states.count('stopped') / total,
        'fraction_moving': states.count('moving') / total,
        'fraction_turning': states.count('turning') / total,
        'n_stops': n_stops,
    }
