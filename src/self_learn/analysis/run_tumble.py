"""Run-and-tumble motility analysis for bacteria and other microswimmers.

Bacteria exhibit alternating runs (straight swimming at constant speed)
and tumbles (brief reorientation events). This module classifies trajectory
segments into runs and tumbles and computes motility statistics.

Key functions:
    step_speeds           -- Per-step displacement speeds from trajectory
    classify_run_tumble   -- Classify each step as run or tumble
    speed_valley          -- Find natural threshold from bimodal distribution
    extract_run_segments  -- Extract contiguous run segments
    analyze_run_tumble    -- Full run-and-tumble motility analysis
    build_tracks          -- Build multi-frame tracks from per-frame detections

Usage::

    # Build tracks from consecutive frame detections
    tracks = build_tracks(all_positions)  # all_positions[i] = Nx2 array

    # Analyze run-and-tumble for each track
    all_speeds = []
    for positions in tracks:
        result = analyze_run_tumble(positions, dt=0.2, pixel_size=0.5)
        all_speeds.extend(result['run_speeds'])

    mean_run_speed = np.mean(all_speeds)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Track building
# ---------------------------------------------------------------------------

def build_tracks(all_positions, max_dist=20, min_length=10, max_gap=0):
    """Build multi-frame tracks by Hungarian matching between consecutive frames.

    Args:
        all_positions: list of Nx2 arrays (row, col) per frame.
            Some frames may have no detections (empty arrays).
        max_dist: Maximum displacement in pixels between frames.
        min_length: Minimum track length to keep (frames).
        max_gap: Maximum number of consecutive frames a track can survive
            without a detection before being terminated. 0 = no gap tolerance
            (original behavior). 1-2 = allow brief detection dropouts.
            During gaps, the track's last known position is used for matching.

    Returns:
        list of dicts, each with:
            'positions': Nx2 array of (row, col) per frame
            'frame_indices': array of frame indices for each position
    """
    tracks_raw = {}  # track_id -> {'positions': [], 'frames': []}
    active = {}      # track_id -> last position
    gap_count = {}   # track_id -> frames since last detection
    next_id = 0

    for fi, pos in enumerate(all_positions):
        if len(pos) == 0:
            if max_gap > 0:
                # Increment gap counter; drop tracks exceeding max_gap
                lost = []
                for tid in active:
                    gap_count[tid] = gap_count.get(tid, 0) + 1
                    if gap_count[tid] > max_gap:
                        lost.append(tid)
                for tid in lost:
                    del active[tid]
                    del gap_count[tid]
            else:
                active = {}
            continue

        if not active:
            for p in pos:
                tracks_raw[next_id] = {'positions': [p.copy()], 'frames': [fi]}
                active[next_id] = p
                gap_count[next_id] = 0
                next_id += 1
            continue

        # Hungarian matching — scale max_dist by gap for tracks with gaps
        active_ids = list(active.keys())
        apos = np.array([active[i] for i in active_ids])

        cost = np.sqrt(((apos[:, None, :] - pos[None, :, :])**2).sum(axis=2))

        # Per-track max_dist: scale by (1 + gap_frames) to allow larger jumps
        # for tracks that missed detections
        per_track_max = np.array([
            max_dist * (1 + gap_count.get(active_ids[i], 0))
            for i in range(len(active_ids))
        ])

        r_a, r_p = linear_sum_assignment(cost)
        valid = cost[r_a, r_p] < per_track_max[r_a]
        r_a, r_p = r_a[valid], r_p[valid]

        matched_a = set(r_a.tolist())
        matched_p = set(r_p.tolist())

        for i, j in zip(r_a, r_p):
            tid = active_ids[i]
            tracks_raw[tid]['positions'].append(pos[j].copy())
            tracks_raw[tid]['frames'].append(fi)
            active[tid] = pos[j]
            gap_count[tid] = 0

        # Unmatched active tracks: increment gap or remove
        for i in range(len(active_ids)):
            if i not in matched_a:
                tid = active_ids[i]
                gap_count[tid] = gap_count.get(tid, 0) + 1
                if gap_count[tid] > max_gap:
                    del active[tid]
                    del gap_count[tid]

        # New tracks from unmatched detections
        for j in range(len(pos)):
            if j not in matched_p:
                tracks_raw[next_id] = {'positions': [pos[j].copy()], 'frames': [fi]}
                active[next_id] = pos[j]
                gap_count[next_id] = 0
                next_id += 1

    # Filter by minimum length and convert to arrays
    result = []
    for t in tracks_raw.values():
        if len(t['positions']) >= min_length:
            result.append({
                'positions': np.array(t['positions']),
                'frame_indices': np.array(t['frames']),
            })
    return result


# ---------------------------------------------------------------------------
# Speed computation
# ---------------------------------------------------------------------------

def step_speeds(positions, frame_indices=None, dt=1.0, pixel_size=1.0):
    """Compute per-step displacement speeds from a trajectory.

    Args:
        positions: Nx2 array of (row, col) positions.
        frame_indices: Optional array of frame indices. If provided, only
            consecutive frames (diff=1) are used.
        dt: Time step between frames (seconds).
        pixel_size: µm per pixel.

    Returns:
        1D array of speeds in µm/s. Length N-1 (or fewer if gaps).
    """
    rows = np.array(positions[:, 0])
    cols = np.array(positions[:, 1])
    n = len(rows)

    speeds = []
    for i in range(n - 1):
        # Skip gaps
        if frame_indices is not None:
            if frame_indices[i+1] - frame_indices[i] != 1:
                continue
        dr = rows[i+1] - rows[i]
        dc = cols[i+1] - cols[i]
        speed = float(np.sqrt(dr**2 + dc**2)) * pixel_size / dt
        speeds.append(speed)

    return np.array(speeds)


# ---------------------------------------------------------------------------
# Bimodal classification
# ---------------------------------------------------------------------------

def speed_valley(speeds, n_bins=30, smoothing=2):
    """Find valley between tumble (slow) and run (fast) speed peaks.

    Uses smoothed histogram to find the trough between the two modes.

    Args:
        speeds: 1D array of speeds.
        n_bins: Number of histogram bins.
        smoothing: Gaussian smoothing sigma for histogram.

    Returns:
        float: Valley threshold speed separating runs from tumbles.
    """
    from scipy.ndimage import gaussian_filter1d

    if len(speeds) == 0:
        return 0.0

    hist, edges = np.histogram(speeds, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(hist.astype(float), sigma=smoothing)

    # Find valley: local minimum between first and last peak
    # Look between 10th and 90th percentile of speed range
    v_lo = int(0.1 * n_bins)
    v_hi = int(0.7 * n_bins)

    valley_idx = v_lo + int(np.argmin(smooth[v_lo:v_hi]))
    return float(centers[valley_idx])


def classify_run_tumble(positions, frame_indices=None, dt=1.0, pixel_size=1.0,
                        angle_threshold=90.0, valley_threshold=None,
                        min_tumble_frames=2):
    """Classify each trajectory step as run or tumble.

    A step is initially flagged as tumble-candidate if:
    - Speed is below the speed valley threshold, OR
    - Direction change from previous step exceeds angle_threshold

    Then, if min_tumble_frames > 1, isolated single-frame tumble candidates
    surrounded by runs are reclassified as runs. This prevents centroid
    noise from creating false tumble events.

    For E. coli: mean tumble duration ~0.1s, mean run duration ~1s,
    so ~91% of time is spent running. Angle threshold of 90° works well
    because real tumbles reorient by ~68° on average but with high variance.

    Args:
        positions: Nx2 array of (row, col) positions.
        frame_indices: Optional array of frame indices (for gap detection).
        dt: Time step between frames (seconds).
        pixel_size: µm per pixel.
        angle_threshold: Maximum direction change (degrees) in a run.
            Default 90° suitable for E. coli (real tumbles avg ~68°).
        valley_threshold: Speed threshold. If None, computed automatically.
        min_tumble_frames: Minimum consecutive tumble-candidate frames to
            confirm a tumble event. Set to 1 for original behavior (no
            filtering). Set to 2 to suppress single-frame noise.

    Returns:
        dict with:
            labels: array of 'run' or 'tumble' per step
            speeds: per-step speeds in µm/s
            angles: per-step direction changes in degrees
            valley_threshold: speed threshold used
    """
    rows = positions[:, 0]
    cols = positions[:, 1]
    n = len(rows)

    speeds = []
    angles = []
    step_indices = []  # which consecutive pairs are used

    for i in range(n - 1):
        if frame_indices is not None and frame_indices[i+1] - frame_indices[i] != 1:
            continue
        dr = rows[i+1] - rows[i]
        dc = cols[i+1] - cols[i]
        speed = float(np.sqrt(dr**2 + dc**2)) * pixel_size / dt
        speeds.append(speed)

        if i > 0 and len(step_indices) > 0 and step_indices[-1] == i - 1:
            dr0 = rows[i] - rows[i-1]
            dc0 = cols[i] - cols[i-1]
            v0 = np.sqrt(dr0**2 + dc0**2)
            v1 = np.sqrt(dr**2 + dc**2)
            if v0 > 1e-6 and v1 > 1e-6:
                cos_a = np.clip((dr0*dr + dc0*dc) / (v0*v1), -1, 1)
                angle = float(np.degrees(np.arccos(cos_a)))
            else:
                angle = 0.0
        else:
            angle = 0.0
        angles.append(angle)
        step_indices.append(i)

    speeds = np.array(speeds)
    angles = np.array(angles)

    if valley_threshold is None and len(speeds) > 5:
        valley_threshold = speed_valley(speeds)
    elif valley_threshold is None:
        valley_threshold = float(np.median(speeds)) * 0.3

    # Initial classification
    labels = []
    for s, a in zip(speeds, angles):
        if s < valley_threshold or a > angle_threshold:
            labels.append('tumble')
        else:
            labels.append('run')

    # Filter isolated tumble frames (suppress single-frame noise)
    if min_tumble_frames > 1 and len(labels) > 2:
        labels = _filter_short_tumbles(labels, min_tumble_frames)

    return {
        'labels': labels,
        'speeds': speeds,
        'angles': angles,
        'valley_threshold': float(valley_threshold),
    }


def _filter_short_tumbles(labels, min_frames):
    """Reclassify tumble runs shorter than min_frames as runs.

    This suppresses false tumble events caused by centroid noise.
    Only isolated tumble stretches shorter than min_frames are affected;
    genuine tumble events spanning multiple frames are preserved.
    """
    result = list(labels)
    n = len(result)
    i = 0
    while i < n:
        if result[i] == 'tumble':
            # Find end of this tumble stretch
            j = i
            while j < n and result[j] == 'tumble':
                j += 1
            tumble_len = j - i
            if tumble_len < min_frames:
                # Too short: reclassify as run
                for k in range(i, j):
                    result[k] = 'run'
            i = j
        else:
            i += 1
    return result


# ---------------------------------------------------------------------------
# Run segment extraction
# ---------------------------------------------------------------------------

def extract_run_segments(labels, speeds, min_length=2):
    """Extract contiguous run segments from classified labels.

    Args:
        labels: list of 'run' or 'tumble' strings.
        speeds: corresponding per-step speeds.
        min_length: minimum number of run steps to keep.

    Returns:
        list of dicts, each with:
            'start_idx': index in labels where run starts
            'length': number of steps
            'speeds': speeds during this run
            'mean_speed': mean speed of this run
    """
    segments = []
    in_run = False
    start = 0

    for i, label in enumerate(labels + ['tumble']):  # sentinel
        if label == 'run' and not in_run:
            in_run = True
            start = i
        elif label != 'run' and in_run:
            length = i - start
            if length >= min_length:
                run_speeds = speeds[start:i]
                segments.append({
                    'start_idx': start,
                    'length': length,
                    'speeds': run_speeds,
                    'mean_speed': float(np.mean(run_speeds)),
                })
            in_run = False

    return segments


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_run_tumble(positions, dt=1.0, pixel_size=1.0,
                       frame_indices=None, angle_threshold=90.0,
                       valley_threshold=None, min_tumble_frames=2):
    """Full run-and-tumble motility analysis for a single trajectory.

    Args:
        positions: Nx2 array of (row, col) positions.
        dt: Time step between frames (seconds).
        pixel_size: µm per pixel.
        frame_indices: Optional array of frame indices.
        angle_threshold: Tumble angle threshold (degrees). Default 90°.
        valley_threshold: Speed threshold for run/tumble separation.
        min_tumble_frames: Min consecutive tumble frames to confirm. Default 2.

    Returns:
        dict with:
            mean_run_speed: mean speed during runs (µm/s)
            median_run_speed: median run speed (µm/s)
            mean_all_speed: mean of all per-step speeds
            run_fraction: fraction of steps classified as runs
            tumble_fraction: fraction classified as tumbles
            n_runs: number of distinct run segments
            n_tumbles: number of tumble events
            run_speeds: all per-step speeds during runs
            all_speeds: all per-step speeds
            valley_threshold: speed threshold used
    """
    classified = classify_run_tumble(
        positions, frame_indices=frame_indices, dt=dt,
        pixel_size=pixel_size, angle_threshold=angle_threshold,
        valley_threshold=valley_threshold,
        min_tumble_frames=min_tumble_frames,
    )

    labels = classified['labels']
    speeds = classified['speeds']
    vt = classified['valley_threshold']

    if len(labels) == 0:
        return {
            'mean_run_speed': 0.0,
            'median_run_speed': 0.0,
            'mean_all_speed': 0.0,
            'run_fraction': 0.0,
            'tumble_fraction': 0.0,
            'n_runs': 0,
            'n_tumbles': 0,
            'run_speeds': np.array([]),
            'all_speeds': np.array([]),
            'valley_threshold': vt,
        }

    run_mask = np.array([l == 'run' for l in labels])
    run_speeds = speeds[run_mask] if run_mask.any() else np.array([])

    run_segments = extract_run_segments(labels, speeds)

    n_tumbles = sum(1 for l in labels if l == 'tumble')
    n_runs = len(run_segments)

    return {
        'mean_run_speed': float(np.mean(run_speeds)) if len(run_speeds) > 0 else 0.0,
        'median_run_speed': float(np.median(run_speeds)) if len(run_speeds) > 0 else 0.0,
        'mean_all_speed': float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
        'run_fraction': float(run_mask.mean()) if len(run_mask) > 0 else 0.0,
        'tumble_fraction': float((~run_mask).mean()) if len(run_mask) > 0 else 0.0,
        'n_runs': n_runs,
        'n_tumbles': n_tumbles,
        'run_speeds': run_speeds,
        'all_speeds': speeds,
        'valley_threshold': vt,
    }


def validate_tracks(tracks, n_visible, n_frames=None):
    """Sanity-check track count against visible cell count.

    Detects severe track fragmentation by comparing the number of
    reconstructed tracks with the number of visible objects. In a healthy
    tracking run, ``len(tracks) ≈ n_visible``. If tracks vastly outnumber
    visible cells, the tracker is fragmenting individual trajectories.

    Args:
        tracks: List of track dicts from :func:`build_tracks`.
        n_visible: Number of distinct cells/bacteria visible in the sample.
        n_frames: Total number of frames (optional, for context in warning).

    Returns:
        dict with:
            is_valid: True if track count is plausible.
            n_tracks: Number of tracks.
            expected_max: 3 × n_visible (generous upper bound).
            ratio: n_tracks / n_visible.
            warning: Warning string if invalid, else empty string.
    """
    n_tracks = len(tracks)
    expected_max = max(1, 3 * n_visible)
    ratio = n_tracks / max(n_visible, 1)
    is_valid = n_tracks <= expected_max

    warning = ""
    if not is_valid:
        warning = (
            f"Track fragmentation detected: {n_tracks} tracks from "
            f"{n_visible} visible objects (ratio {ratio:.1f}x). "
            f"Consider increasing max_gap or raising angle_threshold."
        )

    return {
        'is_valid': is_valid,
        'n_tracks': n_tracks,
        'expected_max': expected_max,
        'ratio': round(ratio, 2),
        'warning': warning,
    }


def aggregate_run_tumble(track_results):
    """Aggregate run-and-tumble results across multiple tracks.

    Args:
        track_results: list of dicts from analyze_run_tumble().

    Returns:
        dict with population-level statistics.
    """
    all_run_speeds = np.concatenate([r['run_speeds'] for r in track_results
                                     if len(r['run_speeds']) > 0])
    all_speeds = np.concatenate([r['all_speeds'] for r in track_results
                                 if len(r['all_speeds']) > 0])
    run_fractions = [r['run_fraction'] for r in track_results]
    tumble_fractions = [r['tumble_fraction'] for r in track_results]

    return {
        'population_mean_run_speed': float(np.mean(all_run_speeds)) if len(all_run_speeds) > 0 else 0.0,
        'population_median_run_speed': float(np.median(all_run_speeds)) if len(all_run_speeds) > 0 else 0.0,
        'population_std_run_speed': float(np.std(all_run_speeds)) if len(all_run_speeds) > 0 else 0.0,
        'mean_run_fraction': float(np.mean(run_fractions)) if run_fractions else 0.0,
        'mean_tumble_fraction': float(np.mean(tumble_fractions)) if tumble_fractions else 0.0,
        'total_run_speed_samples': len(all_run_speeds),
        'n_tracks_analyzed': len(track_results),
        'all_run_speeds': all_run_speeds,
        'all_speeds': all_speeds,
    }
