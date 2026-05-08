"""Closed-loop stage tracking for following moving targets.

The core workflow for real microscopy: snap → detect → locate target →
compute offset from center → move stage → repeat. Handles target loss
with spiral search, predicts motion with linear velocity extrapolation.

Key functions:
    locate_target      -- snap image, detect objects, match to tracked target
    predict_position   -- linear extrapolation from recent velocity
    center_on_target   -- one tracking step (locate + move)
    track_target       -- full N-step tracking loop with loss recovery
    track_target_mda   -- MDA-native tracking via generator + on_frame
    spiral_search      -- expanding search pattern when target is lost

Design:
    - Works in WORLD coordinates throughout
    - Detection-agnostic: pass any detect_fn(image) → list of (x, y) centroids
    - TrackerState is a plain dict, easy to serialize/inspect
"""

import math
import numpy as np
from useq import MDAEvent

from ..hardware import core as hw
from ..hardware.config import resolve_brightfield_channel


# ---------------------------------------------------------------------------
# Tracker state
# ---------------------------------------------------------------------------

def make_tracker(target_world_x, target_world_y):
    """Create a new tracker state dict.

    Args:
        target_world_x, target_world_y: Initial world position of target.

    Returns:
        dict with position, velocity, history, and status fields.
    """
    return {
        'x': float(target_world_x),
        'y': float(target_world_y),
        'vx': 0.0,
        'vy': 0.0,
        'history': [(float(target_world_x), float(target_world_y))],
        'step': 0,
        'lost_count': 0,
        'status': 'tracking',  # 'tracking', 'lost', 'recovered'
    }


# ---------------------------------------------------------------------------
# Motion prediction
# ---------------------------------------------------------------------------

def predict_position(tracker, n_steps=1):
    """Predict target position using linear velocity extrapolation.

    Uses the tracker's current velocity estimate to predict where the
    target will be after n_steps.

    Args:
        tracker: Tracker state dict.
        n_steps: How many steps ahead to predict.

    Returns:
        (predicted_x, predicted_y) tuple.
    """
    px = tracker['x'] + tracker['vx'] * n_steps
    py = tracker['y'] + tracker['vy'] * n_steps
    return float(px), float(py)


def update_velocity(tracker, new_x, new_y, smoothing=0.5):
    """Update velocity estimate with exponential smoothing.

    Args:
        tracker: Tracker state dict (modified in place).
        new_x, new_y: New observed position.
        smoothing: Weight of new measurement (0-1). Higher = more responsive.
    """
    dx = new_x - tracker['x']
    dy = new_y - tracker['y']
    tracker['vx'] = smoothing * dx + (1 - smoothing) * tracker['vx']
    tracker['vy'] = smoothing * dy + (1 - smoothing) * tracker['vy']


# ---------------------------------------------------------------------------
# Core tracking operations
# ---------------------------------------------------------------------------

def locate_target(core, tracker, channel=None,
                  detect_fn=None, max_match_dist=100):
    """Snap an image and find the tracked target.

    Snaps at current stage position, detects objects, matches the closest
    detection to the predicted target position.

    Args:
        core: Microscope core.
        tracker: Tracker state dict.
        channel: Channel to snap.
        detect_fn: Function(image) → list of (x_px, y_px) pixel centroids.
                   If None, uses default threshold detection.
        max_match_dist: Maximum world-coordinate distance to accept a match.

    Returns:
        dict with:
            found: bool — whether target was located
            world_x, world_y: matched world position (or None)
            pixel_x, pixel_y: pixel position in current image (or None)
            image: the snapped image
            n_detections: how many objects were detected
            match_dist: distance to matched target (or None)
    """
    channel = resolve_brightfield_channel(core, channel) or 'brightfield'
    img = hw.snap(core, channel=channel)
    sx, sy = hw.get_position(core)

    # Detect
    if detect_fn is not None:
        centroids_px = detect_fn(img)
    else:
        centroids_px = _default_detect(img)

    if not centroids_px:
        return {
            'found': False, 'world_x': None, 'world_y': None,
            'pixel_x': None, 'pixel_y': None,
            'image': img, 'n_detections': 0, 'match_dist': None,
        }

    # Convert to world coords (prefer core-based config discovery)
    world_pts = []
    for px, py in centroids_px:
        wx, wy = hw.pixel_to_world(px, py, sx, sy, core=core)
        world_pts.append((wx, wy))

    # Predict where target should be
    pred_x, pred_y = predict_position(tracker)

    # Find closest detection to prediction
    best_idx, best_dist = -1, float('inf')
    for i, (wx, wy) in enumerate(world_pts):
        d = math.hypot(wx - pred_x, wy - pred_y)
        if d < best_dist:
            best_dist = d
            best_idx = i

    if best_dist > max_match_dist:
        return {
            'found': False, 'world_x': None, 'world_y': None,
            'pixel_x': None, 'pixel_y': None,
            'image': img, 'n_detections': len(centroids_px),
            'match_dist': round(best_dist, 2),
        }

    wx, wy = world_pts[best_idx]
    px_x, px_y = centroids_px[best_idx]

    return {
        'found': True,
        'world_x': wx, 'world_y': wy,
        'pixel_x': px_x, 'pixel_y': px_y,
        'image': img,
        'n_detections': len(centroids_px),
        'match_dist': round(best_dist, 2),
    }


def center_on_target(core, tracker, channel=None,
                     detect_fn=None, max_match_dist=100,
                     velocity_smoothing=0.5):
    """One tracking step: locate target, update state, move stage to re-center.

    Args:
        core: Microscope core.
        tracker: Tracker state dict (modified in place).
        channel: Channel to snap.
        detect_fn: Detection function (see locate_target).
        max_match_dist: Max distance to accept match.
        velocity_smoothing: Velocity EMA smoothing factor.

    Returns:
        dict with locate_target result plus 'moved' bool.
    """
    result = locate_target(core, tracker, channel=channel,
                           detect_fn=detect_fn,
                           max_match_dist=max_match_dist)

    if result['found']:
        new_x, new_y = result['world_x'], result['world_y']
        update_velocity(tracker, new_x, new_y, smoothing=velocity_smoothing)
        tracker['x'] = new_x
        tracker['y'] = new_y
        tracker['history'].append((new_x, new_y))
        tracker['step'] += 1
        tracker['lost_count'] = 0
        tracker['status'] = 'tracking'

        # Move stage to center on target
        hw.move_to(core, new_x, new_y)
        result['moved'] = True
    else:
        tracker['lost_count'] += 1
        tracker['status'] = 'lost'
        result['moved'] = False

    return result


def spiral_search(core, tracker, channel=None,
                  detect_fn=None, step_size=None, max_rings=3):
    """Search for lost target in expanding spiral pattern.

    Moves stage in a spiral around the last known (or predicted) position.
    Each ring expands by step_size (default: half FOV). At each search
    position, accepts the closest detection to the FOV center (any detection
    within half-FOV of center is a candidate — unlike normal tracking which
    matches against the predicted position).

    Args:
        core: Microscope core.
        tracker: Tracker state dict (modified if target found).
        channel: Channel to snap.
        detect_fn: Detection function.
        step_size: Distance between spiral arms (world px). Default: FOV/2.
        max_rings: Number of spiral rings to search.

    Returns:
        dict with:
            found: bool
            world_x, world_y: found position (or None)
            n_positions_searched: how many positions were checked
            ring_found: which ring (0=center, 1=first ring, etc.) or None
    """
    fov = hw.fov_size(core)
    if step_size is None:
        step_size = fov / 2

    # Start from predicted position
    pred_x, pred_y = predict_position(tracker)
    positions_searched = 0

    # Search positions: center first, then expanding rings
    search_positions = [(pred_x, pred_y, 0)]
    for ring in range(1, max_rings + 1):
        for dx, dy in _ring_offsets(ring, step_size):
            search_positions.append((pred_x + dx, pred_y + dy, ring))

    for sx, sy, ring in search_positions:
        hw.move_to(core, sx, sy)
        # Use a temporary tracker centered on search position so match
        # distance is relative to FOV center, not the old prediction
        search_tracker = make_tracker(sx, sy)
        result = locate_target(core, search_tracker, channel=channel,
                               detect_fn=detect_fn,
                               max_match_dist=fov / 2)
        positions_searched += 1

        if result['found']:
            _apply_found(tracker, result, core=core)
            return {
                'found': True,
                'world_x': result['world_x'],
                'world_y': result['world_y'],
                'n_positions_searched': positions_searched,
                'ring_found': ring,
            }

    return {
        'found': False, 'world_x': None, 'world_y': None,
        'n_positions_searched': positions_searched, 'ring_found': None,
    }


# ---------------------------------------------------------------------------
# Full tracking loop
# ---------------------------------------------------------------------------

def track_target(core, tracker, n_steps=10, channel=None,
                 detect_fn=None, max_match_dist=100,
                 velocity_smoothing=0.5, on_step=None,
                 search_on_loss=True, max_search_rings=3,
                 advance_fn=None):
    """Run N steps of closed-loop target tracking.

    At each step:
    1. (Optional) advance_fn() to advance simulation/time
    2. Snap image, detect, match target
    3. Update velocity estimate
    4. Move stage to re-center target
    5. If lost, optionally spiral search

    Args:
        core: Microscope core.
        tracker: Tracker state dict (modified in place).
        n_steps: Number of tracking steps.
        channel: Channel to snap.
        detect_fn: Detection function (see locate_target).
        max_match_dist: Max distance for target matching.
        velocity_smoothing: EMA smoothing for velocity.
        on_step: Optional callback(step, tracker, result) called after each step.
        search_on_loss: If True, run spiral_search when target is lost.
        max_search_rings: Max rings for spiral search.
        advance_fn: Optional function() called before each step to advance time.

    Returns:
        dict with:
            trajectory: list of (x, y) world positions
            steps_completed: number of successful tracking steps
            steps_lost: number of steps where target was lost
            total_distance: total path length of trajectory
            mean_step_size: average distance per step
            final_status: 'tracking' or 'lost'
            velocity: (vx, vy) final velocity estimate
    """
    steps_lost = 0

    for step in range(n_steps):
        if advance_fn is not None:
            advance_fn()

        result = center_on_target(
            core, tracker, channel=channel, detect_fn=detect_fn,
            max_match_dist=max_match_dist,
            velocity_smoothing=velocity_smoothing,
        )

        if not result['found'] and search_on_loss:
            search = spiral_search(
                core, tracker, channel=channel, detect_fn=detect_fn,
                max_rings=max_search_rings,
            )
            if search['found']:
                tracker['status'] = 'recovered'
            else:
                steps_lost += 1

        elif not result['found']:
            steps_lost += 1
            # Use prediction to keep moving
            pred_x, pred_y = predict_position(tracker)
            tracker['x'] = pred_x
            tracker['y'] = pred_y
            tracker['history'].append((pred_x, pred_y))
            tracker['step'] += 1
            hw.move_to(core, pred_x, pred_y)

        if on_step is not None:
            on_step(step, tracker, result)

    # Compute trajectory stats
    traj = tracker['history']
    total_dist = 0.0
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i - 1][0]
        dy = traj[i][1] - traj[i - 1][1]
        total_dist += math.hypot(dx, dy)

    n_steps_actual = max(len(traj) - 1, 1)

    return {
        'trajectory': list(traj),
        'steps_completed': len(traj) - 1,
        'steps_lost': steps_lost,
        'total_distance': round(total_dist, 2),
        'mean_step_size': round(total_dist / n_steps_actual, 2),
        'final_status': tracker['status'],
        'velocity': (round(tracker['vx'], 3), round(tracker['vy'], 3)),
    }


# ---------------------------------------------------------------------------
# MDA-native tracking
# ---------------------------------------------------------------------------

def track_target_mda(initial_x, initial_y, n_steps=10,
                     channel=None, exposure=50.0,
                     detect_fn=None, max_match_dist=100,
                     velocity_smoothing=0.5, pixel_size=1.0,
                     image_center=256):
    """MDA-native closed-loop tracking via generator + on_frame.

    Returns (event_generator, on_frame, state) for use with run_events().
    The generator yields MDAEvents with dynamic x/y positions based on
    detection feedback from the on_frame callback.

    Execution order per step (guaranteed by run_events):
        1. Generator yields MDAEvent with x_pos, y_pos
        2. run_events moves stage, snaps image
        3. on_frame called with (image, event) — updates tracker state
        4. Generator resumes, reads updated tracker, yields next event

    Args:
        initial_x, initial_y: Starting world position of target.
        n_steps: Number of tracking steps.
        channel: Channel config name.
        exposure: Exposure time in ms.
        detect_fn: callable(image) → list of (x_px, y_px) pixel centroids.
            If None, uses default threshold detection.
        max_match_dist: Maximum world-coord distance to accept a match.
        velocity_smoothing: EMA smoothing for velocity estimate.
        pixel_size: Physical pixel size (um/px). Default 1.0 (10x).
        image_center: Image center in pixels. Default 256 (512px sensor).

    Returns:
        Tuple of (event_generator_fn, on_frame_fn, shared_state_dict).

    Usage::

        gen, on_frame, state = track_target_mda(500, 800, n_steps=60)
        results = run_events(core, gen(), on_frame=on_frame)
        trajectory = state['tracker']['history']
    """
    tracker = make_tracker(initial_x, initial_y)

    state = {
        'tracker': tracker,
        'steps_lost': 0,
        'images': [],
    }

    def on_frame(image, event, meta=None):
        """Detect target in image and update tracker state."""
        t = tracker  # alias
        sx = event.x_pos if event.x_pos is not None else t['x']
        sy = event.y_pos if event.y_pos is not None else t['y']

        # Detect objects
        if detect_fn is not None:
            centroids_px = detect_fn(image)
        else:
            centroids_px = _default_detect(image)

        if not centroids_px:
            t['lost_count'] += 1
            t['status'] = 'lost'
            state['steps_lost'] += 1
            # Use velocity prediction
            pred_x, pred_y = predict_position(t)
            t['x'] = pred_x
            t['y'] = pred_y
            t['history'].append((pred_x, pred_y))
            t['step'] += 1
            state['images'].append(image)
            return

        # Convert pixel centroids to world coordinates
        center = image_center
        world_pts = []
        for px, py in centroids_px:
            wx = sx + (px - center) * pixel_size
            wy = sy + (py - center) * pixel_size
            world_pts.append((wx, wy))

        # Match closest to prediction
        pred_x, pred_y = predict_position(t)
        best_idx, best_dist = -1, float('inf')
        for i, (wx, wy) in enumerate(world_pts):
            d = math.hypot(wx - pred_x, wy - pred_y)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_dist > max_match_dist:
            t['lost_count'] += 1
            t['status'] = 'lost'
            state['steps_lost'] += 1
            pred_x, pred_y = predict_position(t)
            t['x'] = pred_x
            t['y'] = pred_y
            t['history'].append((pred_x, pred_y))
        else:
            new_x, new_y = world_pts[best_idx]
            update_velocity(t, new_x, new_y, smoothing=velocity_smoothing)
            t['x'] = new_x
            t['y'] = new_y
            t['history'].append((new_x, new_y))
            t['lost_count'] = 0
            t['status'] = 'tracking'

        t['step'] += 1
        state['images'].append(image)

    def event_generator():
        """Yield tracking MDAEvents with dynamic positions."""
        for step in range(n_steps):
            x, y = tracker['x'], tracker['y']
            yield MDAEvent(
                x_pos=float(x),
                y_pos=float(y),
                exposure=exposure,
                channel={'config': channel} if channel else None,
                index={'t': step},
            )

    return event_generator, on_frame, state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_detect(image, threshold_sigma=2.5, min_area_px=30):
    """Default detection: threshold-based, returns list of (cx, cy) pixel coords."""
    from ..detection.cells import detect_cells
    cells = detect_cells(image, threshold_sigma=threshold_sigma,
                         min_area_px=min_area_px, fill_holes=True)
    return [(c['centroid_px'][0], c['centroid_px'][1]) for c in cells]


def _ring_offsets(ring, step_size):
    """Generate (dx, dy) offsets for a spiral ring.

    Ring 1 = 8 positions (3x3 minus center), ring 2 = 16 positions, etc.
    """
    offsets = []
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            if abs(dx) == ring or abs(dy) == ring:  # only perimeter
                offsets.append((dx * step_size, dy * step_size))
    return offsets


def _apply_found(tracker, locate_result, core=None):
    """Update tracker when target is found during search."""
    new_x, new_y = locate_result['world_x'], locate_result['world_y']
    update_velocity(tracker, new_x, new_y, smoothing=0.3)
    tracker['x'] = new_x
    tracker['y'] = new_y
    tracker['history'].append((new_x, new_y))
    tracker['step'] += 1
    tracker['lost_count'] = 0
    tracker['status'] = 'recovered'
    if core is not None:
        hw.move_to(core, new_x, new_y)


# ---------------------------------------------------------------------------
# Multi-target tracking
# ---------------------------------------------------------------------------

def track_multiple(core, targets, n_rounds=5, channel=None,
                   detect_fn=None, max_match_dist=100,
                   velocity_smoothing=0.5, advance_fn=None):
    """Track multiple targets in round-robin fashion.

    At each round, visits each target's predicted position, snaps,
    detects, and updates. Useful for monitoring multiple cells/organisms
    spread across the sample.

    Args:
        core: Microscope core.
        targets: List of tracker dicts (from make_tracker).
        n_rounds: Number of complete rounds (each round visits all targets).
        channel: Channel to snap.
        detect_fn: Detection function.
        max_match_dist: Max match distance.
        velocity_smoothing: EMA smoothing.
        advance_fn: Optional function() called before each snap.

    Returns:
        Dict with:
            trackers: the updated tracker list
            per_target: list of per-target summary dicts
            total_snaps: total number of snaps taken
    """
    total_snaps = 0

    for round_idx in range(n_rounds):
        for t_idx, tracker in enumerate(targets):
            if advance_fn is not None:
                advance_fn()

            result = center_on_target(
                core, tracker, channel=channel, detect_fn=detect_fn,
                max_match_dist=max_match_dist,
                velocity_smoothing=velocity_smoothing,
            )
            total_snaps += 1

    # Summarize
    per_target = []
    for tracker in targets:
        traj = tracker['history']
        total_dist = 0.0
        for i in range(1, len(traj)):
            dx = traj[i][0] - traj[i - 1][0]
            dy = traj[i][1] - traj[i - 1][1]
            total_dist += math.hypot(dx, dy)

        per_target.append({
            'trajectory': list(traj),
            'total_distance': round(total_dist, 2),
            'steps': len(traj) - 1,
            'lost_count': tracker['lost_count'],
            'status': tracker['status'],
        })

    return {
        'trackers': targets,
        'per_target': per_target,
        'total_snaps': total_snaps,
    }
