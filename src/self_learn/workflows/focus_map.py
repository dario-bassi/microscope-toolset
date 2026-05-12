"""Focus map for multi-position Z calibration.

Calibrates best Z-focus at a grid of reference stage positions, then
interpolates to predict the optimal Z at arbitrary (x, y) locations.
Avoids running full autofocus at every acquisition position.

On real microscopes, the sample is rarely perfectly flat — there's always
some tilt or curvature across the stage. Running autofocus at every position
is slow and wastes photons. A focus map measures Z at a sparse grid and
fits a surface, reducing autofocus calls by 5-10x.

Key functions:
    build_focus_map       -- Calibrate Z at reference positions (imperative)
    predict_z             -- Interpolate Z for any (x, y) from a focus map
    apply_focus_corrections -- Apply Z corrections to MDA events
    focus_map_events      -- MDA-native focus calibration
    refine_focus_map      -- Add new calibration points to existing map

Design:
    - Uses scipy.interpolate for 2D surface fitting
    - Falls back to planar fit when <4 calibration points
    - Compatible with any autofocus function (coarse_fine_focus, sweep_focus)
    - FocusMap is a plain dict — easy to serialize/inspect/resume
"""

import numpy as np


# ---------------------------------------------------------------------------
# Focus map data structure
# ---------------------------------------------------------------------------

def make_focus_map(calibration_points=None):
    """Create a new focus map state dict.

    Args:
        calibration_points: Optional list of (x, y, z) tuples from
            prior calibration. If provided, the surface is fit immediately.

    Returns:
        dict with calibration data and surface fit parameters.
    """
    fm = {
        'points': [],       # list of (x, y, best_z, score) tuples
        'fit_type': None,   # 'planar', 'griddata', or None
        'fit_params': None, # planar coefficients or None
        'residual_rms': 0.0,
    }
    if calibration_points:
        for p in calibration_points:
            x, y, z = p[0], p[1], p[2]
            score = p[3] if len(p) > 3 else 0.0
            fm['points'].append((float(x), float(y), float(z), float(score)))
        _fit_surface(fm)
    return fm


def _fit_surface(fm):
    """Fit a Z surface to the calibration points.

    Uses planar fit (ax + by + c) when possible, with fallback to
    griddata interpolation for non-planar surfaces.
    """
    pts = fm['points']
    n = len(pts)
    if n == 0:
        fm['fit_type'] = None
        fm['fit_params'] = None
        return

    if n == 1:
        # Single point: constant Z
        fm['fit_type'] = 'constant'
        fm['fit_params'] = {'z': pts[0][2]}
        fm['residual_rms'] = 0.0
        return

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    zs = np.array([p[2] for p in pts])

    if n <= 3:
        # Fit a plane: z = ax + by + c
        A = np.column_stack([xs, ys, np.ones(n)])
        # Use least-squares (works even for n=2 with underdetermined system)
        result = np.linalg.lstsq(A, zs, rcond=None)
        coeffs = result[0]
        fm['fit_type'] = 'planar'
        fm['fit_params'] = {'a': float(coeffs[0]), 'b': float(coeffs[1]),
                            'c': float(coeffs[2])}
        predicted = A @ coeffs
        fm['residual_rms'] = float(np.sqrt(np.mean((zs - predicted) ** 2)))
        return

    # 4+ points: fit plane first, check residuals
    A = np.column_stack([xs, ys, np.ones(n)])
    result = np.linalg.lstsq(A, zs, rcond=None)
    coeffs = result[0]
    predicted = A @ coeffs
    rms = float(np.sqrt(np.mean((zs - predicted) ** 2)))

    # If planar fit is good enough (RMS < 1.0 µm), use it
    if rms < 1.0:
        fm['fit_type'] = 'planar'
        fm['fit_params'] = {'a': float(coeffs[0]), 'b': float(coeffs[1]),
                            'c': float(coeffs[2])}
        fm['residual_rms'] = rms
    else:
        # Non-planar: store for griddata interpolation
        fm['fit_type'] = 'griddata'
        fm['fit_params'] = {'planar_fallback': {
            'a': float(coeffs[0]), 'b': float(coeffs[1]),
            'c': float(coeffs[2]),
        }}
        fm['residual_rms'] = rms


def predict_z(fm, x, y):
    """Predict best Z position for a given (x, y) stage coordinate.

    Args:
        fm: Focus map dict from build_focus_map or make_focus_map.
        x, y: Stage coordinates.

    Returns:
        float: Predicted best Z position.

    Raises:
        ValueError: If focus map has no calibration points.
    """
    if fm['fit_type'] is None:
        raise ValueError("Focus map has no calibration points.")

    if fm['fit_type'] == 'constant':
        return fm['fit_params']['z']

    if fm['fit_type'] == 'planar':
        p = fm['fit_params']
        return p['a'] * x + p['b'] * y + p['c']

    if fm['fit_type'] == 'griddata':
        # Use scipy griddata for non-planar interpolation
        from scipy.interpolate import griddata
        pts = fm['points']
        xy = np.array([[p[0], p[1]] for p in pts])
        zs = np.array([p[2] for p in pts])
        z = griddata(xy, zs, [[x, y]], method='cubic')
        if np.isnan(z[0]):
            # Fall back to linear, then nearest
            z = griddata(xy, zs, [[x, y]], method='linear')
        if np.isnan(z[0]):
            z = griddata(xy, zs, [[x, y]], method='nearest')
        if np.isnan(z[0]):
            # Final fallback: planar
            p = fm['fit_params']['planar_fallback']
            return p['a'] * x + p['b'] * y + p['c']
        return float(z[0])

    raise ValueError(f"Unknown fit_type: {fm['fit_type']}")


# ---------------------------------------------------------------------------
# Imperative API
# ---------------------------------------------------------------------------

def build_focus_map(core, positions, z_range=20.0, coarse_step=2.0,
                    fine_step=0.5, channel=None, method='brenner'):
    """Build a focus map by running autofocus at each reference position.

    Args:
        core: Microscope core.
        positions: List of (x, y) stage positions for calibration.
        z_range: Half-range for autofocus Z sweep.
        coarse_step: Coarse pass step size.
        fine_step: Fine pass step size.
        channel: Channel to use for focus measurement.
        method: Focus metric method.

    Returns:
        dict: Focus map with calibration data and surface fit.
    """
    from ..hardware.core import move_to, set_z
    from .autofocus import coarse_fine_focus

    fm = make_focus_map()

    for x, y in positions:
        move_to(core, x, y)
        result = coarse_fine_focus(
            core, z_range=z_range, coarse_step=coarse_step,
            fine_step=fine_step, channel=channel, method=method,
        )
        best_z = result['best_z']
        best_score = result['best_score']
        fm['points'].append((float(x), float(y), float(best_z),
                             float(best_score)))

    _fit_surface(fm)
    return fm


def refine_focus_map(fm, x, y, z, score=0.0):
    """Add a new calibration point and refit the surface.

    Use this to incrementally improve the focus map as you visit new
    positions during an experiment.

    Args:
        fm: Existing focus map dict (modified in place).
        x, y: Stage position.
        z: Measured best Z.
        score: Focus quality score.
    """
    fm['points'].append((float(x), float(y), float(z), float(score)))
    _fit_surface(fm)


# ---------------------------------------------------------------------------
# Applying focus corrections to MDA events
# ---------------------------------------------------------------------------

def apply_focus_corrections(events, fm):
    """Apply Z corrections from a focus map to MDA events.

    For each event with x_pos and y_pos, predicts the best Z and sets
    z_pos accordingly. Events without position info are passed through.

    Args:
        events: Iterable of MDAEvent objects.
        fm: Focus map dict.

    Yields:
        MDAEvent objects with z_pos set from focus map prediction.
    """
    from useq import MDAEvent

    for event in events:
        x = event.x_pos
        y = event.y_pos
        if x is not None and y is not None and fm['fit_type'] is not None:
            z = predict_z(fm, x, y)
            # Replace z_pos while preserving all other fields
            event = event.model_copy(update={'z_pos': z})
        yield event


# ---------------------------------------------------------------------------
# MDA-native focus calibration
# ---------------------------------------------------------------------------

def focus_map_events(positions, z_range=20.0, n_coarse=11, n_fine=11,
                     method='brenner', exposure=50.0, channel=None):
    """MDA-native focus map calibration.

    Yields autofocus Z-sweep events for each calibration position.
    The on_frame callback measures focus quality and the generator
    selects the best Z after each position's sweep.

    After all positions are calibrated, the focus map surface is fit
    and stored in shared_state['focus_map'].

    Usage::

        positions = [(0, 0), (500, 0), (0, 500), (500, 500)]
        gen, on_frame, state = focus_map_events(positions)
        results = run_events(core, gen(), on_frame=on_frame)
        fm = state['focus_map']
        z = predict_z(fm, 250, 250)  # interpolated Z

    Args:
        positions: List of (x, y) reference positions.
        z_range: Half-range for Z sweep at each position.
        n_coarse: Number of Z steps in coarse pass.
        n_fine: Number of Z steps in fine pass.
        method: Focus metric method.
        exposure: Exposure time in ms.
        channel: Channel config name.

    Returns:
        Tuple of (event_generator_factory, on_frame_callback, shared_state).
    """
    from useq import MDAEvent
    from .autofocus import focus_metric

    coarse_zs = np.linspace(-z_range, z_range, n_coarse)

    state = {
        'focus_map': make_focus_map(),
        'current_pos_idx': 0,
        'current_phase': 'coarse',
        'current_scores': [],
        'current_best_z': 0.0,
        'current_best_score': 0.0,
        'method': method,
        'positions_completed': 0,
    }

    def on_frame(image, event, meta=None):
        z = event.z_pos if event.z_pos is not None else 0.0
        score = focus_metric(image, method=state['method'])
        state['current_scores'].append((z, score))

        if score > state['current_best_score']:
            state['current_best_score'] = score
            state['current_best_z'] = z

    def event_generator():
        for pos_idx, (x, y) in enumerate(positions):
            state['current_pos_idx'] = pos_idx

            # Coarse pass
            state['current_phase'] = 'coarse'
            state['current_scores'] = []
            state['current_best_z'] = 0.0
            state['current_best_score'] = 0.0

            for i, z in enumerate(coarse_zs):
                yield MDAEvent(
                    x_pos=float(x), y_pos=float(y), z_pos=float(z),
                    exposure=exposure,
                    channel={'config': channel} if channel else None,
                    metadata={'focus_map': True, 'phase': 'coarse',
                              'pos_idx': pos_idx, 'z_idx': i},
                )

            # Fine pass around coarse best
            coarse_best = state['current_best_z']
            fine_half = (2 * z_range / max(n_coarse - 1, 1)) * 1.5
            fine_zs = np.linspace(coarse_best - fine_half,
                                  coarse_best + fine_half, n_fine)

            state['current_phase'] = 'fine'
            state['current_best_score'] = 0.0
            state['current_best_z'] = coarse_best
            state['current_scores'] = []

            for i, z in enumerate(fine_zs):
                yield MDAEvent(
                    x_pos=float(x), y_pos=float(y), z_pos=float(z),
                    exposure=exposure,
                    channel={'config': channel} if channel else None,
                    metadata={'focus_map': True, 'phase': 'fine',
                              'pos_idx': pos_idx, 'z_idx': i},
                )

            # Record calibration point
            best_z = state['current_best_z']
            best_score = state['current_best_score']
            state['focus_map']['points'].append(
                (float(x), float(y), float(best_z), float(best_score))
            )
            state['positions_completed'] = pos_idx + 1

        # All positions calibrated — fit surface
        _fit_surface(state['focus_map'])

    return event_generator, on_frame, state
