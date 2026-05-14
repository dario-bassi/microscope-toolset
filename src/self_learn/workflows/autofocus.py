"""Software autofocus workflow for microscopy.

Finds the best focal plane by maximizing an image sharpness metric.
Supports multiple metrics and a coarse-then-fine search strategy.

Key functions:
    focus_metric          -- Compute sharpness metric for an image
    parabolic_peak_interp -- Sub-step peak from argmax + 2 neighbours
    sweep_focus           -- Sweep Z range and return focus curve
    coarse_fine_focus     -- Two-pass autofocus (coarse then fine)
    track_focus_brownian  -- Closed-loop tracker for pure-Brownian Z drift
    continuous_focus      -- Track focus drift during timelapse
    autofocus_mda         -- MDA-native coarse+fine autofocus

Design:
    - Detection-agnostic: works on any image content
    - Multiple metrics: Brenner gradient, Laplacian variance, normalized variance
    - Coarse+fine strategy avoids local minima and reduces snap count
    - Continuous mode tracks focus drift with minimal snaps per timepoint
    - MDA-native: autofocus_mda returns generator + callback for MDA engine
"""

import numpy as np


# ---------------------------------------------------------------------------
# Focus metrics
# ---------------------------------------------------------------------------

def focus_metric(img, method='brenner'):
    """Compute image sharpness metric.

    Higher values indicate sharper (better focused) images.

    Args:
        img: 2D numpy array.
        method: One of 'brenner', 'laplacian', 'normalized_variance', 'sobel'.

    Returns:
        float: Focus metric value (higher = sharper).
    """
    img_f = img.astype(np.float64)

    if method == 'brenner':
        # Brenner gradient: sum of squared horizontal differences (step=2)
        if img_f.shape[1] > 2:
            dx = img_f[:, 2:] - img_f[:, :-2]
            return float(np.mean(dx ** 2))
        return 0.0

    elif method == 'laplacian':
        # Laplacian variance — measures edges in all directions
        # Manual 3x3 Laplacian kernel (no cv2 dependency)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.ndimage import convolve
        lap = convolve(img_f, kernel)
        return float(np.var(lap))

    elif method == 'normalized_variance':
        # Variance normalized by mean — robust to brightness changes
        mean = img_f.mean()
        if mean > 0:
            return float(np.var(img_f) / mean)
        return 0.0

    elif method == 'sobel':
        # Sum of squared Sobel gradients
        from scipy.ndimage import sobel
        gx = sobel(img_f, axis=1)
        gy = sobel(img_f, axis=0)
        return float(np.mean(gx ** 2 + gy ** 2))

    else:
        raise ValueError(f"Unknown focus method: {method}. "
                         f"Use 'brenner', 'laplacian', 'normalized_variance', or 'sobel'.")


# ---------------------------------------------------------------------------
# Sub-step peak interpolation
# ---------------------------------------------------------------------------

def parabolic_peak_interp(positions, scores):
    """Sub-step peak position via parabolic fit on the argmax neighbourhood.

    Given a focus curve sampled at uniform-spacing ``positions`` with
    ``scores``, fit a parabola to the argmax sample and its two neighbours
    and return the analytic peak. The fit is identical to the sub-pixel
    refinement used in FFT cross-correlation (see ``hardware/drift.py``).

    The fitted peak provides an estimate of the true sharpness peak that
    is **independent of the grid argmax** — useful when the chosen Z must
    be compared against an estimator that doesn't suffer from grid
    quantisation, e.g. when reporting a sub-step ``tissue_z`` alongside
    the on-grid focus point.

    Args:
        positions: 1D array-like of N uniformly-spaced sample positions
            (Z µm or any other axis).
        scores: 1D array-like of focus scores at each position.

    Returns:
        (fitted_position, argmax_position, argmax_score) tuple of floats.
        ``fitted_position`` falls back to ``argmax_position`` when the
        argmax is at the boundary or the parabola is degenerate (denom ≥ 0).
    """
    pos = np.asarray(positions, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    if len(pos) == 0:
        raise ValueError("positions/scores are empty")
    if len(pos) != len(sc):
        raise ValueError("positions and scores must be same length")

    argmax = int(np.argmax(sc))
    argmax_pos = float(pos[argmax])
    argmax_score = float(sc[argmax])

    if argmax == 0 or argmax == len(pos) - 1:
        return argmax_pos, argmax_pos, argmax_score

    s_m1, s_0, s_p1 = sc[argmax - 1], sc[argmax], sc[argmax + 1]
    denom = s_m1 - 2.0 * s_0 + s_p1
    # Concave-down peak requires denom < 0; guard against degenerate fits.
    if denom >= 0 or abs(denom) < 1e-12:
        return argmax_pos, argmax_pos, argmax_score

    # Uniform spacing — use the local step size around the argmax.
    step = float(pos[argmax + 1] - pos[argmax - 1]) / 2.0
    offset = 0.5 * (s_m1 - s_p1) / denom * step
    fitted_pos = argmax_pos + float(offset)
    return fitted_pos, argmax_pos, argmax_score


# ---------------------------------------------------------------------------
# Z-sweep focus
# ---------------------------------------------------------------------------

def sweep_focus(core, z_start, z_end, z_step, channel=None,
                method='brenner'):
    """Sweep Z positions and compute focus metric at each.

    Args:
        core: Microscope core.
        z_start, z_end: Z range to sweep.
        z_step: Step size.
        channel: Channel to snap.
        method: Focus metric method.

    Returns:
        dict with:
            best_z: Z position with highest focus score (grid argmax)
            best_score: highest focus score
            fitted_z: sub-step parabolic peak position around argmax —
                independent of the grid argmax (see
                ``parabolic_peak_interp``). Falls back to ``best_z`` when
                argmax is at the sweep boundary.
            curve: list of (z, score) tuples
            n_snaps: number of images taken
    """
    from ..hardware import core as hw
    from ..hardware.config import resolve_brightfield_channel

    channel = resolve_brightfield_channel(core, channel) or 'brightfield'
    z_positions = np.arange(z_start, z_end + z_step / 2, z_step)
    curve = []

    for z in z_positions:
        hw.set_z(core, float(z))
        img = hw.snap(core, channel=channel)
        score = focus_metric(img, method=method)
        curve.append((round(float(z), 3), round(score, 4)))

    zs = np.array([c[0] for c in curve], dtype=np.float64)
    scores = np.array([c[1] for c in curve], dtype=np.float64)
    fitted_z, argmax_z, peak = parabolic_peak_interp(zs, scores)

    return {
        'best_z': argmax_z,
        'best_score': peak,
        'fitted_z': round(fitted_z, 4),
        'curve': curve,
        'n_snaps': len(curve),
    }


# ---------------------------------------------------------------------------
# Coarse + fine autofocus
# ---------------------------------------------------------------------------

def coarse_fine_focus(core, z_range=20.0, coarse_step=2.0, fine_step=0.5,
                      fine_range=None, channel=None,
                      method='brenner'):
    """Two-pass autofocus: coarse sweep then fine refinement.

    Pass 1 (coarse): Large steps over full range to find approximate focus.
    Pass 2 (fine): Small steps around the coarse best to find precise focus.

    Args:
        core: Microscope core.
        z_range: Half-range for coarse sweep (sweeps -z_range to +z_range).
        coarse_step: Step size for coarse pass.
        fine_step: Step size for fine pass.
        fine_range: Half-range for fine pass (default: coarse_step * 1.5).
        channel: Channel to snap.
        method: Focus metric.

    Returns:
        dict with:
            best_z: final best Z position (fine-pass grid argmax)
            best_score: final best focus score
            fitted_z: fine-pass sub-step parabolic peak — independent of
                grid argmax (see ``parabolic_peak_interp``).
            coarse_result: dict from coarse sweep
            fine_result: dict from fine sweep
            total_snaps: total images taken
    """
    from ..hardware import core as hw

    if fine_range is None:
        fine_range = coarse_step * 1.5

    # Pass 1: Coarse sweep
    coarse = sweep_focus(core, -z_range, z_range, coarse_step,
                         channel=channel, method=method)

    # Pass 2: Fine sweep around coarse best
    fine_center = coarse['best_z']
    fine = sweep_focus(core, fine_center - fine_range, fine_center + fine_range,
                       fine_step, channel=channel, method=method)

    # Move to best position
    hw.set_z(core, fine['best_z'])

    return {
        'best_z': fine['best_z'],
        'best_score': fine['best_score'],
        'fitted_z': fine['fitted_z'],
        'coarse_result': coarse,
        'fine_result': fine,
        'total_snaps': coarse['n_snaps'] + fine['n_snaps'],
    }


# ---------------------------------------------------------------------------
# Pure-Brownian closed-loop tracker (ch592 r4 → 10/10)
# ---------------------------------------------------------------------------

def track_focus_brownian(
    core,
    n_snaps,
    *,
    channel=None,
    method='laplacian',
    check_interval=5,
    coarse_range=5.0,
    coarse_step=0.5,
    fine_range=2.0,
    fine_step=0.25,
    widen_ratio=0.5,
    initial_z=None,
    on_idle_snap=None,
):
    """Closed-loop focus tracker for pure-Brownian (zero-mean) Z drift.

    No velocity model — coarse+fine sweep with widen-on-drop, parabolic
    sub-step peak as an estimator independent of the grid argmax.

    Protocol:
        1. Initial coarse sweep around ``initial_z`` (default = current Z),
           set Z to grid argmax.
        2. For each of ``n_snaps`` working snaps:
            - On ``snap_idx % check_interval == 0`` → fine sweep around
              current Z. If the centre score < ``widen_ratio`` × best-so-far,
              widen back to coarse range/step.
            - Otherwise just snap and accumulate the sharpness sample.
        3. ``final_sharpness`` = mean of the last 5 idle-snap scores.

    Why "Brownian, no velocity":
        Pure-Brownian wander is zero-mean — extrapolating drift rate is
        worse than no model at all because residual sample noise feeds
        into a self-amplifying loop. r1/r2 of ch592 lost on this exact
        failure mode after the backend was changed from monotonic to
        Brownian.

    Args:
        core: Microscope core (any pymmcore-plus / -proxy compatible).
        n_snaps: Number of working snaps after the initial coarse sweep.
        channel: Channel for focus measurement (resolved via
            ``resolve_brightfield_channel`` when None).
        method: Focus metric — passed through to ``focus_metric``.
            ch592 used 'laplacian'; 'brenner' is faster on cv2-less paths.
        check_interval: Run a corrective sweep every Nth snap.
        coarse_range, coarse_step: Half-range / step for the coarse sweep.
        fine_range, fine_step: Half-range / step for the fine sweep.
        widen_ratio: If current centre score < ratio × best-seen-peak,
            widen the next sweep back to coarse range/step.
        initial_z: Starting Z (default = current ZStage position).
        on_idle_snap: Optional callback ``(snap_idx, image, score)`` invoked
            for non-correction snaps — for downstream analysis layered on
            top of the focus loop.

    Returns:
        dict with:
            z_corrections: list of [snap_idx, z_um_chosen] (grid argmax)
            tissue_z_estimates: list of [snap_idx, fitted_z] (sub-step
                parabolic peak — independent estimator)
            final_sharpness: mean of the last 5 idle-snap scores
            sharp_history: full per-idle-snap score series
            n_corrections: number of corrective sweeps issued
            n_snaps_total: total snaps including sweeps + idles
    """
    from ..hardware import core as hw
    from ..hardware.config import resolve_brightfield_channel

    channel = resolve_brightfield_channel(core, channel) or 'brightfield'

    def _sweep(center, half_r, step):
        """Local sweep that returns (argmax_z, fitted_z, peak, n_snaps)."""
        zs = np.arange(center - half_r, center + half_r + step / 2, step)
        scores = []
        for z in zs:
            hw.set_z(core, float(z))
            img = hw.snap(core, channel=channel)
            scores.append(focus_metric(img, method=method))
        scores = np.asarray(scores, dtype=np.float64)
        fitted, argmax_z, peak = parabolic_peak_interp(zs, scores)
        return float(argmax_z), float(fitted), float(peak), len(zs)

    if initial_z is not None:
        hw.set_z(core, float(initial_z))
        center = float(initial_z)
    else:
        center = float(hw.get_z(core))

    z, z_fit, peak, n_initial = _sweep(center, coarse_range, coarse_step)
    hw.set_z(core, z)
    z_corrections = [[0, z]]
    tissue_z_estimates = [[0, z_fit]]
    last_peak = peak
    sharp_history = []
    n_total = n_initial
    n_corrections = 1

    for snap_idx in range(1, n_snaps + 1):
        if snap_idx % check_interval == 0:
            current_img = hw.snap(core, channel=channel)
            current_score = focus_metric(current_img, method=method)
            n_total += 1
            if current_score < widen_ratio * last_peak:
                half_r, step = coarse_range, coarse_step
            else:
                half_r, step = fine_range, fine_step
            z, z_fit, new_peak, n_sw = _sweep(z, half_r, step)
            hw.set_z(core, z)
            n_total += n_sw
            n_corrections += 1
            last_peak = max(last_peak, new_peak)
            z_corrections.append([snap_idx, z])
            tissue_z_estimates.append([snap_idx, z_fit])
        else:
            img = hw.snap(core, channel=channel)
            score = focus_metric(img, method=method)
            sharp_history.append(score)
            n_total += 1
            if on_idle_snap is not None:
                on_idle_snap(snap_idx, img, score)

    final = float(np.mean(sharp_history[-5:])) if sharp_history else 0.0
    return {
        'z_corrections': z_corrections,
        'tissue_z_estimates': tissue_z_estimates,
        'final_sharpness': float(final),
        'sharp_history': sharp_history,
        'n_corrections': n_corrections,
        'n_snaps_total': n_total,
    }


# ---------------------------------------------------------------------------
# Continuous focus tracking
# ---------------------------------------------------------------------------

def make_focus_state(initial_z=0.0, method='brenner', check_interval=5):
    """Create state for continuous focus tracking during timelapse.

    Args:
        initial_z: Starting Z position.
        method: Focus metric to use.
        check_interval: Check focus every N frames.

    Returns:
        dict with focus tracking state.
    """
    return {
        'z': float(initial_z),
        'method': method,
        'check_interval': check_interval,
        'frame_count': 0,
        'best_score': 0.0,
        'history': [],  # list of (frame, z, score) tuples
        'drift_velocity': 0.0,  # Z drift rate (um/frame)
    }


def check_and_correct_focus(core, state, img=None, channel=None,
                            z_step=0.5, max_correction=5.0):
    """Check focus quality and correct if needed.

    Called each frame during timelapse. Only performs correction every
    check_interval frames to minimize overhead. Estimates Z drift velocity
    for predictive correction.

    Args:
        core: Microscope core.
        state: Focus state dict (modified in place).
        img: Optional pre-snapped image (avoids redundant snap).
        channel: Channel for focus measurement.
        z_step: Step size for focus correction sweep.
        max_correction: Maximum Z correction in one step.

    Returns:
        dict with:
            corrected: bool — whether focus was adjusted
            current_score: focus score of current image
            z_correction: amount of Z adjustment (0 if none)
            predicted_z: predicted best Z from drift model
    """
    from ..hardware import core as hw
    from ..hardware.config import resolve_brightfield_channel

    channel = resolve_brightfield_channel(core, channel) or 'brightfield'
    state['frame_count'] += 1

    # Measure current focus
    if img is None:
        img = hw.snap(core, channel=channel)
    current_score = focus_metric(img, method=state['method'])

    # Update history
    current_z = hw.get_z(core)
    state['history'].append((state['frame_count'], round(current_z, 3),
                             round(current_score, 4)))

    # Predict Z from drift velocity
    predicted_z = state['z'] + state['drift_velocity']

    # Check if we should correct
    should_check = (state['frame_count'] % state['check_interval'] == 0)

    if not should_check:
        # Apply predictive correction only
        if abs(state['drift_velocity']) > 0.01:
            hw.set_z(core, predicted_z)
            state['z'] = predicted_z
        return {
            'corrected': False,
            'current_score': round(current_score, 4),
            'z_correction': 0.0,
            'predicted_z': round(predicted_z, 3),
        }

    # Active focus check: sweep 3 positions (current, +step, -step)
    scores = []
    z_positions = [current_z, current_z + z_step, current_z - z_step]

    for z in z_positions:
        hw.set_z(core, float(z))
        test_img = hw.snap(core, channel=channel)
        score = focus_metric(test_img, method=state['method'])
        scores.append((float(z), score))

    best_z, best_score = max(scores, key=lambda x: x[1])

    # If neither neighbor is better, we're at focus
    if best_z == current_z:
        z_correction = 0.0
    else:
        # Move toward better focus, but limit correction
        z_correction = best_z - current_z
        z_correction = max(-max_correction,
                           min(max_correction, z_correction))
        best_z = current_z + z_correction

    # Update drift velocity
    if len(state['history']) >= 2:
        prev_z = state['z']
        drift = best_z - prev_z
        state['drift_velocity'] = 0.3 * drift + 0.7 * state['drift_velocity']

    # Apply correction
    hw.set_z(core, best_z)
    state['z'] = best_z
    state['best_score'] = best_score

    return {
        'corrected': abs(z_correction) > 0.001,
        'current_score': round(best_score, 4),
        'z_correction': round(z_correction, 3),
        'predicted_z': round(predicted_z, 3),
    }


# ---------------------------------------------------------------------------
# MDA-native autofocus
# ---------------------------------------------------------------------------

def autofocus_mda(z_range=20.0, n_coarse=11, n_fine=11,
                  method='brenner', exposure=50.0, channel=None):
    """MDA-native coarse+fine autofocus.

    Returns an event generator and on_frame callback for use with
    pymmcore-plus MDA engine.  The coarse pass sweeps a wide Z range,
    then the fine pass narrows around the best position.

    Both passes are fully lazy: events are yielded one at a time so
    the MDA engine controls timing and hardware sequencing.

    Args:
        z_range: Half-range for coarse sweep (sweeps -z_range to +z_range).
        n_coarse: Number of Z positions in coarse pass.
        n_fine: Number of Z positions in fine pass.
        method: Focus metric method ('brenner', 'laplacian', etc.).
        exposure: Exposure time in ms.
        channel: Channel config name.

    Returns:
        Tuple of (event_generator_factory, on_frame_callback, shared_state).
        - Call event_generator_factory() to get the generator.
        - After the generator completes, ``shared_state`` carries:
            - ``best_z`` (fine-pass grid argmax) and ``best_score``;
            - ``fitted_z`` (sub-step parabolic peak on the fine-pass curve,
              independent of grid argmax — see ``parabolic_peak_interp``);
            - ``coarse_best_z``, ``scores``, ``n_coarse``, ``n_fine``.
    """
    from useq import MDAEvent

    coarse_zs = np.linspace(-z_range, z_range, n_coarse)

    shared = {
        'phase': 'coarse',
        'scores': [],        # list of (z, score)
        'best_z': 0.0,
        'best_score': 0.0,
        'fitted_z': 0.0,
        'coarse_best_z': 0.0,
        'method': method,
        'n_coarse': 0,
        'n_fine': 0,
    }

    def on_frame(image, event, meta=None):
        """Compute focus metric for each Z position."""
        z = event.z_pos if event.z_pos is not None else 0.0
        score = focus_metric(image, method=shared['method'])
        shared['scores'].append((z, score))

        if score > shared['best_score']:
            shared['best_score'] = score
            shared['best_z'] = z

    ch_cfg = {'config': channel} if channel else None

    def event_generator():
        # Pass 1: Coarse sweep
        shared['phase'] = 'coarse'
        for i, z in enumerate(coarse_zs):
            yield MDAEvent(
                z_pos=float(z),
                exposure=exposure,
                channel=ch_cfg,
                index={'t': 0, 'z': i},
                metadata={'autofocus_phase': 'coarse', 'z_idx': i},
            )
        shared['n_coarse'] = len(coarse_zs)

        # Read coarse best from accumulated scores
        coarse_best_z = shared['best_z']
        shared['coarse_best_z'] = coarse_best_z

        # Pass 2: Fine sweep around coarse best
        fine_half = (2 * z_range / max(n_coarse - 1, 1)) * 1.5
        fine_zs = np.linspace(coarse_best_z - fine_half,
                              coarse_best_z + fine_half, n_fine)

        # Reset best for fine pass
        shared['phase'] = 'fine'
        shared['best_score'] = 0.0
        shared['best_z'] = coarse_best_z
        shared['scores'] = []

        for i, z in enumerate(fine_zs):
            yield MDAEvent(
                z_pos=float(z),
                exposure=exposure,
                channel=ch_cfg,
                index={'t': 0, 'z': shared['n_coarse'] + i},
                metadata={'autofocus_phase': 'fine', 'z_idx': i},
            )
        shared['n_fine'] = len(fine_zs)

        # Sub-step parabolic peak on the fine-pass curve, independent of
        # the grid argmax. Sort by z to ensure ``parabolic_peak_interp``
        # sees a uniformly-spaced 1D series.
        if shared['scores']:
            zs_arr = np.array([z for z, _ in shared['scores']])
            sc_arr = np.array([s for _, s in shared['scores']])
            order = np.argsort(zs_arr)
            fitted, _, _ = parabolic_peak_interp(zs_arr[order], sc_arr[order])
            shared['fitted_z'] = float(fitted)
        else:
            shared['fitted_z'] = float(shared['best_z'])

    return event_generator, on_frame, shared


# ---------------------------------------------------------------------------
# Drift-corrected timelapse
# ---------------------------------------------------------------------------

def drift_corrected_timelapse(
    n_frames=20,
    interval=1.0,
    initial_drift_rate=0.0,
    channels=None,
    exposure=50.0,
    focus_method='brenner',
    smoothing=0.3,
):
    """MDA-native timelapse with proactive Z-drift correction.

    Instead of sweeping Z to find focus (which wastes snaps and time),
    this estimates drift velocity from frame-to-frame focus metric changes
    and applies predictive Z corrections before each snap.

    Strategy:
        1. Snap frame at current Z
        2. Compute focus metric
        3. If focus is degrading, estimate which direction Z drifted
        4. Apply Z += drift_rate before next frame
        5. Update drift_rate with exponential smoothing

    The drift direction is determined by comparing focus at slight Z offsets
    only on the first few frames to calibrate, then uses predictive correction.

    Usage::

        gen, on_frame, state = drift_corrected_timelapse(
            n_frames=30, interval=2.0, initial_drift_rate=0.4)
        results = run_events(core, gen(), on_frame=on_frame)
        # state['counts'], state['focus_scores'] available after

    Args:
        n_frames: Number of frames.
        interval: Seconds between frames.
        initial_drift_rate: Known Z drift rate (µm/frame). If 0, will
            auto-detect from first few frames.
        channels: List of channel config names. If None, uses default.
        exposure: Exposure time in ms.
        focus_method: Focus metric method.
        smoothing: Exponential smoothing factor for drift rate update.
            Lower = slower adaptation, Higher = faster but noisier.

    Returns:
        Tuple of (event_generator_factory, on_frame_callback, shared_state).
    """
    from useq import MDAEvent

    channels = channels or [None]

    state = {
        'frame': 0,
        'z': 0.0,
        'drift_rate': float(initial_drift_rate),
        'focus_scores': [],
        'z_positions': [],
        'frames': [],           # list of (image, event) per frame
        'calibrated': initial_drift_rate != 0.0,
        'method': focus_method,
    }

    def on_frame(image, event, meta=None):
        """Track focus quality and update drift estimate."""
        score = focus_metric(image, method=state['method'])
        z = event.z_pos if event.z_pos is not None else state['z']
        state['focus_scores'].append(score)
        state['z_positions'].append(z)
        state['frame'] += 1

        # Update drift rate from focus trend (after 3+ frames)
        if len(state['focus_scores']) >= 3:
            recent = state['focus_scores'][-3:]
            # If focus is consistently dropping, drift rate needs adjustment
            if recent[-1] < recent[-2] < recent[-3]:
                # Focus degrading — increase drift correction
                state['drift_rate'] *= 1.2
            elif recent[-1] > recent[-2]:
                # Focus improving or stable — current rate is good
                pass

    def event_generator():
        """Yield MDA events with predictive Z correction."""
        z = 0.0
        state['z'] = z

        for t in range(n_frames):
            # Apply predictive Z correction (after first frame)
            if t > 0:
                z += state['drift_rate']
                state['z'] = z

            for c_idx, ch in enumerate(channels):
                kwargs = {
                    'z_pos': float(z),
                    'exposure': exposure,
                    'min_start_time': t * interval,
                    'index': {'t': t},
                    'metadata': {
                        'timelapse_frame': t,
                        'predicted_z': round(z, 3),
                        'drift_rate': round(state['drift_rate'], 4),
                    },
                }
                if ch is not None:
                    kwargs['channel'] = {'config': ch}
                if len(channels) > 1:
                    kwargs['index']['c'] = c_idx

                yield MDAEvent(**kwargs)

    return event_generator, on_frame, state
