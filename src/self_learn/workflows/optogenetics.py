"""Optogenetics experiment workflows.

Provides reusable functions for SLM-targeted stimulation experiments:
- Soma detection from structural fluorescence channels
- Baseline/stimulation/recovery acquisition with proper MDA timing
- ΔF/F analysis with ROI extraction
"""

import numpy as np
from useq import MDAEvent
from scipy.ndimage import label, gaussian_filter

from ..hardware.core import run_events, make_slm_circle


def detect_somata_fluorescence(image, min_area=50, edge_margin=20, sigma=3.0,
                               percentile=95):
    """Detect neuron somata from a structural fluorescence image.

    Uses high-threshold connected components on Gaussian-smoothed image.
    Filters by area and edge proximity.

    Note: Named ``detect_somata_fluorescence`` to distinguish from
    ``src.core.detection.neurons.detect_somata`` which uses MAP2 + distance
    transform. This function is generic CC-based detection.

    Args:
        image: 2D structural channel image (e.g., MAP2, GFP).
        min_area: Minimum soma area in pixels.
        edge_margin: Exclude somata within this distance of image edge.
        sigma: Gaussian smoothing sigma for noise reduction.
        percentile: Percentile of non-zero pixels to use as threshold.

    Returns:
        List of dicts with keys: id, cx, cy, area, label (CC label),
        radius (estimated from area).
    """
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    smooth = gaussian_filter(img, sigma=sigma)
    nonzero = smooth[smooth > 5]
    if len(nonzero) < 10:
        return []

    thresh = np.percentile(nonzero, percentile)
    binary = smooth > thresh
    labeled, n = label(binary)

    somata = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        area = len(ys)
        if area < min_area:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        if cx < edge_margin or cx > w - edge_margin:
            continue
        if cy < edge_margin or cy > h - edge_margin:
            continue

        radius = float(np.sqrt(area / np.pi))
        somata.append({
            'id': len(somata),
            'cx': cx, 'cy': cy,
            'area': area,
            'label': i,
            'radius': radius,
        })

    return somata


# Backward-compatible alias
detect_somata = detect_somata_fluorescence


def make_soma_rois(somata, labeled_image, dilation=1.3, min_radius=12):
    """Create boolean ROI masks for each soma.

    Combines the labeled CC mask with a circular region around the centroid
    for robust intensity measurement.

    Args:
        somata: List of soma dicts (from detect_somata).
        labeled_image: The labeled image from scipy.ndimage.label.
        dilation: Factor to dilate the circular ROI beyond soma radius.
        min_radius: Minimum ROI radius in pixels.

    Returns:
        Dict mapping soma id -> 2D boolean mask.
    """
    h, w = labeled_image.shape
    yy, xx = np.ogrid[:h, :w]
    rois = {}

    for s in somata:
        base_mask = labeled_image == s['label']
        roi_r = max(min_radius, s['radius'] * dilation)
        circ = ((xx - s['cx'])**2 + (yy - s['cy'])**2) <= roi_r**2
        rois[s['id']] = base_mask | circ

    return rois


def slm_stimulation_experiment(core, channel_config, channel_group,
                               target_center, slm_radius=20,
                               n_baseline=8, n_stim=10, n_recovery=8,
                               baseline_interval=2.0, stim_interval=1.5,
                               recovery_interval=2.0, exposure=100.0,
                               decay_wait=0.0):
    """Run a complete baseline → stimulation → recovery experiment.

    Uses MDA events with min_start_time for proper temporal spacing.

    Args:
        core: Microscope core connection.
        channel_config: Channel config name (e.g., 'nucleus-channel').
        channel_group: Channel group name (e.g., 'Fake').
        target_center: (cx, cy) in viewport pixels for SLM targeting.
        slm_radius: Radius of SLM stimulation circle (pixels).
        n_baseline: Number of baseline frames.
        n_stim: Number of stimulation frames.
        n_recovery: Number of recovery frames.
        baseline_interval: Time between baseline frames (seconds).
        stim_interval: Time between stim frames (seconds).
        recovery_interval: Time between recovery frames (seconds).
        exposure: Camera exposure (ms).
        decay_wait: Extra wait before baseline starts (seconds).

    Returns:
        dict with keys: baseline_frames, stim_frames, recovery_frames
        (each a list of numpy arrays).
    """
    import time

    if decay_wait > 0:
        clear = np.zeros((512, 512), dtype=np.uint8)
        core.setSLMImage("SLM", clear)
        core.displaySLMImage("SLM")
        time.sleep(decay_wait)

    ch_kw = {"config": channel_config, "group": channel_group}
    data = {
        'baseline_frames': [],
        'stim_frames': [],
        'recovery_frames': [],
    }
    current_phase = ['baseline']

    def on_frame(img, event):
        key = current_phase[0] + '_frames'
        data[key].append(img.copy())

    # Phase 1: Baseline
    current_phase[0] = 'baseline'
    baseline_events = [
        MDAEvent(channel=ch_kw, exposure=exposure,
                 min_start_time=i * baseline_interval)
        for i in range(n_baseline)
    ]
    run_events(core, baseline_events, on_frame=on_frame)

    # Phase 2: Stimulation (set SLM before acquiring)
    current_phase[0] = 'stim'
    cx, cy = int(target_center[0]), int(target_center[1])
    slm_mask = make_slm_circle((cx, cy), slm_radius, size=512)
    core.setSLMImage("SLM", slm_mask)
    core.displaySLMImage("SLM")

    stim_events = [
        MDAEvent(channel=ch_kw, exposure=exposure,
                 min_start_time=i * stim_interval)
        for i in range(n_stim)
    ]
    run_events(core, stim_events, on_frame=on_frame)

    # Phase 3: Recovery (clear SLM)
    current_phase[0] = 'recovery'
    clear = np.zeros((512, 512), dtype=np.uint8)
    core.setSLMImage("SLM", clear)
    core.displaySLMImage("SLM")
    recovery_events = [
        MDAEvent(channel=ch_kw, exposure=exposure,
                 min_start_time=i * recovery_interval)
        for i in range(n_recovery)
    ]
    run_events(core, recovery_events, on_frame=on_frame)

    return data


def analyze_stimulation(data, soma_rois, target_id):
    """Analyze ΔF/F results from a stimulation experiment.

    Args:
        data: Dict from slm_stimulation_experiment with frame lists.
        soma_rois: Dict from make_soma_rois.
        target_id: ID of the stimulated soma.

    Returns:
        Dict mapping soma_id -> dict with baseline_mean, stim_mean,
        recovery_mean, dff_mean, dff_peak, is_target, traces.
    """
    from ..analysis.intensity import extract_roi_traces, compute_dff

    results = {}

    for phase_key in ['baseline_frames', 'stim_frames', 'recovery_frames']:
        frames = data[phase_key]
        if not frames:
            continue
        traces = extract_roi_traces(frames, soma_rois)
        for sid, trace in traces.items():
            if sid not in results:
                results[sid] = {'is_target': (sid == target_id)}
            short_key = phase_key.replace('_frames', '')
            results[sid][f'{short_key}_trace'] = trace
            results[sid][f'{short_key}_mean'] = float(trace.mean())

    # Compute ΔF/F using baseline as F₀
    for sid, r in results.items():
        bl = r.get('baseline_mean', 0)
        st = r.get('stim_mean', 0)
        if bl > 1:
            r['dff_mean'] = (st - bl) / bl
            st_trace = r.get('stim_trace', np.array([st]))
            r['dff_peak'] = float((st_trace.max() - bl) / bl)
        else:
            r['dff_mean'] = st - bl
            r['dff_peak'] = st - bl

    return results


def connectivity_mapping(core, soma_positions, channel_config='nucleus-channel',
                         channel_group=None, slm_radius=20, exposure=50,
                         n_post_frames=3, n_decay_snaps=8,
                         soma_roi_radius=15, resting_threshold=50,
                         response_threshold=120,
                         dff_threshold=None, direct_only=True):
    """Map directed functional connectivity by stimulating each neuron.

    Protocol per neuron:
    1. Baseline snap (SLM off) — all neurons at resting intensity
    2. SLM on, stim snap — target neuron fires
    3. SLM still on, n_post_frames observation snaps — direct connections
       fire on the first post-stim snap (1 synaptic hop delay)
    4. SLM off, n_decay_snaps decay frames — calcium decays to baseline

    SLM stays ON during observation so the source neuron sustains firing,
    giving downstream neurons time to respond.

    Args:
        core: Microscope core connection.
        soma_positions: List of (cx, cy) pixel coordinates for each neuron.
        channel_config: GCaMP channel config name.
        channel_group: Config group name.
        slm_radius: Radius of SLM stimulation circle (pixels).
        exposure: Camera exposure in ms.
        n_post_frames: Number of observation frames with SLM still on.
        n_decay_snaps: Number of decay frames with SLM off.
        soma_roi_radius: Pixel radius for intensity measurement ROI.
        resting_threshold: Max intensity considered "resting" (below = resting).
            Used only when dff_threshold is None.
        response_threshold: Min intensity to be considered "responding".
            Used only when dff_threshold is None.
        dff_threshold: If set, use ΔF/F-based detection instead of absolute
            thresholds. Recommended: 2.0 (= 3× baseline intensity). This is
            scale-invariant and generally preferred.
        direct_only: When using dff_threshold, only include direct (1-hop)
            connections (peak at first observation snap). Default True.

    Returns:
        dict with:
            adjacency: dict mapping neuron_index -> list of target indices.
            stim_data: dict mapping neuron_index -> {baseline, stim, post} values.
            n_connections: int, total directed edges.
    """
    from ..hardware.core import make_slm_circle, run_events
    from ..hardware.config import resolve_channel_group

    n_neurons = len(soma_positions)
    channel_group = resolve_channel_group(core, channel_group)
    ch_kw = {"config": channel_config}
    if channel_group is not None:
        ch_kw["group"] = channel_group

    def _measure_somata(img):
        """Measure mean intensity at each soma within circular ROI."""
        h, w = img.shape[:2]
        vals = []
        r = soma_roi_radius
        for cx, cy in soma_positions:
            ci, ri = int(cx), int(cy)
            y0, y1 = max(0, ri - r), min(h, ri + r + 1)
            x0, x1 = max(0, ci - r), min(w, ci + r + 1)
            patch = img[y0:y1, x0:x1].astype(float)
            yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            mask = ((xx - (ci - x0))**2 + (yy - (ri - y0))**2) <= r**2
            vals.append(float(patch[mask].mean()) if mask.any() else 0.0)
        return vals

    clear_mask = np.zeros((512, 512), dtype=np.uint8)

    state = {
        'adjacency': {i: [] for i in range(n_neurons)},
        'stim_data': {i: {'baseline': None, 'stim': None, 'post': []}
                      for i in range(n_neurons)},
        'n_connections': 0,
    }

    def on_frame(image, event):
        md = getattr(event, 'metadata', {}) or {}
        neuron_idx = md.get('neuron', 0)
        phase = md.get('phase', 'unknown')
        if phase == 'decay':
            return
        vals = _measure_somata(image)
        if phase == 'baseline':
            state['stim_data'][neuron_idx]['baseline'] = vals
        elif phase == 'stim':
            state['stim_data'][neuron_idx]['stim'] = vals
        elif phase == 'post':
            state['stim_data'][neuron_idx]['post'].append(vals)

    def events():
        """Generator: control SLM between event yields."""
        for stim_idx in range(n_neurons):
            cx, cy = soma_positions[stim_idx]

            # Baseline (SLM off)
            core.setSLMImage("SLM", clear_mask)
            core.displaySLMImage("SLM")
            yield MDAEvent(
                channel=ch_kw, exposure=exposure,
                metadata={'neuron': stim_idx, 'phase': 'baseline'},
            )

            # Stimulation (SLM on)
            slm_mask = make_slm_circle((cx, cy), radius=slm_radius,
                                       size=512, core=core)
            core.setSLMImage("SLM", slm_mask)
            core.displaySLMImage("SLM")
            yield MDAEvent(
                channel=ch_kw, exposure=exposure,
                metadata={'neuron': stim_idx, 'phase': 'stim'},
            )

            # Observation (SLM stays on — cascade propagation)
            for pf in range(n_post_frames):
                yield MDAEvent(
                    channel=ch_kw, exposure=exposure,
                    metadata={'neuron': stim_idx, 'phase': 'post',
                              'post_idx': pf},
                )

            # Decay (SLM off — calcium returns to baseline)
            core.setSLMImage("SLM", clear_mask)
            core.displaySLMImage("SLM")
            for _ in range(n_decay_snaps):
                yield MDAEvent(
                    channel=ch_kw, exposure=exposure,
                    metadata={'neuron': stim_idx, 'phase': 'decay'},
                )

        # Clear SLM after all stimulations
        core.setSLMImage("SLM", clear_mask)
        core.displaySLMImage("SLM")

    run_events(core, events(), on_frame=on_frame)

    if dff_threshold is not None:
        result = infer_connectivity_dff(
            state, n_neurons, dff_threshold=dff_threshold,
            direct_only=direct_only,
        )
    else:
        _infer_adjacency(state, n_neurons, resting_threshold,
                         response_threshold)

    return {
        'adjacency': state['adjacency'],
        'stim_data': state['stim_data'],
        'n_connections': state['n_connections'],
    }


# ---------------------------------------------------------------------------
# MDA-native connectivity mapping
# ---------------------------------------------------------------------------

def connectivity_mapping_events(
    core,
    soma_positions,
    channel_config='nucleus-channel',
    channel_group=None,
    slm_radius=20,
    exposure=50,
    n_post_frames=3,
    n_decay_snaps=8,
    soma_roi_radius=15,
    resting_threshold=50,
    response_threshold=120,
):
    """MDA-native connectivity mapping via sequential neuron stimulation.

    Protocol per neuron:
    1. Baseline snap (SLM off)
    2. SLM on, stim snap — target fires
    3. SLM still on, n_post_frames observation snaps — cascade propagation
    4. SLM off, n_decay_snaps decay frames — calcium returns to baseline

    SLM stays ON during observation so the source neuron sustains firing.
    The generator controls the SLM device directly between yields
    (via core.setSLMImage/displaySLMImage) rather than embedding
    SLMImage objects in MDA events, which avoids serialization issues
    when running over pymmcore-proxy.

    Usage::

        gen, on_frame, state = connectivity_mapping_events(
            core,
            [(100, 200), (300, 150)],
            channel_config='GCaMP', channel_group='Fake',
        )
        results = run_events(core, gen(), on_frame=on_frame)
        print(state['adjacency'])
        print(state['n_connections'])

    Args:
        core: Microscope core connection (needed for SLM control).
        soma_positions: List of (cx, cy) pixel coordinates for each neuron.
        channel_config: GCaMP channel config name.
        channel_group: Config group name.
        slm_radius: Radius of SLM stimulation circle (pixels).
        exposure: Camera exposure in ms.
        n_post_frames: Number of observation frames with SLM still on.
        n_decay_snaps: Number of decay frames with SLM off.
        soma_roi_radius: Pixel radius for intensity measurement ROI.
        resting_threshold: Max intensity considered resting.
        response_threshold: Min intensity to be considered responding.

    Returns:
        Tuple of (event_generator_factory, on_frame_callback, shared_state).
        - Call event_generator_factory() to get the generator.
        - shared_state['adjacency'], shared_state['n_connections'] are
          populated after the generator completes.
    """
    from ..hardware.config import resolve_channel_group

    n_neurons = len(soma_positions)
    channel_group = resolve_channel_group(core, channel_group)
    ch_kw = {"config": channel_config}
    if channel_group is not None:
        ch_kw["group"] = channel_group

    shared = {
        'adjacency': {i: [] for i in range(n_neurons)},
        'stim_data': {i: {'baseline': None, 'stim': None, 'post': []}
                      for i in range(n_neurons)},
        'n_connections': 0,
        '_current_neuron': 0,
        '_current_phase': 'baseline',
    }

    def _measure_somata(img):
        """Measure mean intensity at each soma within circular ROI."""
        h, w = img.shape[:2]
        vals = []
        r = soma_roi_radius
        for cx, cy in soma_positions:
            ci, ri = int(cx), int(cy)
            y0, y1 = max(0, ri - r), min(h, ri + r + 1)
            x0, x1 = max(0, ci - r), min(w, ci + r + 1)
            patch = img[y0:y1, x0:x1].astype(float)
            yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            mask = ((xx - (ci - x0))**2 + (yy - (ri - y0))**2) <= r**2
            vals.append(float(patch[mask].mean()) if mask.any() else 0.0)
        return vals

    def on_frame(image, event, meta=None):
        """Measure somata intensities and store per-neuron per-phase."""
        md = getattr(event, 'metadata', {}) or {}
        neuron_idx = md.get('neuron', shared['_current_neuron'])
        phase = md.get('phase', shared['_current_phase'])
        if phase == 'decay':
            return
        vals = _measure_somata(image)

        if phase == 'baseline':
            shared['stim_data'][neuron_idx]['baseline'] = vals
        elif phase == 'stim':
            shared['stim_data'][neuron_idx]['stim'] = vals
        elif phase == 'post':
            shared['stim_data'][neuron_idx]['post'].append(vals)

    def event_generator():
        """Yield events for sequential neuron stimulation.

        Controls the SLM device directly between yields rather than
        embedding SLMImage in events (avoids serialization issues over
        pymmcore-proxy). SLM stays ON during observation snaps so the
        source neuron sustains firing for cascade propagation.
        """
        from ..hardware.core import make_slm_circle
        clear_mask = np.zeros((512, 512), dtype=np.uint8)

        for stim_idx in range(n_neurons):
            shared['_current_neuron'] = stim_idx
            cx, cy = soma_positions[stim_idx]

            # Baseline frame (SLM off)
            shared['_current_phase'] = 'baseline'
            core.setSLMImage("SLM", clear_mask)
            core.displaySLMImage("SLM")
            yield MDAEvent(
                channel=ch_kw, exposure=exposure,
                metadata={'neuron': stim_idx, 'phase': 'baseline'},
            )

            # Stimulation frame (SLM on)
            shared['_current_phase'] = 'stim'
            slm_mask = make_slm_circle((cx, cy), slm_radius, size=512)
            core.setSLMImage("SLM", slm_mask)
            core.displaySLMImage("SLM")
            yield MDAEvent(
                channel=ch_kw, exposure=exposure,
                metadata={'neuron': stim_idx, 'phase': 'stim'},
            )

            # Observation frames (SLM stays on — cascade propagation)
            shared['_current_phase'] = 'post'
            for pf in range(n_post_frames):
                yield MDAEvent(
                    channel=ch_kw, exposure=exposure,
                    metadata={'neuron': stim_idx, 'phase': 'post',
                              'post_idx': pf},
                )

            # Decay frames (SLM off — calcium returns to baseline)
            core.setSLMImage("SLM", clear_mask)
            core.displaySLMImage("SLM")
            for _ in range(n_decay_snaps):
                yield MDAEvent(
                    channel=ch_kw, exposure=exposure,
                    metadata={'neuron': stim_idx, 'phase': 'decay'},
                )

        # Clear SLM and compute adjacency
        core.setSLMImage("SLM", clear_mask)
        core.displaySLMImage("SLM")
        _infer_adjacency(shared, n_neurons, resting_threshold,
                         response_threshold)

    return event_generator, on_frame, shared


def _infer_adjacency(state, n_neurons, resting_threshold, response_threshold):
    """Compute directed connectivity from cascade timing data.

    Shared by both imperative and MDA-native connectivity_mapping.
    Direct (1-hop) connections: respond in post-1, start decaying by post-2.
    Indirect (2-hop): still rising from post-1 to post-2 (filtered out).
    """
    adjacency = state['adjacency']
    for stim_idx in range(n_neurons):
        data = state['stim_data'][stim_idx]
        baseline = data['baseline']
        post = data['post']
        if not post or baseline is None:
            continue
        post1 = post[0]
        post2 = post[1] if len(post) > 1 else None

        for target in range(n_neurons):
            if target == stim_idx:
                continue
            if target >= len(post1) or target >= len(baseline):
                continue
            bl = baseline[target]
            p1 = post1[target]

            if bl >= resting_threshold or p1 <= response_threshold:
                continue

            if post2 is not None and target < len(post2):
                p2 = post2[target]
                if p2 > p1 * 1.05:
                    continue

            adjacency[stim_idx].append(target)

    state['n_connections'] = sum(len(v) for v in adjacency.values())


def infer_connectivity_dff(state, n_neurons, dff_threshold=2.0,
                           min_baseline=1.0, direct_only=True):
    """Compute directed connectivity using scale-invariant DF/F metric.

    Preferred over ``_infer_adjacency`` because it uses normalized DF/F
    ratios rather than absolute intensity thresholds, making it robust
    across different fluorescence levels and imaging conditions.

    For each stimulated neuron, computes ``(peak - baseline) / baseline``
    for all other neurons across post-stimulation observation frames.
    Connections are classified as direct (peak at snap 1) or indirect
    (peak at snap 2+) based on cascade timing.

    Expects ``state['stim_data'][i]`` with keys:
    - ``baseline``: list of per-neuron intensity measurements (SLM off)
    - ``post``: list of per-snap measurement lists (observation frames)

    Args:
        state: Dict containing ``stim_data`` and ``adjacency`` dicts.
        n_neurons: Number of neurons stimulated.
        dff_threshold: Minimum DF/F for a connection (default 2.0 = 3x
            baseline). Typical calcium responses show DF/F of 4-6.
        min_baseline: Skip targets with baseline below this (avoids
            divide-by-near-zero).
        direct_only: If True, only include connections where peak response
            is at the first observation snap (direct/1-hop connections).
            Set False to include multi-hop connections.

    Returns:
        dict with keys:
            adjacency: {neuron_idx: [target_indices]}
            n_connections: int
            details: list of dicts with src, tgt, dff, peak_snap info
    """
    adjacency = {i: [] for i in range(n_neurons)}
    details = []

    for stim_idx in range(n_neurons):
        data = state['stim_data'][stim_idx]
        bl = data.get('baseline')
        post = data.get('post', [])
        if bl is None or not post:
            continue

        for target in range(n_neurons):
            if target == stim_idx:
                continue
            if target >= len(bl):
                continue

            bl_val = bl[target]
            if bl_val < min_baseline:
                continue

            # Find peak response across observation snaps
            peak_val = bl_val
            peak_snap = -1
            for si, snap_vals in enumerate(post):
                if target < len(snap_vals) and snap_vals[target] > peak_val:
                    peak_val = snap_vals[target]
                    peak_snap = si

            if peak_snap < 0:
                continue

            dff = (peak_val - bl_val) / bl_val
            if dff < dff_threshold:
                continue

            if direct_only and peak_snap > 0:
                continue

            adjacency[stim_idx].append(target)
            details.append({
                'src': stim_idx, 'tgt': target,
                'dff': float(dff), 'peak_snap': peak_snap,
                'baseline': float(bl_val), 'peak': float(peak_val),
            })

    n_connections = sum(len(v) for v in adjacency.values())

    # Update state in-place for compatibility
    state['adjacency'] = adjacency
    state['n_connections'] = n_connections

    return {
        'adjacency': adjacency,
        'n_connections': n_connections,
        'details': details,
    }
