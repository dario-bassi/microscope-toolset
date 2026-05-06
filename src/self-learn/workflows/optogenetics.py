"""Optogenetics experiment workflows.

Provides reusable functions for SLM-targeted stimulation experiments:
- Soma detection from structural fluorescence channels
- Baseline/stimulation/recovery acquisition with proper MDA timing
- ΔF/F analysis with ROI extraction

Proven pattern from ch413: ChR2/GCaMP calcium imaging with 541% ΔF/F.

Real microscope note:
    This workflow assumes activation time = exposure time. Real opsins have
    non-negligible kinetics:
    - ChR2: activation τ_on ~1–5 ms, deactivation τ_off ~50–100 ms
    - ReaChR: slower τ_on ~10–20 ms, τ_off ~100–200 ms
    - CsChrimson: red-shifted but slower kinetics
    For high temporal precision (< 10 ms resolution):
    - Add opsin-specific kinetic models (tau_on, tau_off parameters)
    - Account for activation lag in stimulus timing
    - Use faster frame rate if measuring sub-100ms response dynamics
    - Validate kinetics match literature for your opsin/cell type combo
    For typical experiments (> 100 ms resolution), current timing is adequate.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label
from useq import MDAEvent, SLMImage

from ..hardware.core import make_slm_circle, run_events


def detect_somata_fluorescence(image, min_area=50, edge_margin=20, sigma=3.0, percentile=95):
    """Detect neuron somata from a structural fluorescence image.

    Uses high-threshold connected components on Gaussian-smoothed image.
    Filters by area and edge proximity.

    Note: Named ``detect_somata_fluorescence`` to distinguish from
    ``src.detection.neurons.detect_somata`` which uses MAP2 + distance
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
        somata.append(
            {
                "id": len(somata),
                "cx": cx,
                "cy": cy,
                "area": area,
                "label": i,
                "radius": radius,
            }
        )

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
        base_mask = labeled_image == s["label"]
        roi_r = max(min_radius, s["radius"] * dilation)
        circ = ((xx - s["cx"]) ** 2 + (yy - s["cy"]) ** 2) <= roi_r**2
        rois[s["id"]] = base_mask | circ

    return rois


def slm_stimulation_experiment(
    core,
    channel_config,
    channel_group,
    target_center,
    slm_radius=20,
    n_baseline=8,
    n_stim=10,
    n_recovery=8,
    baseline_interval=2.0,
    stim_interval=1.5,
    recovery_interval=2.0,
    exposure=100.0,
    decay_wait=0.0,
):
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
        core.setSLMPixelsTo("SLM", 0)
        time.sleep(decay_wait)

    ch_kw = {"config": channel_config, "group": channel_group}
    data = {
        "baseline_frames": [],
        "stim_frames": [],
        "recovery_frames": [],
    }
    current_phase = ["baseline"]

    def on_frame(img, event):
        key = current_phase[0] + "_frames"
        data[key].append(img.copy())

    # Phase 1: Baseline
    current_phase[0] = "baseline"
    baseline_events = [
        MDAEvent(channel=ch_kw, exposure=exposure, min_start_time=i * baseline_interval)
        for i in range(n_baseline)
    ]
    run_events(core, baseline_events, on_frame=on_frame)

    # Phase 2: Stimulation
    current_phase[0] = "stim"
    cx, cy = int(target_center[0]), int(target_center[1])
    slm_mask = make_slm_circle((cx, cy), slm_radius, size=512)
    slm = SLMImage(data=slm_mask, device="SLM")

    stim_events = [
        MDAEvent(channel=ch_kw, exposure=exposure, slm_image=slm, min_start_time=i * stim_interval)
        for i in range(n_stim)
    ]
    run_events(core, stim_events, on_frame=on_frame)

    # Phase 3: Recovery
    current_phase[0] = "recovery"
    core.setSLMPixelsTo("SLM", 0)
    recovery_events = [
        MDAEvent(channel=ch_kw, exposure=exposure, min_start_time=i * recovery_interval)
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
    from ..analysis.intensity import extract_roi_traces

    results = {}

    for phase_key in ["baseline_frames", "stim_frames", "recovery_frames"]:
        frames = data[phase_key]
        if not frames:
            continue
        traces = extract_roi_traces(frames, soma_rois)
        for sid, trace in traces.items():
            if sid not in results:
                results[sid] = {"is_target": (sid == target_id)}
            short_key = phase_key.replace("_frames", "")
            results[sid][f"{short_key}_trace"] = trace
            results[sid][f"{short_key}_mean"] = float(trace.mean())

    # Compute ΔF/F using baseline as F₀
    for _sid, r in results.items():
        bl = r.get("baseline_mean", 0)
        st = r.get("stim_mean", 0)
        if bl > 1:
            r["dff_mean"] = (st - bl) / bl
            st_trace = r.get("stim_trace", np.array([st]))
            r["dff_peak"] = float((st_trace.max() - bl) / bl)
        else:
            r["dff_mean"] = st - bl
            r["dff_peak"] = st - bl

    return results


def connectivity_mapping(
    core,
    soma_positions,
    channel_config="nucleus-channel",
    channel_group="Fake",
    slm_radius=20,
    exposure=50,
    n_post_frames=6,
    post_interval=0.1,
    refractory_wait=4.0,
    soma_roi_radius=15,
    resting_threshold=50,
    response_threshold=120,
):
    """Map directed functional connectivity by stimulating each neuron.

    For each neuron in turn: apply SLM stimulation, capture rapid post-stim
    frames to detect cascade propagation, then wait for refractory period.
    Direct connections are inferred from neurons that respond in the first
    post-stimulation frame (1 synaptic hop delay).

    Works with **real-time** calcium dynamics where time.sleep() creates
    actual delays for cascade propagation.

    Args:
        core: Microscope core connection.
        soma_positions: List of (cx, cy) pixel coordinates for each neuron.
        channel_config: GCaMP channel config name.
        channel_group: Config group name.
        slm_radius: Radius of SLM stimulation circle (pixels).
        exposure: Camera exposure in ms.
        n_post_frames: Number of post-stim frames to capture.
        post_interval: Seconds between post-stim frames (~100ms for 1-hop).
        refractory_wait: Seconds to wait between stimulations.
        soma_roi_radius: Pixel radius for intensity measurement ROI.
        resting_threshold: Max intensity considered "resting" (below = resting).
        response_threshold: Min intensity to be considered "responding".

    Returns:
        dict with:
            adjacency: dict mapping neuron_index -> list of target indices.
            stim_data: dict mapping neuron_index -> {baseline, stim, post} values.
            n_connections: int, total directed edges.
    """
    import time

    from ..hardware.core import make_slm_circle, run_events, snap

    n_neurons = len(soma_positions)
    ch_kw = {"config": channel_config, "group": channel_group}

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
            yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
            mask = ((xx - (ci - x0)) ** 2 + (yy - (ri - y0)) ** 2) <= r**2
            vals.append(float(patch[mask].mean()) if mask.any() else 0.0)
        return vals

    all_data = {}

    for stim_idx in range(n_neurons):
        cx, cy = soma_positions[stim_idx]

        # Baseline snap
        baseline_img = snap(core, channel_config)
        baseline_vals = _measure_somata(baseline_img)

        # SLM stimulation
        slm_mask = make_slm_circle((cx, cy), radius=slm_radius, size=512, core=core)
        slm = SLMImage(data=slm_mask, device="SLM")
        stim_frames = run_events(core, [MDAEvent(channel=ch_kw, exposure=exposure, slm_image=slm)])
        stim_vals = _measure_somata(stim_frames[0][0])

        # Post-stim cascade frames
        post_vals_list = []
        for _ in range(n_post_frames):
            time.sleep(post_interval)
            post_img = snap(core, channel_config)
            post_vals_list.append(_measure_somata(post_img))

        all_data[stim_idx] = {
            "baseline": baseline_vals,
            "stim": stim_vals,
            "post": post_vals_list,
        }

        time.sleep(refractory_wait)

    # Infer connections from cascade timing
    # Direct (1-hop): responds in post-1, begins to decay by post-2
    # Indirect (2-hop): still rising from post-1 to post-2
    adjacency = {i: [] for i in range(n_neurons)}
    for stim_idx in range(n_neurons):
        data = all_data[stim_idx]
        baseline = data["baseline"]
        post = data["post"]
        if not post:
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

            # Must be resting at baseline and activated in post-1
            if bl >= resting_threshold or p1 <= response_threshold:
                continue

            # Timing filter: direct connections peak quickly and start
            # decaying by post-2. Indirect connections are still rising.
            if post2 is not None and target < len(post2):
                p2 = post2[target]
                # If intensity is STILL RISING from post-1 to post-2,
                # this is likely a 2-hop (indirect) connection
                if p2 > p1 * 1.05:  # >5% increase = still rising
                    continue

            adjacency[stim_idx].append(target)

    n_connections = sum(len(v) for v in adjacency.values())

    return {
        "adjacency": adjacency,
        "stim_data": all_data,
        "n_connections": n_connections,
    }
