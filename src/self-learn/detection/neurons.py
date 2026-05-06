"""Neuron detection and morphometry.

Functions for detecting neuron somata, counting synaptic puncta,
estimating neurite length, and classifying neuron types from
MAP2/synaptophysin immunostaining.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, maximum_filter


def detect_somata(map2, sigma=2.0, min_dt=5, filter_size=9):
    """Detect neuron somata from MAP2 immunostaining.

    Parameters
    ----------
    map2 : ndarray
        MAP2 channel image (bright dendrites + somata on dark bg).
    sigma : float
        Threshold = mean + sigma * std.
    min_dt : float
        Minimum distance transform value for soma center.
    filter_size : int
        Size of maximum filter for local maxima detection.

    Returns
    -------
    list of dict
        Each dict has: centroid_px (cx, cy), soma_radius, peak_intensity.
    """
    mask = map2 > (map2.mean() + sigma * map2.std())
    dist = distance_transform_edt(mask)

    local_max = (maximum_filter(dist, size=filter_size) == dist) & (dist > min_dt)
    markers, n_somata = ndimage.label(local_max)

    peak_ys, peak_xs = np.where(local_max)
    somata = []
    for mid in range(1, n_somata + 1):
        idx = np.where(markers[peak_ys, peak_xs] == mid)[0]
        if len(idx) == 0:
            continue
        sy, sx = peak_ys[idx[0]], peak_xs[idx[0]]
        somata.append(
            {
                "centroid_px": (int(sx), int(sy)),
                "soma_radius": float(dist[sy, sx]),
                "peak_intensity": float(map2[sy, sx]),
            }
        )
    return somata


def count_puncta(syn, sigma=2.5, min_area=1, max_area=200):
    """Count synaptic puncta from synaptophysin channel.

    Parameters
    ----------
    syn : ndarray
        Synaptophysin channel (bright puncta on dark bg).
    sigma : float
        Threshold = mean + sigma * std.
    min_area, max_area : int
        Size filter for puncta (exclude somata and noise).

    Returns
    -------
    int
        Number of puncta.
    list of dict
        Each dict has: centroid_px, area_px, peak_intensity.
    """
    mask = syn > (syn.mean() + sigma * syn.std())
    labeled, n_cc = ndimage.label(mask)

    puncta = []
    for lid in range(1, n_cc + 1):
        region = labeled == lid
        area = int(region.sum())
        if min_area <= area <= max_area:
            ys, xs = np.where(region)
            cy, cx = ys.mean(), xs.mean()
            peak = float(syn[region].max())
            puncta.append(
                {
                    "centroid_px": (float(cx), float(cy)),
                    "area_px": area,
                    "peak_intensity": peak,
                }
            )
    return len(puncta), puncta


def estimate_neurite_length(map2, threshold=15):
    """Estimate total neurite length from MAP2 channel.

    Uses area / mean_width approximation since skimage may not be available.

    Parameters
    ----------
    map2 : ndarray
        MAP2 channel image.
    threshold : float
        Intensity threshold for dendrite detection.

    Returns
    -------
    float
        Estimated total neurite length in pixels.
    """
    mask = map2 > threshold
    if not mask.any():
        return 0.0

    dt = distance_transform_edt(mask)
    mean_width = 2 * np.mean(dt[mask])
    if mean_width < 1:
        mean_width = 1.0

    return float(mask.sum() / mean_width)


def count_primary_processes(map2, cx, cy, search_radius=20, threshold=15):
    """Count primary processes emerging from a soma.

    Samples directions radially and counts distinct process exits.

    Parameters
    ----------
    map2 : ndarray
        MAP2 channel image.
    cx, cy : float
        Soma center coordinates (col, row).
    search_radius : int
        Distance from soma to sample.
    threshold : float
        Intensity threshold for MAP2 signal.

    Returns
    -------
    int
        Number of primary processes.
    list of float
        Angles (degrees) of each process.
    """
    h, w = map2.shape
    n_angles = 36
    process_starts = []

    prev_has_signal = False
    for i in range(n_angles):
        angle = i * (2 * np.pi / n_angles)
        py = int(cy + search_radius * np.sin(angle))
        px = int(cx + search_radius * np.cos(angle))

        has_signal = False
        if 0 <= py < h and 0 <= px < w:
            has_signal = map2[py, px] > threshold

        if has_signal and not prev_has_signal:
            process_starts.append(i * (360 / n_angles))

        prev_has_signal = has_signal

    # Handle wrap-around: if first and last are both signal, merge
    if len(process_starts) > 1:
        # Check if the signal wraps from last angle to first
        py = int(cy + search_radius * np.sin(0))
        px = int(cx + search_radius * np.cos(0))
        last_angle = (n_angles - 1) * (2 * np.pi / n_angles)
        py_last = int(cy + search_radius * np.sin(last_angle))
        px_last = int(cx + search_radius * np.cos(last_angle))

        first_signal = 0 <= py < h and 0 <= px < w and map2[py, px] > threshold
        last_signal = 0 <= py_last < h and 0 <= px_last < w and map2[py_last, px_last] > threshold

        if first_signal and last_signal and process_starts[0] == 0:
            process_starts.pop(0)

    return len(process_starts), process_starts


def classify_neuron(map2, cx, cy, soma_radius, search_radius=20, threshold=15):
    """Classify neuron type based on morphology.

    Parameters
    ----------
    map2 : ndarray
        MAP2 channel image.
    cx, cy : float
        Soma center coordinates.
    soma_radius : float
        Soma radius from distance transform.
    search_radius : int
        Distance from soma to sample processes.
    threshold : float
        MAP2 intensity threshold.

    Returns
    -------
    str
        'pyramidal', 'stellate', or 'bipolar'.
    int
        Number of primary processes detected.
    """
    n_processes, angles = count_primary_processes(
        map2, cx, cy, search_radius=search_radius, threshold=threshold
    )

    # Bipolar: exactly 2 processes in roughly opposite directions
    if n_processes == 2 and len(angles) == 2:
        diff = abs(angles[1] - angles[0])
        if diff > 180:
            diff = 360 - diff
        if 120 <= diff <= 240:
            return "bipolar", n_processes

    # Bipolar: small soma + few processes
    if n_processes <= 2 and soma_radius < 5:
        return "bipolar", n_processes

    # Stellate: many processes radiating out (>= 5)
    if n_processes >= 5:
        return "stellate", n_processes

    # Pyramidal: large soma, moderate processes (3-4)
    if soma_radius >= 7 or n_processes <= 4:
        return "pyramidal", n_processes

    return "stellate", n_processes


def count_branch_points(map2, somata=None, sigma=1.0):
    """Count dendrite branch points (bifurcations) per neuron.

    Skeletonizes the MAP2 mask, then traces connected skeleton components
    from each soma to count junction pixels (3+ skeleton neighbors) that
    belong to each neuron's dendritic tree. Uses flood-fill from soma
    positions instead of Voronoi partition to correctly handle overlapping
    dendrites from nearby neurons.

    Parameters
    ----------
    map2 : ndarray
        MAP2 channel image.
    somata : list of dict, optional
        Output of detect_somata(). If None, detects automatically.
    sigma : float
        Threshold = mean + sigma * std for MAP2 mask.

    Returns
    -------
    list of dict
        Each dict has: centroid_px, soma_radius, branch_points, skeleton_pixels.
        Sorted by branch_points descending (most complex first).
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError as err:
        raise ImportError("scikit-image required for count_branch_points") from err

    from scipy.signal import convolve2d

    if somata is None:
        somata = detect_somata(map2, sigma=sigma)

    if not somata:
        return []

    # Threshold and skeletonize
    thresh = map2.mean() + sigma * map2.std()
    mask = map2 > thresh
    skel = skeletonize(mask)

    # Junction pixels: 3+ skeleton neighbors
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    neighbor_count = convolve2d(skel.astype(int), kernel, mode="same")
    junctions = skel & (neighbor_count >= 3)
    junction_labeled, n_junctions = ndimage.label(junctions)

    # Label connected components of the skeleton
    skel_labeled, n_skel_cc = ndimage.label(skel)

    # Assign each skeleton component to the soma whose center falls in or
    # nearest to that component.  For each soma, find which skeleton CC
    # its centroid overlaps (using the thresholded mask expanded by soma radius).
    h, w = map2.shape
    soma_to_cc = {}  # soma_index -> set of skeleton CC labels
    cc_to_soma = {}  # skeleton CC label -> soma_index (first claim)

    for si, s in enumerate(somata):
        cx, cy = s["centroid_px"]
        r = max(int(s["soma_radius"]) + 3, 5)
        # Check skeleton labels within soma region
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        local_labels = skel_labeled[y0:y1, x0:x1]
        cc_ids = set(local_labels[local_labels > 0].tolist())
        soma_to_cc[si] = cc_ids
        for cc in cc_ids:
            if cc not in cc_to_soma:
                cc_to_soma[cc] = si

    # For skeleton CCs not claimed by any soma, assign to nearest soma
    for cc_id in range(1, n_skel_cc + 1):
        if cc_id not in cc_to_soma:
            cc_ys, cc_xs = np.where(skel_labeled == cc_id)
            cc_cx, cc_cy = cc_xs.mean(), cc_ys.mean()
            best_dist = float("inf")
            best_si = 0
            for si, s in enumerate(somata):
                sx, sy = s["centroid_px"]
                d = (sx - cc_cx) ** 2 + (sy - cc_cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_si = si
            cc_to_soma[cc_id] = best_si
            soma_to_cc.setdefault(best_si, set()).add(cc_id)

    # Count skeleton pixels and junction clusters per soma
    skel_per_soma = [0] * len(somata)
    bp_per_soma = [0] * len(somata)

    for cc_id in range(1, n_skel_cc + 1):
        si = cc_to_soma.get(cc_id, 0)
        skel_per_soma[si] += int((skel_labeled == cc_id).sum())

    # Assign each junction cluster to the soma that owns its skeleton CC
    for jid in range(1, n_junctions + 1):
        jys, jxs = np.where(junction_labeled == jid)
        # Find which skeleton CC this junction belongs to
        jy, jx = jys[0], jxs[0]
        cc_id = int(skel_labeled[jy, jx])
        if cc_id > 0 and cc_id in cc_to_soma:
            bp_per_soma[cc_to_soma[cc_id]] += 1
        else:
            # Fallback: assign to nearest soma
            jcx, jcy = float(jxs.mean()), float(jys.mean())
            best_dist = float("inf")
            best_si = 0
            for si, s in enumerate(somata):
                sx, sy = s["centroid_px"]
                d = (sx - jcx) ** 2 + (sy - jcy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_si = si
            bp_per_soma[best_si] += 1

    results = []
    for i, s in enumerate(somata):
        results.append(
            {
                "centroid_px": s["centroid_px"],
                "soma_radius": s["soma_radius"],
                "peak_intensity": s["peak_intensity"],
                "branch_points": bp_per_soma[i],
                "skeleton_pixels": skel_per_soma[i],
            }
        )

    results.sort(key=lambda r: -r["branch_points"])
    return results
