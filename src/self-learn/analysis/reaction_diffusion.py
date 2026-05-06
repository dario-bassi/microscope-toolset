"""Reaction-diffusion pattern classification.

Classifies excitable-media / reaction-diffusion patterns into canonical
types (waves, spirals, spots, stripes, mitosis, coral) using both static
morphology features and temporal dynamics. A single static frame is often
ambiguous (mitosis vs coral look similar), so temporal analysis across
3-5 frames is critical for reliable classification.

Functions:
    classify_rd_pattern   -- Full classification pipeline (static + temporal)
    static_features       -- Extract morphological features from a single frame
    temporal_features     -- Extract dynamic features from a frame stack
    component_dynamics    -- Track connected-component count over time
    detect_spiral_arms    -- Detect rotating spiral arms in a stack
"""

import numpy as np
from scipy import ndimage


def classify_rd_pattern(stack, threshold=None):
    """Classify a reaction-diffusion pattern from a timelapse stack.

    Uses static morphology from the middle frame plus temporal dynamics
    across the stack for robust classification. Handles single frames
    (less reliable) and multi-frame stacks (preferred).

    Args:
        stack: 2D array (single frame) or 3D array (T, H, W) timelapse.
        threshold: float or None. Binarization threshold for active regions.
            If None, uses Otsu's method on the middle frame.

    Returns:
        dict with:
            pattern: str, one of 'waves', 'spirals', 'spots', 'stripes',
                'mitosis', 'coral'.
            confidence: float 0-1, classification confidence.
            static: dict, static morphological features.
            temporal: dict or None, temporal features (None if single frame).
            scores: dict, per-pattern score breakdown.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[np.newaxis]

    n_frames = stack.shape[0]
    mid = n_frames // 2

    # Threshold
    if threshold is None:
        threshold = _otsu_threshold(stack[mid])

    # Extract features
    sf = static_features(stack[mid], threshold)

    tf = None
    if n_frames >= 3:
        tf = temporal_features(stack, threshold)

    # Score each pattern
    scores = _compute_scores(sf, tf)

    # Pick winner
    best = max(scores, key=scores.get)
    best_score = scores[best]
    second = sorted(scores.values(), reverse=True)[1]
    confidence = min(1.0, (best_score - second) / max(best_score, 0.01) + 0.5)
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    return {
        "pattern": best,
        "confidence": confidence,
        "static": sf,
        "temporal": tf,
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }


def static_features(image, threshold=None):
    """Extract morphological features from a single frame.

    Args:
        image: 2D array.
        threshold: float or None. If None, uses Otsu.

    Returns:
        dict with:
            coverage: float, fraction of active pixels.
            n_components: int, number of connected components.
            mean_area: float, mean component area in pixels.
            mean_eccentricity: float, mean eccentricity (0=round, 1=line).
            elongation_ratio: float, max_axis/min_axis of largest component.
            branching_score: float, skeleton branch points / skeleton length.
            compactness: float, mean area / perimeter^2 (high = round).
            euler_number: int, topology measure (1=solid, <1=has holes).
    """
    image = np.asarray(image, dtype=float)
    if threshold is None:
        threshold = _otsu_threshold(image)

    binary = image > threshold
    coverage = float(binary.sum()) / binary.size

    labeled, n_cc = ndimage.label(binary)

    if n_cc == 0:
        return {
            "coverage": coverage,
            "n_components": 0,
            "mean_area": 0.0,
            "mean_eccentricity": 0.0,
            "elongation_ratio": 1.0,
            "branching_score": 0.0,
            "compactness": 0.0,
            "euler_number": 0,
        }

    # Component properties
    areas = ndimage.sum(binary, labeled, range(1, n_cc + 1))
    areas = np.array(areas)

    # Eccentricity via second moments
    eccentricities = []
    for i in range(1, n_cc + 1):
        mask = labeled == i
        if mask.sum() < 5:
            continue
        ys, xs = np.where(mask)
        if len(ys) < 2:
            eccentricities.append(0.0)
            continue
        cov = np.cov(xs.astype(float), ys.astype(float))
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 0)
        if eigvals[-1] > 0:
            ecc = float(np.sqrt(1 - eigvals[0] / eigvals[-1]))
        else:
            ecc = 0.0
        eccentricities.append(ecc)

    mean_ecc = float(np.mean(eccentricities)) if eccentricities else 0.0

    # Elongation of largest component
    largest_idx = int(np.argmax(areas)) + 1
    largest_mask = labeled == largest_idx
    ys, xs = np.where(largest_mask)
    if len(ys) > 2:
        cov = np.cov(xs.astype(float), ys.astype(float))
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        elongation = float(np.sqrt(eigvals[-1] / eigvals[0]))
    else:
        elongation = 1.0

    # Branching score via skeleton
    branching = _branching_score(binary)

    # Compactness: 4π·area / perimeter²
    perimeters = []
    for i in range(1, n_cc + 1):
        mask = labeled == i
        eroded = ndimage.binary_erosion(mask)
        perimeter = float((mask & ~eroded).sum())
        perimeters.append(max(perimeter, 1))
    compactness_vals = [4 * np.pi * a / (p * p) for a, p in zip(areas, perimeters, strict=False)]
    mean_compact = float(np.mean(compactness_vals))

    # Euler number (topology)
    euler = _euler_number_2d(binary)

    return {
        "coverage": round(coverage, 4),
        "n_components": int(n_cc),
        "mean_area": round(float(np.mean(areas)), 2),
        "mean_eccentricity": round(mean_ecc, 4),
        "elongation_ratio": round(elongation, 2),
        "branching_score": round(branching, 4),
        "compactness": round(mean_compact, 4),
        "euler_number": int(euler),
    }


def temporal_features(stack, threshold=None):
    """Extract dynamic features from a timelapse stack.

    Args:
        stack: 3D array (T, H, W).
        threshold: float or None. Applied uniformly across all frames.

    Returns:
        dict with:
            component_counts: list of int, CC count per frame.
            count_trend: str, 'increasing'/'decreasing'/'stable'.
            count_change_rate: float, mean change in CC count per frame.
            mean_displacement: float, mean pixel displacement between frames.
            has_traveling_front: bool, whether a propagating wavefront is detected.
            has_rotation: bool, whether spiral rotation is detected.
            coverage_trend: str, 'increasing'/'decreasing'/'stable'.
    """
    stack = np.asarray(stack, dtype=float)
    n_frames = stack.shape[0]

    if threshold is None:
        threshold = _otsu_threshold(stack[n_frames // 2])

    # Component counts
    cd = component_dynamics(stack, threshold)
    counts = cd["counts"]

    # Count trend
    if len(counts) >= 3:
        x = np.arange(len(counts), dtype=float)
        slope = np.polyfit(x, counts, 1)[0]
        mean_count = np.mean(counts)
        if abs(slope) < max(0.5, mean_count * 0.05):
            count_trend = "stable"
        elif slope > 0:
            count_trend = "increasing"
        else:
            count_trend = "decreasing"
        count_rate = float(slope)
    else:
        count_trend = "stable"
        count_rate = 0.0

    # Mean displacement (optical flow approximation)
    displacements = []
    for i in range(1, n_frames):
        diff = np.abs(stack[i] - stack[i - 1])
        displacements.append(float(diff.mean()))
    mean_disp = float(np.mean(displacements)) if displacements else 0.0

    # Traveling front detection
    has_front = _detect_traveling_front(stack, threshold)

    # Rotation detection
    has_rot = False
    if n_frames >= 3:
        spiral_result = detect_spiral_arms(stack, threshold)
        has_rot = spiral_result["has_rotation"]

    # Coverage trend
    coverages = []
    for i in range(n_frames):
        binary = stack[i] > threshold
        coverages.append(float(binary.sum()) / binary.size)

    if len(coverages) >= 3:
        x = np.arange(len(coverages), dtype=float)
        cov_slope = np.polyfit(x, coverages, 1)[0]
        mean_cov = np.mean(coverages)
        if abs(cov_slope) < max(0.01, mean_cov * 0.05):
            cov_trend = "stable"
        elif cov_slope > 0:
            cov_trend = "increasing"
        else:
            cov_trend = "decreasing"
    else:
        cov_trend = "stable"

    return {
        "component_counts": counts,
        "count_trend": count_trend,
        "count_change_rate": round(count_rate, 3),
        "mean_displacement": round(mean_disp, 3),
        "has_traveling_front": has_front,
        "has_rotation": has_rot,
        "coverage_trend": cov_trend,
    }


def component_dynamics(stack, threshold=None):
    """Track connected-component count over time.

    Critical for distinguishing mitosis (count increases as spots split)
    from coral (count stable, structures grow from tips).

    Args:
        stack: 3D array (T, H, W).
        threshold: float or None.

    Returns:
        dict with:
            counts: list of int, CC count per frame.
            mean_count: float.
            std_count: float.
            total_change: int, last - first count.
            monotonic: bool, whether count is mostly monotonic.
    """
    stack = np.asarray(stack, dtype=float)
    n_frames = stack.shape[0]

    if threshold is None:
        threshold = _otsu_threshold(stack[n_frames // 2])

    counts = []
    for i in range(n_frames):
        binary = stack[i] > threshold
        # Remove tiny noise
        binary = ndimage.binary_opening(binary, iterations=1)
        _, n_cc = ndimage.label(binary)
        counts.append(int(n_cc))

    counts_arr = np.array(counts)
    diffs = np.diff(counts_arr)
    n_increasing = int(np.sum(diffs > 0))
    n_decreasing = int(np.sum(diffs < 0))
    dominant = max(n_increasing, n_decreasing)
    mono = dominant >= 0.7 * len(diffs) if len(diffs) > 0 else True

    return {
        "counts": counts,
        "mean_count": round(float(np.mean(counts_arr)), 2),
        "std_count": round(float(np.std(counts_arr)), 2),
        "total_change": int(counts_arr[-1] - counts_arr[0]) if len(counts_arr) > 1 else 0,
        "monotonic": bool(mono),
    }


def detect_spiral_arms(stack, threshold=None):
    """Detect rotating spiral arms in a timelapse.

    Uses angular intensity profile around the image center. Spiral
    rotation manifests as a consistent angular shift of intensity
    peaks between frames.

    Args:
        stack: 3D array (T, H, W).
        threshold: float or None.

    Returns:
        dict with:
            has_rotation: bool.
            rotation_rate: float, degrees per frame (0 if no rotation).
            n_arms: int, estimated spiral arm count.
    """
    stack = np.asarray(stack, dtype=float)
    n_frames = stack.shape[0]
    h, w = stack.shape[1], stack.shape[2]
    cy, cx = h // 2, w // 2

    n_angles = 72  # 5-degree bins
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Build angular profiles
    ys, xs = np.mgrid[:h, :w]
    angle_map = np.arctan2(ys - cy, xs - cx) % (2 * np.pi)
    radius_map = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = min(h, w) // 3
    ring_mask = (radius_map > max_r * 0.3) & (radius_map < max_r)

    profiles = []
    for i in range(n_frames):
        profile = np.zeros(n_angles)
        for j in range(n_angles):
            lo = angles[j]
            hi = lo + 2 * np.pi / n_angles
            sector = ring_mask & (angle_map >= lo) & (angle_map < hi)
            if sector.sum() > 0:
                profile[j] = float(np.mean(stack[i][sector]))
        profiles.append(profile)

    if n_frames < 2:
        return {"has_rotation": False, "rotation_rate": 0.0, "n_arms": 0}

    # Check if frames actually differ (identical frames = no rotation)
    frame_diffs = []
    for i in range(1, n_frames):
        diff = float(np.mean(np.abs(profiles[i] - profiles[i - 1])))
        frame_diffs.append(diff)
    mean_diff = float(np.mean(frame_diffs)) if frame_diffs else 0

    # If profiles are essentially identical, no rotation
    profile_range = float(np.ptp(profiles[n_frames // 2]))
    if mean_diff < profile_range * 0.05 or profile_range < 1e-6:
        return {"has_rotation": False, "rotation_rate": 0.0, "n_arms": 0}

    # Cross-correlate consecutive angular profiles
    shifts = []
    peak_qualities = []
    for i in range(1, n_frames):
        cc = np.correlate(np.tile(profiles[i], 3), profiles[i - 1], mode="valid")
        # cc length = 2*n_angles+1, peak at n_angles = zero shift
        peak = int(np.argmax(cc))
        shift = peak - n_angles  # in bins
        shifts.append(shift)
        # Quality: how much better is peak than mean
        cc_range = float(np.ptp(cc))
        if cc_range > 0:
            peak_qualities.append(float((cc[peak] - np.mean(cc)) / cc_range))
        else:
            peak_qualities.append(0.0)

    mean_shift = float(np.mean(shifts))
    std_shift = float(np.std(shifts))
    mean_quality = float(np.mean(peak_qualities))

    # Require: consistent non-zero shift, low variance, and good correlation quality
    # Random spots produce poor cross-correlation quality and inconsistent shifts
    has_rot = (
        abs(mean_shift) > 0.5 and std_shift < max(abs(mean_shift) * 0.5, 1.5) and mean_quality > 0.4
    )

    # Count arms from angular profile peaks
    mid_profile = profiles[n_frames // 2]
    mid_profile = mid_profile - np.mean(mid_profile)
    peaks = []
    for j in range(n_angles):
        prev = mid_profile[(j - 1) % n_angles]
        curr = mid_profile[j]
        nxt = mid_profile[(j + 1) % n_angles]
        if curr > prev and curr > nxt and curr > np.std(mid_profile) * 0.5:
            peaks.append(j)
    n_arms = len(peaks)

    return {
        "has_rotation": bool(has_rot),
        "rotation_rate": round(mean_shift * 360.0 / n_angles, 2),
        "n_arms": int(n_arms),
    }


# ── Private helpers ──────────────────────────────────────────────────


def _otsu_threshold(image):
    """Simple Otsu threshold for binarization."""
    image = np.asarray(image, dtype=float)
    vals = image.ravel()
    n_bins = 256
    hist, edges = np.histogram(vals, bins=n_bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return float(np.mean(vals))

    best_thresh = float(bin_centers[0])
    best_var = 0

    w0 = 0
    sum0 = 0
    total_sum = float(np.sum(hist * bin_centers))

    for i in range(n_bins):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * bin_centers[i]
        mean0 = sum0 / w0
        mean1 = (total_sum - sum0) / w1
        var_between = w0 * w1 * (mean0 - mean1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = float(bin_centers[i])

    return best_thresh


def _branching_score(binary):
    """Compute branch points / skeleton length."""
    from skimage.morphology import skeletonize

    skel = skeletonize(binary)
    skel_length = int(skel.sum())
    if skel_length < 5:
        return 0.0

    # Branch points: pixels with 3+ neighbors
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    neighbors = ndimage.convolve(skel.astype(int), kernel)
    branch_pts = int(((neighbors >= 3) & skel).sum())

    return float(branch_pts) / skel_length


def _euler_number_2d(binary):
    """Compute Euler number (objects - holes) for a 2D binary image."""
    labeled, n_obj = ndimage.label(binary)
    inv_labeled, n_holes = ndimage.label(~binary)
    # Subtract 1 for the background component
    n_holes = max(0, n_holes - 1)
    return n_obj - n_holes


def _detect_traveling_front(stack, threshold):
    """Detect if there's a propagating wavefront."""
    n_frames = stack.shape[0]
    if n_frames < 3:
        return False

    _h, _w = stack.shape[1], stack.shape[2]

    # Track center of mass of active region
    coms = []
    for i in range(n_frames):
        binary = stack[i] > threshold
        if binary.sum() > 0:
            cy, cx = ndimage.center_of_mass(binary)
            coms.append((float(cy), float(cx)))
        else:
            coms.append(None)

    valid = [c for c in coms if c is not None]
    if len(valid) < 3:
        return False

    # Check if COM moves consistently in one direction
    displacements = []
    for i in range(1, len(valid)):
        dy = valid[i][0] - valid[i - 1][0]
        dx = valid[i][1] - valid[i - 1][1]
        displacements.append((dy, dx))

    if not displacements:
        return False

    mean_dy = np.mean([d[0] for d in displacements])
    mean_dx = np.mean([d[1] for d in displacements])
    speed = np.sqrt(mean_dy**2 + mean_dx**2)

    # Directional consistency
    if speed < 1.0:
        return False

    angles = [np.arctan2(d[0], d[1]) for d in displacements]
    angle_std = float(np.std(angles))

    return angle_std < 1.0 and speed > 2.0


def _compute_scores(sf, tf):
    """Score each pattern type from features."""
    scores = {
        "waves": 0.0,
        "spirals": 0.0,
        "spots": 0.0,
        "stripes": 0.0,
        "mitosis": 0.0,
        "coral": 0.0,
    }

    coverage = sf["coverage"]
    n_cc = sf["n_components"]
    ecc = sf["mean_eccentricity"]
    elong = sf["elongation_ratio"]
    branch = sf["branching_score"]
    compact = sf["compactness"]

    # ── Spots: sparse, round, isolated ──
    if coverage < 0.2:
        scores["spots"] += 2.0
    elif coverage < 0.35:
        scores["spots"] += 1.0
    if compact > 0.3:
        scores["spots"] += 1.0
    if n_cc > 5 and ecc < 0.5:
        scores["spots"] += 1.0

    # ── Stripes: elongated, parallel, LOW branching ──
    if ecc > 0.7 and branch < 0.05:
        scores["stripes"] += 2.0
    elif ecc > 0.7:
        scores["stripes"] += 0.5  # branching reduces stripe likelihood
    if elong > 3 and branch < 0.05:
        scores["stripes"] += 1.5
    if 0.3 < coverage < 0.6 and branch < 0.05:
        scores["stripes"] += 0.5

    # ── Waves: traveling front, moderate coverage ──
    if 0.05 < coverage < 0.4:
        scores["waves"] += 0.5
    if n_cc < 5 and elong > 2:
        scores["waves"] += 0.5

    # ── Spirals: rotating arms ──
    if branch > 0.01:
        scores["spirals"] += 0.3

    # ── Mitosis: moderate coverage, many components ──
    if 0.2 < coverage < 0.6:
        scores["mitosis"] += 0.5
    if n_cc > 10:
        scores["mitosis"] += 0.5
    if branch > 0.02:
        scores["mitosis"] += 0.3

    # ── Coral: high coverage, branching, many components ──
    if coverage > 0.3:
        scores["coral"] += 1.0
    if branch > 0.02:
        scores["coral"] += 1.5
    if branch > 0.05:
        scores["coral"] += 1.0  # strong branching is very coral-like
    if coverage > 0.3 and branch > 0.02:
        scores["coral"] += 1.0  # combo bonus

    # ── Temporal features (high weight — critical for disambiguation) ──
    if tf is not None:
        # Traveling front → waves
        if tf["has_traveling_front"]:
            scores["waves"] += 3.0

        # Rotation → spirals
        if tf["has_rotation"]:
            scores["spirals"] += 4.0

        # Component count increasing → mitosis (spots splitting)
        if tf["count_trend"] == "increasing" and tf["count_change_rate"] > 0.5:
            scores["mitosis"] += 4.0
            scores["coral"] -= 1.0
            scores["spots"] -= 1.0  # static spots don't increase in count

        # Stable component count + high coverage → coral
        if tf["count_trend"] == "stable" and coverage > 0.35:
            scores["coral"] += 2.0
            scores["mitosis"] -= 1.0

        # Stable + low coverage → static spots
        if tf["count_trend"] == "stable" and coverage < 0.2:
            scores["spots"] += 2.0

        # High displacement → dynamic pattern (waves/spirals)
        if tf["mean_displacement"] > 5:
            scores["waves"] += 0.5
            scores["spirals"] += 0.5
        elif tf["mean_displacement"] < 1:
            scores["spots"] += 1.0
            scores["stripes"] += 0.5

    # Ensure no negative scores
    for k in scores:
        scores[k] = max(0.0, scores[k])

    return scores
