"""Multi-method detection consensus for robust cell counting.

Runs multiple detection methods and combines results to improve
accuracy. Addresses the recurring under/over-counting issue by
cross-validating different approaches.

Functions:
    consensus_count    -- Run multiple counters and return consensus
    merge_detections   -- Merge overlapping detections from multiple methods
    validate_detections -- Cross-validate detections between two channels
"""

import numpy as np


def consensus_count(image, methods, weights=None):
    """Run multiple detection methods and return consensus count.

    Each method is a callable that takes an image and returns a list
    of detections (dicts with 'cx', 'cy' keys or (x, y) tuples).

    Args:
        image: 2D input image.
        methods: List of callables, each: image → list of detections.
        weights: Optional list of weights per method (default: equal).

    Returns:
        dict with:
            consensus_count: Weighted median count.
            counts: List of individual method counts.
            mean_count: Mean count across methods.
            std_count: Standard deviation of counts.
            agreement: Fraction of methods within 20% of consensus.
            best_method_idx: Index of method closest to consensus.
    """
    if not methods:
        return {
            "consensus_count": 0,
            "counts": [],
            "mean_count": 0.0,
            "std_count": 0.0,
            "agreement": 0.0,
            "best_method_idx": 0,
        }

    counts = []
    for method in methods:
        try:
            detections = method(image)
            counts.append(len(detections))
        except Exception:
            counts.append(0)

    counts_arr = np.array(counts, dtype=np.float64)

    if weights is None:
        weights = np.ones(len(counts))
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    # Weighted median
    sorted_idx = np.argsort(counts_arr)
    cumweight = np.cumsum(weights[sorted_idx])
    median_idx = np.searchsorted(cumweight, 0.5)
    median_idx = min(median_idx, len(sorted_idx) - 1)
    consensus = int(counts_arr[sorted_idx[median_idx]])

    mean_count = float(np.average(counts_arr, weights=weights))
    std_count = float(np.sqrt(np.average((counts_arr - mean_count) ** 2, weights=weights)))

    # Agreement: fraction of methods within 20% of consensus
    if consensus > 0:
        within = sum(1 for c in counts if abs(c - consensus) / consensus < 0.2)
    else:
        within = sum(1 for c in counts if c == 0)
    agreement = within / len(counts)

    # Best method: closest to consensus
    diffs = [abs(c - consensus) for c in counts]
    best_idx = int(np.argmin(diffs))

    return {
        "consensus_count": consensus,
        "counts": counts,
        "mean_count": round(mean_count, 1),
        "std_count": round(std_count, 1),
        "agreement": round(agreement, 3),
        "best_method_idx": best_idx,
    }


def merge_detections(detection_lists, merge_radius=10):
    """Merge overlapping detections from multiple methods.

    Detections within merge_radius pixels of each other are considered
    the same object. Objects detected by more methods are more reliable.

    Args:
        detection_lists: List of lists of (x, y) or dict with 'cx','cy'.
        merge_radius: Maximum distance to consider same object (pixels).

    Returns:
        dict with:
            merged: List of merged detection dicts with
                cx, cy, confidence (fraction of methods that detected it).
            n_merged: Number of unique objects after merging.
            n_input: Total detections across all methods.
    """
    # Collect all detections with method labels
    all_points = []
    all_method_ids = []

    for method_idx, detections in enumerate(detection_lists):
        for det in detections:
            if isinstance(det, dict):
                x, y = det.get("cx", det.get("x", 0)), det.get("cy", det.get("y", 0))
            else:
                x, y = float(det[0]), float(det[1])
            all_points.append((x, y))
            all_method_ids.append(method_idx)

    n_input = len(all_points)
    n_methods = len(detection_lists)

    if n_input == 0:
        return {"merged": [], "n_merged": 0, "n_input": 0}

    points = np.array(all_points)
    methods = np.array(all_method_ids)

    # Greedy clustering
    used = np.zeros(n_input, dtype=bool)
    merged = []

    # Sort by distance to center of mass (process center first)
    center = points.mean(axis=0)
    dist_to_center = np.sqrt(np.sum((points - center) ** 2, axis=1))
    order = np.argsort(dist_to_center)

    for idx in order:
        if used[idx]:
            continue

        # Find all unmerged points within radius
        dists = np.sqrt(np.sum((points - points[idx]) ** 2, axis=1))
        nearby = np.where((dists < merge_radius) & ~used)[0]

        # Mark as used
        used[nearby] = True

        # Compute merged position (mean of nearby)
        group_points = points[nearby]
        group_methods = set(methods[nearby])

        cx = float(group_points[:, 0].mean())
        cy = float(group_points[:, 1].mean())
        confidence = len(group_methods) / n_methods

        merged.append(
            {
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "confidence": round(confidence, 3),
                "n_detections": len(nearby),
            }
        )

    return {
        "merged": merged,
        "n_merged": len(merged),
        "n_input": n_input,
    }


def validate_detections(detections, validation_image, validation_fn, match_radius=10):
    """Cross-validate detections against a second channel or method.

    For each detection, check if validation_fn confirms an object
    at that location in the validation_image.

    Args:
        detections: List of (x, y) or dicts with 'cx', 'cy'.
        validation_image: 2D image for cross-validation.
        validation_fn: Callable(image_patch) → bool.
            Takes a local patch and returns True if an object is confirmed.
        match_radius: Radius of the patch to extract around each detection.

    Returns:
        dict with:
            validated: List of detections that passed validation.
            rejected: List of detections that failed validation.
            validation_rate: Fraction that passed.
    """
    img = np.asarray(validation_image, dtype=np.float64)
    h, w = img.shape[:2]

    validated = []
    rejected = []

    for det in detections:
        if isinstance(det, dict):
            cx = det.get("cx", det.get("x", 0))
            cy = det.get("cy", det.get("y", 0))
        else:
            cx, cy = float(det[0]), float(det[1])

        # Extract patch
        x0 = max(0, int(cx - match_radius))
        x1 = min(w, int(cx + match_radius + 1))
        y0 = max(0, int(cy - match_radius))
        y1 = min(h, int(cy + match_radius + 1))

        if x1 <= x0 or y1 <= y0:
            rejected.append(det)
            continue

        patch = img[y0:y1, x0:x1]

        try:
            if validation_fn(patch):
                validated.append(det)
            else:
                rejected.append(det)
        except Exception:
            rejected.append(det)

    total = len(validated) + len(rejected)
    rate = len(validated) / total if total > 0 else 0.0

    return {
        "validated": validated,
        "rejected": rejected,
        "validation_rate": round(rate, 3),
    }
