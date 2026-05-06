"""Morphological cell classifier.

Unsupervised and rule-based classification of cells from shape features.
Works with measurement dicts from morphometry.measure_objects().

Functions:
    extract_features     -- Build feature matrix from measurement dicts
    classify_kmeans      -- Unsupervised k-means clustering
    classify_rules       -- Rule-based classification with decision boundaries
    classify_composite   -- Weighted scoring across multiple features
    classify_size_bins   -- Simple size-based binning (small/medium/large)
    feature_importance   -- Rank features by inter-class separability
"""

import numpy as np

# Default feature keys available from measure_objects()
DEFAULT_FEATURES = ("area_um2", "eccentricity", "solidity", "diameter_um", "major_um", "minor_um")


def extract_features(measurements, features=None, normalize=True):
    """Build a feature matrix from measurement dicts.

    Args:
        measurements: List of dicts from morphometry.measure_objects().
        features: Tuple of feature keys to extract. Defaults to
            DEFAULT_FEATURES. Any numeric key in the dicts works.
        normalize: If True, z-score normalize each feature (zero mean,
            unit variance). Recommended for k-means.

    Returns:
        dict with:
            matrix: 2D numpy array (n_cells × n_features).
            features: Tuple of feature names used.
            means: Per-feature means (before normalization).
            stds: Per-feature standard deviations.
    """
    if features is None:
        features = DEFAULT_FEATURES

    n = len(measurements)
    if n == 0:
        return {
            "matrix": np.empty((0, len(features))),
            "features": features,
            "means": np.zeros(len(features)),
            "stds": np.ones(len(features)),
        }

    matrix = np.zeros((n, len(features)))
    for i, m in enumerate(measurements):
        for j, f in enumerate(features):
            matrix[i, j] = float(m.get(f, 0.0))

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant features

    if normalize:
        norm_matrix = (matrix - means) / stds
    else:
        norm_matrix = matrix.copy()

    return {
        "matrix": norm_matrix,
        "features": features,
        "means": means,
        "stds": stds,
    }


def classify_kmeans(measurements, n_classes, features=None, max_iter=100, n_init=10, seed=42):
    """Unsupervised k-means clustering on morphometric features.

    Uses Lloyd's algorithm with multiple random initializations.
    No sklearn dependency — pure numpy implementation.

    Args:
        measurements: List of dicts from morphometry.measure_objects().
        n_classes: Number of clusters (k).
        features: Feature keys to use (default: DEFAULT_FEATURES).
        max_iter: Maximum iterations per initialization.
        n_init: Number of random restarts (best inertia wins).
        seed: Random seed for reproducibility.

    Returns:
        dict with:
            labels: 1D int array of cluster labels (0..k-1).
            centers: 2D array of cluster centers (in normalized space).
            centers_raw: 2D array of cluster centers (in original units).
            inertia: Sum of squared distances to nearest center.
            n_per_class: Dict mapping label → count.
            features: Feature names used.
            class_profiles: Dict mapping label → dict of mean feature values
                (in original units).
    """
    feat = extract_features(measurements, features, normalize=True)
    X = feat["matrix"]
    n = X.shape[0]

    if n == 0 or n_classes < 1:
        return {
            "labels": np.array([], dtype=int),
            "centers": np.empty((0, len(feat["features"]))),
            "centers_raw": np.empty((0, len(feat["features"]))),
            "inertia": 0.0,
            "n_per_class": {},
            "features": feat["features"],
            "class_profiles": {},
        }

    k = min(n_classes, n)
    rng = np.random.RandomState(seed)

    best_labels = None
    best_centers = None
    best_inertia = float("inf")

    for _ in range(n_init):
        # Random initialization (k-means++)
        idx = rng.choice(n, k, replace=False)
        centers = X[idx].copy()

        for _ in range(max_iter):
            # Assign each point to nearest center
            dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = dists.argmin(axis=1)

            # Update centers
            new_centers = np.zeros_like(centers)
            for c in range(k):
                members = X[labels == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)
                else:
                    new_centers[c] = centers[c]

            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers

        inertia = sum(np.sum((X[labels == c] - centers[c]) ** 2) for c in range(k))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    # Sort clusters by size (largest cluster = label 0)
    counts = np.bincount(best_labels, minlength=k)
    order = np.argsort(-counts)
    remap = np.zeros(k, dtype=int)
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label
    best_labels = remap[best_labels]
    best_centers = best_centers[order]

    # Convert centers back to original units
    centers_raw = best_centers * feat["stds"] + feat["means"]

    # Build class profiles
    n_per_class = {}
    class_profiles = {}
    for c in range(k):
        mask = best_labels == c
        n_per_class[c] = int(mask.sum())
        profile = {}
        for j, fname in enumerate(feat["features"]):
            profile[fname] = float(centers_raw[c, j])
        class_profiles[c] = profile

    return {
        "labels": best_labels,
        "centers": best_centers,
        "centers_raw": centers_raw,
        "inertia": float(best_inertia),
        "n_per_class": n_per_class,
        "features": feat["features"],
        "class_profiles": class_profiles,
    }


def classify_rules(measurements, rules):
    """Rule-based classification using decision boundaries.

    Each rule is a dict with 'name' and a list of conditions. A cell
    matches the first rule whose conditions are all satisfied. Cells
    matching no rule get label 'unclassified'.

    Args:
        measurements: List of dicts from morphometry.measure_objects().
        rules: List of dicts, each with:
            name: str — class name.
            conditions: List of (feature_key, operator, value) tuples.
                Operators: '>', '<', '>=', '<=', '==', 'between'.
                For 'between': value is (low, high).

    Returns:
        dict with:
            labels: List of class name strings.
            counts: Dict mapping class name → count.
            indices: Dict mapping class name → list of indices.

    Example:
        rules = [
            {'name': 'lymphocyte', 'conditions': [
                ('diameter_um', '<', 10),
                ('eccentricity', '<', 0.5),
            ]},
            {'name': 'neutrophil', 'conditions': [
                ('diameter_um', 'between', (10, 16)),
            ]},
            {'name': 'monocyte', 'conditions': [
                ('diameter_um', '>', 16),
            ]},
        ]
    """
    labels = []
    indices = {}
    counts = {}

    for i, m in enumerate(measurements):
        assigned = "unclassified"
        for rule in rules:
            if _matches(m, rule["conditions"]):
                assigned = rule["name"]
                break
        labels.append(assigned)
        if assigned not in indices:
            indices[assigned] = []
            counts[assigned] = 0
        indices[assigned].append(i)
        counts[assigned] += 1

    return {"labels": labels, "counts": counts, "indices": indices}


def _matches(measurement, conditions):
    """Check if a measurement satisfies all conditions."""
    for feat_key, op, val in conditions:
        v = measurement.get(feat_key, None)
        if v is None:
            return False
        if op == ">":
            if not (v > val):
                return False
        elif op == "<":
            if not (v < val):
                return False
        elif op == ">=":
            if not (v >= val):
                return False
        elif op == "<=":
            if not (v <= val):
                return False
        elif op == "==":
            if not (v == val):
                return False
        elif op == "between":
            lo, hi = val
            if not (lo <= v <= hi):
                return False
        else:
            raise ValueError(f"Unknown operator: {op}")
    return True


def classify_composite(
    measurements, criteria, threshold=0, positive_label="positive", negative_label="negative"
):
    """Classify cells using a weighted composite score.

    Each criterion adds (or subtracts) a weight when a feature condition
    is met.  The total score determines the classification.  This is useful
    when no single feature separates classes cleanly but the combination does
    (e.g. budding index: low solidity + high eccentricity + large area).

    Args:
        measurements: List of dicts with numeric feature values.
        criteria: List of (feature_key, operator, value, weight) tuples.
            Operators: '>', '<', '>=', '<=', 'between'.
            Weight is added to the score when the condition is satisfied.
        threshold: Score >= threshold → positive_label.  Default 0.
        positive_label: Label for cells meeting the threshold.
        negative_label: Label for cells below the threshold.

    Returns:
        dict with:
            labels: List of str labels.
            scores: List of float scores.
            counts: Dict mapping label → count.
            indices: Dict mapping label → list of indices.
            threshold: The threshold used.

    Example:
        criteria = [
            ('solidity', '<', 0.82, 1),
            ('solidity', '<', 0.76, 1),      # strong concavity
            ('eccentricity', '>', 0.75, 1),   # elongated
            ('aspect_ratio', '>', 1.7, 1),
            ('area_um2', '>', 20, 1),
            ('solidity', '>', 0.92, -1),      # penalty: very round
            ('eccentricity', '<', 0.40, -1),  # penalty: very circular
        ]
        result = classify_composite(measurements, criteria, threshold=2,
                                    positive_label='budding',
                                    negative_label='non_budding')
    """
    labels = []
    scores = []
    indices = {positive_label: [], negative_label: []}
    counts = {positive_label: 0, negative_label: 0}

    for i, m in enumerate(measurements):
        score = 0.0
        for feat_key, op, val, weight in criteria:
            v = m.get(feat_key)
            if v is None:
                continue
            hit = False
            if op == ">":
                hit = v > val
            elif op == "<":
                hit = v < val
            elif op == ">=":
                hit = v >= val
            elif op == "<=":
                hit = v <= val
            elif op == "between":
                lo, hi = val
                hit = lo <= v <= hi
            if hit:
                score += weight

        label = positive_label if score >= threshold else negative_label
        labels.append(label)
        scores.append(round(score, 4))
        indices[label].append(i)
        counts[label] += 1

    return {
        "labels": labels,
        "scores": scores,
        "counts": counts,
        "indices": indices,
        "threshold": threshold,
    }


def classify_size_bins(measurements, key="diameter_um", n_bins=3, bin_names=None):
    """Simple binning by a single measurement dimension.

    Splits the range of values into n_bins equal-frequency bins
    (quantile-based).

    Args:
        measurements: List of dicts from morphometry.measure_objects().
        key: Feature key to bin on.
        n_bins: Number of bins.
        bin_names: List of names for each bin (default: 'small', 'medium',
            'large' for 3 bins, or 'bin_0', 'bin_1', ... for others).

    Returns:
        dict with:
            labels: List of bin name strings.
            counts: Dict mapping bin name → count.
            bin_edges: List of bin edge values (length n_bins + 1).
            bin_names: List of bin names used.
    """
    if bin_names is None:
        if n_bins == 3:
            bin_names = ["small", "medium", "large"]
        elif n_bins == 2:
            bin_names = ["small", "large"]
        else:
            bin_names = [f"bin_{i}" for i in range(n_bins)]

    values = np.array([m[key] for m in measurements])
    n = len(values)

    if n == 0:
        return {
            "labels": [],
            "counts": dict.fromkeys(bin_names, 0),
            "bin_edges": [],
            "bin_names": bin_names,
        }

    # Quantile-based edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = [float(np.percentile(values, q)) for q in quantiles]
    # Ensure last edge includes max
    edges[-1] = float(values.max() + 1e-10)

    labels = []
    counts = dict.fromkeys(bin_names, 0)

    for v in values:
        for b in range(n_bins):
            if edges[b] <= v < edges[b + 1]:
                labels.append(bin_names[b])
                counts[bin_names[b]] += 1
                break
        else:
            # Shouldn't happen, but assign to last bin
            labels.append(bin_names[-1])
            counts[bin_names[-1]] += 1

    return {
        "labels": labels,
        "counts": counts,
        "bin_edges": edges,
        "bin_names": bin_names,
    }


def feature_importance(measurements, labels, features=None):
    """Rank features by inter-class separability.

    Uses the ratio of between-class variance to within-class variance
    (Fisher criterion) for each feature independently.

    Args:
        measurements: List of dicts from morphometry.measure_objects().
        labels: List/array of class labels (int or str) for each measurement.
        features: Feature keys to evaluate (default: DEFAULT_FEATURES).

    Returns:
        dict with:
            ranking: List of (feature_name, score) tuples sorted by
                descending separability score.
            scores: Dict mapping feature_name → Fisher criterion score.
    """
    if features is None:
        features = DEFAULT_FEATURES

    labels_arr = np.asarray(labels)
    classes = np.unique(labels_arr)

    if len(classes) < 2 or len(measurements) < 2:
        scores = dict.fromkeys(features, 0.0)
        ranking = [(f, 0.0) for f in features]
        return {"ranking": ranking, "scores": scores}

    scores = {}
    for fname in features:
        values = np.array([float(m.get(fname, 0.0)) for m in measurements])
        grand_mean = values.mean()

        between = 0.0
        within = 0.0

        for c in classes:
            mask = labels_arr == c
            class_vals = values[mask]
            nc = len(class_vals)
            if nc == 0:
                continue
            class_mean = class_vals.mean()
            between += nc * (class_mean - grand_mean) ** 2
            within += class_vals.var() * nc

        if within > 0:
            scores[fname] = float(between / within)
        else:
            scores[fname] = float("inf") if between > 0 else 0.0

    ranking = sorted(scores.items(), key=lambda x: -x[1])
    return {"ranking": ranking, "scores": scores}
