"""Morphological cell classifier.

Unsupervised and rule-based classification of cells from shape features.
Works with measurement dicts from morphometry.measure_objects().

Functions:
    extract_features     -- Build feature matrix from measurement dicts
    classify_kmeans      -- Unsupervised k-means clustering
    classify_gmm         -- Gaussian mixture model (soft clustering, BIC)
    classify_dbscan      -- Density-based clustering (outlier-aware)
    classify_rules       -- Rule-based classification with decision boundaries
    classify_composite   -- Weighted scoring across multiple features
    classify_size_bins   -- Simple size-based binning (small/medium/large)
    multi_otsu           -- Multi-level Otsu threshold for 1D data
    feature_importance   -- Rank features by inter-class separability
"""

import numpy as np


# Default feature keys available from measure_objects()
DEFAULT_FEATURES = ('area_um2', 'eccentricity', 'solidity', 'diameter_um',
                    'major_um', 'minor_um')

# Extended features including intensity and shape (when intensity_image provided)
EXTENDED_FEATURES = ('area_um2', 'eccentricity', 'solidity', 'circularity',
                     'aspect_ratio', 'mean_intensity', 'std_intensity')


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
            'matrix': np.empty((0, len(features))),
            'features': features,
            'means': np.zeros(len(features)),
            'stds': np.ones(len(features)),
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
        'matrix': norm_matrix,
        'features': features,
        'means': means,
        'stds': stds,
    }


def classify_kmeans(measurements, n_classes, features=None, max_iter=100,
                    n_init=10, seed=42):
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
    X = feat['matrix']
    n = X.shape[0]

    if n == 0 or n_classes < 1:
        return {
            'labels': np.array([], dtype=int),
            'centers': np.empty((0, len(feat['features']))),
            'centers_raw': np.empty((0, len(feat['features']))),
            'inertia': 0.0,
            'n_per_class': {},
            'features': feat['features'],
            'class_profiles': {},
        }

    k = min(n_classes, n)
    rng = np.random.RandomState(seed)

    best_labels = None
    best_centers = None
    best_inertia = float('inf')

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

        inertia = sum(
            np.sum((X[labels == c] - centers[c]) ** 2)
            for c in range(k)
        )

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
    centers_raw = best_centers * feat['stds'] + feat['means']

    # Build class profiles
    n_per_class = {}
    class_profiles = {}
    for c in range(k):
        mask = best_labels == c
        n_per_class[c] = int(mask.sum())
        profile = {}
        for j, fname in enumerate(feat['features']):
            profile[fname] = float(centers_raw[c, j])
        class_profiles[c] = profile

    return {
        'labels': best_labels,
        'centers': best_centers,
        'centers_raw': centers_raw,
        'inertia': float(best_inertia),
        'n_per_class': n_per_class,
        'features': feat['features'],
        'class_profiles': class_profiles,
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
        assigned = 'unclassified'
        for rule in rules:
            if _matches(m, rule['conditions']):
                assigned = rule['name']
                break
        labels.append(assigned)
        if assigned not in indices:
            indices[assigned] = []
            counts[assigned] = 0
        indices[assigned].append(i)
        counts[assigned] += 1

    return {'labels': labels, 'counts': counts, 'indices': indices}


def _matches(measurement, conditions):
    """Check if a measurement satisfies all conditions."""
    for feat_key, op, val in conditions:
        v = measurement.get(feat_key, None)
        if v is None:
            return False
        if op == '>':
            if not (v > val):
                return False
        elif op == '<':
            if not (v < val):
                return False
        elif op == '>=':
            if not (v >= val):
                return False
        elif op == '<=':
            if not (v <= val):
                return False
        elif op == '==':
            if not (v == val):
                return False
        elif op == 'between':
            lo, hi = val
            if not (lo <= v <= hi):
                return False
        else:
            raise ValueError(f"Unknown operator: {op}")
    return True


def classify_composite(measurements, criteria, threshold=0,
                       positive_label='positive', negative_label='negative'):
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
            if op == '>':
                hit = v > val
            elif op == '<':
                hit = v < val
            elif op == '>=':
                hit = v >= val
            elif op == '<=':
                hit = v <= val
            elif op == 'between':
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
        'labels': labels,
        'scores': scores,
        'counts': counts,
        'indices': indices,
        'threshold': threshold,
    }


def classify_size_bins(measurements, key='diameter_um', n_bins=3,
                       bin_names=None):
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
            bin_names = ['small', 'medium', 'large']
        elif n_bins == 2:
            bin_names = ['small', 'large']
        else:
            bin_names = [f'bin_{i}' for i in range(n_bins)]

    values = np.array([m[key] for m in measurements])
    n = len(values)

    if n == 0:
        return {
            'labels': [],
            'counts': {name: 0 for name in bin_names},
            'bin_edges': [],
            'bin_names': bin_names,
        }

    # Quantile-based edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = [float(np.percentile(values, q)) for q in quantiles]
    # Ensure last edge includes max
    edges[-1] = float(values.max() + 1e-10)

    labels = []
    counts = {name: 0 for name in bin_names}

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
        'labels': labels,
        'counts': counts,
        'bin_edges': edges,
        'bin_names': bin_names,
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
        scores = {f: 0.0 for f in features}
        ranking = [(f, 0.0) for f in features]
        return {'ranking': ranking, 'scores': scores}

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
            scores[fname] = float('inf') if between > 0 else 0.0

    ranking = sorted(scores.items(), key=lambda x: -x[1])
    return {'ranking': ranking, 'scores': scores}


# ---------------------------------------------------------------------------
# Gaussian Mixture Model (EM algorithm)
# ---------------------------------------------------------------------------

def classify_gmm(measurements, n_classes='auto', features=None,
                 max_classes=6, max_iter=100, tol=1e-4, n_init=5, seed=42):
    """Gaussian Mixture Model classification with optional BIC-based k selection.

    Fits a mixture of Gaussians using the EM algorithm. Each component
    has its own mean and full covariance matrix. When n_classes='auto',
    uses BIC to select the optimal number of components.

    This is better than k-means when clusters have different shapes/sizes
    or when soft (probabilistic) assignments are needed — e.g. FUCCI
    phase classification where cells near boundaries are ambiguous.

    Args:
        measurements: List of dicts with numeric feature values.
        n_classes: Number of Gaussian components, or 'auto' to select
            via BIC (tests 2..max_classes).
        features: Feature keys to use (default: DEFAULT_FEATURES).
        max_classes: Maximum k to test when n_classes='auto'.
        max_iter: Maximum EM iterations.
        tol: Convergence threshold on log-likelihood change.
        n_init: Number of random restarts (best log-likelihood wins).
        seed: Random seed.

    Returns:
        dict with:
            labels: 1D int array of hard class assignments.
            probabilities: 2D array (n_cells × k) of soft assignments.
            means: 2D array (k × d) of component means (normalized space).
            means_raw: 2D array (k × d) of component means (original units).
            covariances: List of k covariance matrices.
            weights: 1D array (k,) of mixing weights.
            bic: BIC score of the selected model.
            n_classes: Number of classes in the final model.
            class_profiles: Dict mapping label → dict of mean feature values.
    """
    feat = extract_features(measurements, features, normalize=True)
    X = feat['matrix']
    n, d = X.shape

    if n < 2:
        return _empty_gmm_result(feat, n)

    if n_classes == 'auto':
        best_k, best_bic = 2, float('inf')
        for k in range(2, min(max_classes + 1, n)):
            _, bic = _fit_gmm_best(X, k, max_iter, tol, n_init, seed)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        k = best_k
    else:
        k = min(int(n_classes), n)

    params, bic = _fit_gmm_best(X, k, max_iter, tol, n_init, seed)

    # Compute responsibilities for final assignment
    resp = _gmm_e_step(X, params['means'], params['covariances'],
                       params['weights'])
    labels = resp.argmax(axis=1)

    # Sort by cluster size (largest first)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    remap = np.zeros(k, dtype=int)
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label
    labels = remap[labels]
    resp = resp[:, order]
    means = params['means'][order]
    covs = [params['covariances'][i] for i in order]
    weights = params['weights'][order]

    # Convert means back to original units
    means_raw = means * feat['stds'] + feat['means']

    # Build class profiles
    class_profiles = {}
    for c in range(k):
        profile = {}
        for j, fname in enumerate(feat['features']):
            profile[fname] = float(means_raw[c, j])
        class_profiles[c] = profile

    return {
        'labels': labels,
        'probabilities': resp,
        'means': means,
        'means_raw': means_raw,
        'covariances': covs,
        'weights': weights,
        'bic': float(bic),
        'n_classes': k,
        'features': feat['features'],
        'class_profiles': class_profiles,
    }


def _empty_gmm_result(feat, n):
    """Return empty GMM result for degenerate cases."""
    return {
        'labels': np.zeros(n, dtype=int),
        'probabilities': np.ones((n, 1)),
        'means': np.empty((0, len(feat['features']))),
        'means_raw': np.empty((0, len(feat['features']))),
        'covariances': [],
        'weights': np.array([1.0]),
        'bic': 0.0,
        'n_classes': 1,
        'features': feat['features'],
        'class_profiles': {},
    }


def _fit_gmm_best(X, k, max_iter, tol, n_init, seed):
    """Fit a k-component GMM with multiple restarts. Returns best (params, bic)."""
    best_params, best_bic = None, float('inf')
    for i in range(n_init):
        params, bic = _fit_gmm(X, k, max_iter, tol, seed + i)
        if bic < best_bic:
            best_bic = bic
            best_params = params
    return best_params, best_bic


def _fit_gmm(X, k, max_iter, tol, seed):
    """Fit a k-component GMM via EM. Returns (params, bic)."""
    n, d = X.shape
    rng = np.random.RandomState(seed)

    # Initialize with k-means-like seeding
    idx = rng.choice(n, k, replace=False)
    means = X[idx].copy()
    covariances = [np.eye(d) for _ in range(k)]
    weights = np.ones(k) / k

    prev_ll = -float('inf')
    for _ in range(max_iter):
        # E-step
        resp = _gmm_e_step(X, means, covariances, weights)

        # M-step
        Nk = resp.sum(axis=0) + 1e-10  # avoid zero
        weights = Nk / n
        for c in range(k):
            rc = resp[:, c:c+1]
            means[c] = (rc * X).sum(axis=0) / Nk[c]
            diff = X - means[c]
            covariances[c] = (diff * rc).T @ diff / Nk[c]
            # Regularize to prevent singularity
            covariances[c] += 1e-6 * np.eye(d)

        # Log-likelihood
        ll = _gmm_log_likelihood(X, means, covariances, weights)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # BIC = -2 * ll + n_params * log(n)
    n_params = k * d + k * d * (d + 1) / 2 + (k - 1)  # means + covs + weights
    bic = -2 * prev_ll + n_params * np.log(n)

    return {'means': means, 'covariances': covariances, 'weights': weights}, bic


def _gmm_e_step(X, means, covariances, weights):
    """Compute responsibilities (posterior class probabilities)."""
    n = X.shape[0]
    k = len(means)
    log_resp = np.zeros((n, k))

    for c in range(k):
        log_resp[:, c] = _log_mvn_pdf(X, means[c], covariances[c]) + np.log(
            weights[c] + 1e-300)

    # Normalize in log-space for stability
    log_max = log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp - log_max)
    resp /= resp.sum(axis=1, keepdims=True) + 1e-300
    return resp


def _log_mvn_pdf(X, mean, cov):
    """Log-density of multivariate normal."""
    d = len(mean)
    diff = X - mean
    try:
        L = np.linalg.cholesky(cov)
        solve = np.linalg.solve(L, diff.T).T
        maha = np.sum(solve ** 2, axis=1)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # Fallback for non-PD matrices
        inv_cov = np.linalg.pinv(cov)
        maha = np.sum(diff @ inv_cov * diff, axis=1)
        sign, log_det = np.linalg.slogdet(cov)
        log_det = log_det if sign > 0 else 0.0
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


def _gmm_log_likelihood(X, means, covariances, weights):
    """Total log-likelihood of the data."""
    n = X.shape[0]
    k = len(means)
    log_probs = np.zeros((n, k))
    for c in range(k):
        log_probs[:, c] = _log_mvn_pdf(X, means[c], covariances[c]) + np.log(
            weights[c] + 1e-300)
    # log-sum-exp
    max_lp = log_probs.max(axis=1)
    return float(np.sum(max_lp + np.log(np.exp(log_probs - max_lp[:, None]).sum(axis=1))))


# ---------------------------------------------------------------------------
# DBSCAN — density-based clustering
# ---------------------------------------------------------------------------

def classify_dbscan(measurements, features=None, eps='auto', min_samples=5):
    """Density-based clustering (DBSCAN) on measurement features.

    Identifies clusters as dense regions separated by sparse regions.
    Points in sparse regions are labeled as outliers (-1). Unlike k-means,
    DBSCAN doesn't need k in advance and naturally handles outliers.

    Good for identifying anomalous cells, debris, or finding natural
    clusters in heterogeneous populations.

    Args:
        measurements: List of dicts with numeric feature values.
        features: Feature keys (default: DEFAULT_FEATURES).
        eps: Neighborhood radius. 'auto' estimates from the data using
            the k-distance heuristic (elbow of sorted k-NN distances).
        min_samples: Minimum points to form a dense region.

    Returns:
        dict with:
            labels: 1D int array (-1 for outliers, 0+ for clusters).
            n_clusters: Number of clusters found (excluding outliers).
            n_outliers: Number of outlier points.
            cluster_sizes: Dict mapping cluster label → count.
            outlier_indices: List of outlier point indices.
            eps_used: The eps value used (useful when eps='auto').
    """
    feat = extract_features(measurements, features, normalize=True)
    X = feat['matrix']
    n = X.shape[0]

    if n < min_samples:
        return {
            'labels': np.full(n, -1, dtype=int),
            'n_clusters': 0,
            'n_outliers': n,
            'cluster_sizes': {},
            'outlier_indices': list(range(n)),
            'eps_used': 0.0,
        }

    # Compute pairwise distances
    dists = _pairwise_distances(X)

    if eps == 'auto':
        eps = _estimate_eps(dists, min_samples)

    # Core points: have >= min_samples neighbors within eps
    labels = np.full(n, -1, dtype=int)
    neighbors = [np.where(dists[i] <= eps)[0] for i in range(n)]
    is_core = np.array([len(nb) >= min_samples for nb in neighbors])

    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if visited[i] or not is_core[i]:
            continue

        # BFS expansion from core point i
        queue = [i]
        visited[i] = True
        labels[i] = cluster_id

        while queue:
            pt = queue.pop(0)
            for nb in neighbors[pt]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                if not visited[nb]:
                    visited[nb] = True
                    if is_core[nb]:
                        queue.append(nb)

        cluster_id += 1

    n_clusters = cluster_id
    outlier_mask = labels == -1
    cluster_sizes = {}
    for c in range(n_clusters):
        cluster_sizes[c] = int((labels == c).sum())

    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_outliers': int(outlier_mask.sum()),
        'cluster_sizes': cluster_sizes,
        'outlier_indices': list(np.where(outlier_mask)[0]),
        'eps_used': float(eps),
    }


def _pairwise_distances(X):
    """Compute pairwise Euclidean distance matrix."""
    sq = np.sum(X ** 2, axis=1)
    dists = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.maximum(dists, 0, out=dists)
    return np.sqrt(dists)


def _estimate_eps(dists, k):
    """Estimate eps using the k-distance heuristic.

    Sorts the k-th nearest neighbor distance for each point and
    looks for the knee/elbow point. Uses the max-curvature method.
    """
    n = dists.shape[0]
    k_dists = np.sort(dists, axis=1)[:, min(k, n - 1)]
    sorted_kdists = np.sort(k_dists)

    # Simple knee detection: max second derivative
    if len(sorted_kdists) < 3:
        return float(sorted_kdists.max()) if len(sorted_kdists) > 0 else 1.0

    # Smooth slightly and find max curvature
    d2 = np.diff(np.diff(sorted_kdists))
    if len(d2) == 0:
        return float(sorted_kdists[-1])
    knee_idx = int(np.argmax(d2)) + 1
    return float(sorted_kdists[knee_idx])


# ---------------------------------------------------------------------------
# Multi-level Otsu threshold
# ---------------------------------------------------------------------------

def multi_otsu(values, n_classes=3):
    """Multi-level Otsu threshold for 1D data.

    Extends Otsu's method to find N-1 thresholds that minimize within-class
    variance across N classes. For 2 classes, equivalent to standard Otsu.
    For 3 classes, finds two thresholds separating low/medium/high.

    Useful for intensity-based cell classification (e.g. FUCCI phase by
    nuclear intensity) when the number of populations is known.

    Uses histogram-based optimization for speed (O(n_bins * n_classes)).

    Args:
        values: 1D array of measurement values.
        n_classes: Number of classes (2, 3, or 4).

    Returns:
        dict with:
            thresholds: List of N-1 threshold values.
            labels: 1D int array of class assignments (0 = lowest).
            counts: Dict mapping class label → count.
            class_means: List of per-class means.
            between_class_variance: The maximized between-class variance.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = len(arr)

    if n == 0 or n_classes < 2:
        return {
            'thresholds': [],
            'labels': np.zeros(n, dtype=int),
            'counts': {0: n},
            'class_means': [float(arr.mean())] if n > 0 else [],
            'between_class_variance': 0.0,
        }

    # Build histogram
    n_bins = min(256, n)
    hist_counts, bin_edges = np.histogram(arr, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    n_classes = min(n_classes, n_bins)

    if n_classes == 2:
        thresholds = [_otsu_2class(hist_counts, bin_centers)]
    elif n_classes == 3:
        thresholds = _otsu_3class(hist_counts, bin_centers)
    elif n_classes == 4:
        thresholds = _otsu_4class(hist_counts, bin_centers)
    else:
        # For >4, fall back to quantile-based splitting
        quantiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        thresholds = [float(np.percentile(arr, q)) for q in quantiles]

    thresholds = sorted(thresholds)

    # Assign labels
    labels = np.zeros(n, dtype=int)
    for t_idx, t in enumerate(thresholds):
        labels[arr > t] = t_idx + 1

    # Compute per-class statistics
    counts = {}
    class_means = []
    for c in range(n_classes):
        mask = labels == c
        counts[c] = int(mask.sum())
        class_means.append(float(arr[mask].mean()) if mask.any() else 0.0)

    # Between-class variance
    grand_mean = arr.mean()
    bcv = sum(counts[c] * (class_means[c] - grand_mean) ** 2
              for c in range(n_classes)) / max(n, 1)

    return {
        'thresholds': [round(t, 4) for t in thresholds],
        'labels': labels,
        'counts': counts,
        'class_means': [round(m, 4) for m in class_means],
        'between_class_variance': round(float(bcv), 4),
    }


def _otsu_2class(hist_counts, bin_centers):
    """Standard Otsu threshold for 2 classes."""
    total = hist_counts.sum()
    if total == 0:
        return float(bin_centers[len(bin_centers) // 2])

    cum_sum = np.cumsum(hist_counts)
    cum_mean = np.cumsum(hist_counts * bin_centers)
    global_mean = cum_mean[-1]

    best_var = -1
    best_thresh = float(bin_centers[0])

    for i in range(len(hist_counts) - 1):
        w0 = cum_sum[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[i] / w0
        mu1 = (global_mean - cum_mean[i]) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = float(bin_centers[i])

    return best_thresh


def _otsu_3class(hist_counts, bin_centers):
    """Multi-Otsu for 3 classes (2 thresholds)."""
    n_bins = len(hist_counts)
    total = hist_counts.sum()
    if total == 0:
        return [float(bin_centers[n_bins // 3]),
                float(bin_centers[2 * n_bins // 3])]

    cum_sum = np.cumsum(hist_counts)
    cum_mean = np.cumsum(hist_counts * bin_centers)
    global_mean = cum_mean[-1] / total

    best_var = -1
    best_t1, best_t2 = 0, 0

    # Subsample for speed if many bins
    step = max(1, n_bins // 64)

    for i in range(0, n_bins - 2, step):
        w0 = cum_sum[i]
        if w0 == 0:
            continue
        mu0 = cum_mean[i] / w0

        for j in range(i + 1, n_bins - 1, step):
            w1 = cum_sum[j] - cum_sum[i]
            w2 = total - cum_sum[j]
            if w1 == 0 or w2 == 0:
                continue
            mu1 = (cum_mean[j] - cum_mean[i]) / w1
            mu2 = (cum_mean[-1] - cum_mean[j]) / w2

            var = (w0 * (mu0 - global_mean) ** 2 +
                   w1 * (mu1 - global_mean) ** 2 +
                   w2 * (mu2 - global_mean) ** 2)

            if var > best_var:
                best_var = var
                best_t1 = i
                best_t2 = j

    return [float(bin_centers[best_t1]), float(bin_centers[best_t2])]


def _otsu_4class(hist_counts, bin_centers):
    """Multi-Otsu for 4 classes (3 thresholds)."""
    n_bins = len(hist_counts)
    total = hist_counts.sum()
    if total == 0:
        return [float(bin_centers[n_bins // 4]),
                float(bin_centers[n_bins // 2]),
                float(bin_centers[3 * n_bins // 4])]

    cum_sum = np.cumsum(hist_counts)
    cum_mean = np.cumsum(hist_counts * bin_centers)
    global_mean = cum_mean[-1] / total

    best_var = -1
    best_t = [0, 0, 0]

    # Coarser step for 4-class (O(n^3) → manageable)
    step = max(1, n_bins // 32)

    for i in range(0, n_bins - 3, step):
        w0 = cum_sum[i]
        if w0 == 0:
            continue
        mu0 = cum_mean[i] / w0

        for j in range(i + 1, n_bins - 2, step):
            w1 = cum_sum[j] - cum_sum[i]
            if w1 == 0:
                continue
            mu1 = (cum_mean[j] - cum_mean[i]) / w1

            for k in range(j + 1, n_bins - 1, step):
                w2 = cum_sum[k] - cum_sum[j]
                w3 = total - cum_sum[k]
                if w2 == 0 or w3 == 0:
                    continue
                mu2 = (cum_mean[k] - cum_mean[j]) / w2
                mu3 = (cum_mean[-1] - cum_mean[k]) / w3

                var = (w0 * (mu0 - global_mean) ** 2 +
                       w1 * (mu1 - global_mean) ** 2 +
                       w2 * (mu2 - global_mean) ** 2 +
                       w3 * (mu3 - global_mean) ** 2)

                if var > best_var:
                    best_var = var
                    best_t = [i, j, k]

    return [float(bin_centers[t]) for t in best_t]
