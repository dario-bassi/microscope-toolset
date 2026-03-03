"""Spatial statistics for cell distribution analysis.

Quantifies how cells are distributed in space: random, clustered,
or regular. Works with centroid coordinates from morphometry.

Functions:
    nearest_neighbor_distances -- Distance to each cell's nearest neighbor
    clark_evans_index          -- Aggregation index (R < 1 = clustered)
    ripleys_k                  -- Ripley's K function for multi-scale clustering
    ripleys_l                  -- Normalized L function (L > r = clustered)
    voronoi_areas              -- Per-cell territory via Voronoi tessellation
    quadrat_count              -- Count cells per grid cell for chi-squared test
    density_map                -- Smoothed spatial density from cell positions
    detect_density_peaks       -- Find accumulation regions in a density map
"""

import math

import numpy as np


def nearest_neighbor_distances(centroids):
    """Compute distance from each cell to its nearest neighbor.

    Args:
        centroids: Array-like of shape (n, 2) with (x, y) coordinates.

    Returns:
        dict with:
            distances: 1D array of NN distances (length n).
            mean: Mean NN distance.
            std: Standard deviation of NN distances.
            indices: 1D array of nearest neighbor indices.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    n = pts.shape[0]

    if n < 2:
        return {
            'distances': np.array([]),
            'mean': 0.0,
            'std': 0.0,
            'indices': np.array([], dtype=int),
        }

    # Pairwise distance matrix
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)

    nn_idx = dists.argmin(axis=1)
    nn_dist = dists[np.arange(n), nn_idx]

    return {
        'distances': nn_dist,
        'mean': float(nn_dist.mean()),
        'std': float(nn_dist.std()),
        'indices': nn_idx,
    }


def clark_evans_index(centroids, area=None):
    """Clark-Evans aggregation index R.

    R < 1 → clustered, R ≈ 1 → random, R > 1 → regular/dispersed.

    Args:
        centroids: Array-like (n, 2) of (x, y) coordinates.
        area: Total study area. If None, estimated from bounding box
            with a 5% margin.

    Returns:
        dict with:
            R: Clark-Evans ratio.
            observed_mean: Observed mean NN distance.
            expected_mean: Expected mean NN distance under CSR.
            n: Number of points.
            density: Point density (n / area).
            interpretation: 'clustered', 'random', or 'dispersed'.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    n = pts.shape[0]

    if n < 2:
        return {
            'R': 1.0, 'observed_mean': 0.0, 'expected_mean': 0.0,
            'n': n, 'density': 0.0, 'interpretation': 'insufficient_data',
        }

    nn = nearest_neighbor_distances(pts)
    obs_mean = nn['mean']

    if area is None:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        margin = ranges * 0.05
        area = float(np.prod(ranges + 2 * margin))
        if area <= 0:
            area = 1.0

    density = n / area
    expected_mean = 0.5 / np.sqrt(density)

    R = obs_mean / expected_mean if expected_mean > 0 else 1.0

    if R < 0.8:
        interp = 'clustered'
    elif R > 1.2:
        interp = 'dispersed'
    else:
        interp = 'random'

    return {
        'R': float(R),
        'observed_mean': float(obs_mean),
        'expected_mean': float(expected_mean),
        'n': n,
        'density': float(density),
        'interpretation': interp,
    }


def ripleys_k(centroids, radii, area=None):
    """Ripley's K function for multi-scale spatial analysis.

    K(r) > πr² indicates clustering at scale r.

    Args:
        centroids: Array-like (n, 2) of (x, y) coordinates.
        radii: Array-like of distances at which to evaluate K.
        area: Study area. If None, estimated from bounding box.

    Returns:
        dict with:
            radii: 1D array of evaluation radii.
            K: 1D array of K(r) values.
            K_csr: 1D array of expected K under complete spatial randomness (πr²).
            n: Number of points.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = pts.shape[0]

    K_csr = np.pi * radii ** 2

    if n < 2:
        return {
            'radii': radii,
            'K': np.zeros_like(radii),
            'K_csr': K_csr,
            'n': n,
        }

    if area is None:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        margin = ranges * 0.05
        area = float(np.prod(ranges + 2 * margin))
        if area <= 0:
            area = 1.0

    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    K_vals = np.zeros_like(radii)
    for i, r in enumerate(radii):
        # Count pairs within radius r (excluding self)
        count = np.sum(dists < r) - n  # subtract diagonal
        K_vals[i] = area * count / (n * (n - 1)) if n > 1 else 0

    return {
        'radii': radii,
        'K': K_vals,
        'K_csr': K_csr,
        'n': n,
    }


def ripleys_l(centroids, radii, area=None):
    """Normalized Ripley's L function.

    L(r) - r > 0 indicates clustering; L(r) - r < 0 indicates dispersion.

    Args:
        centroids: Array-like (n, 2).
        radii: Array-like of evaluation radii.
        area: Study area (optional).

    Returns:
        dict with:
            radii: 1D array.
            L: 1D array of L(r) values.
            L_minus_r: 1D array of L(r) - r (positive = clustered).
    """
    K_result = ripleys_k(centroids, radii, area)
    K = K_result['K']
    L = np.sqrt(K / np.pi)
    r = K_result['radii']

    return {
        'radii': r,
        'L': L,
        'L_minus_r': L - r,
    }


def voronoi_areas(centroids, bounds=None):
    """Compute Voronoi cell areas for each point.

    Uses a simple grid-based approximation (assigns each pixel to
    nearest centroid). No scipy.spatial dependency.

    Args:
        centroids: Array-like (n, 2) of (x, y) coordinates.
        bounds: (x_min, y_min, x_max, y_max). If None, inferred from
            data with 10% margin.

    Returns:
        dict with:
            areas: 1D array of Voronoi cell areas.
            cv: Coefficient of variation (std/mean) — low CV = regular,
                high CV = clustered.
            mean: Mean area.
            std: Standard deviation.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    n = pts.shape[0]

    if n == 0:
        return {'areas': np.array([]), 'cv': 0.0, 'mean': 0.0, 'std': 0.0}

    if n == 1:
        if bounds is not None:
            total = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        else:
            total = 1.0
        return {'areas': np.array([total]), 'cv': 0.0,
                'mean': total, 'std': 0.0}

    if bounds is None:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        margin = (maxs - mins) * 0.1
        bounds = (mins[0] - margin[0], mins[1] - margin[1],
                  maxs[0] + margin[0], maxs[1] + margin[1])

    # Grid resolution: ~200 cells per side for efficiency
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    res = max(width, height) / 200
    if res <= 0:
        res = 1.0

    nx = max(int(width / res), 1)
    ny = max(int(height / res), 1)

    xs = np.linspace(bounds[0], bounds[2], nx)
    ys = np.linspace(bounds[1], bounds[3], ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Assign each grid point to nearest centroid
    diff = grid_pts[:, None, :] - pts[None, :, :]
    dists = (diff ** 2).sum(axis=2)
    labels = dists.argmin(axis=1)

    pixel_area = (width / nx) * (height / ny)
    areas = np.zeros(n)
    for i in range(n):
        areas[i] = np.sum(labels == i) * pixel_area

    mean = float(areas.mean())
    std = float(areas.std())

    return {
        'areas': areas,
        'cv': std / mean if mean > 0 else 0.0,
        'mean': mean,
        'std': std,
    }


def quadrat_count(centroids, grid_size, bounds=None):
    """Count cells per grid quadrat for uniformity testing.

    Args:
        centroids: Array-like (n, 2) of (x, y) coordinates.
        grid_size: Side length of each square quadrat.
        bounds: (x_min, y_min, x_max, y_max). If None, inferred.

    Returns:
        dict with:
            counts: 2D array of cell counts per quadrat.
            chi2: Chi-squared statistic vs expected uniform distribution.
            p_uniform: p-value for uniformity (< 0.05 = non-uniform).
            VMR: Variance-to-mean ratio (> 1 = clustered, < 1 = regular).
            n_quadrats: Total number of quadrats.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    n = pts.shape[0]

    if bounds is None:
        if n == 0:
            bounds = (0, 0, 1, 1)
        else:
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            bounds = (float(mins[0]), float(mins[1]),
                      float(maxs[0] + grid_size), float(maxs[1] + grid_size))

    nx = max(int(np.ceil((bounds[2] - bounds[0]) / grid_size)), 1)
    ny = max(int(np.ceil((bounds[3] - bounds[1]) / grid_size)), 1)

    counts = np.zeros((ny, nx), dtype=int)

    for p in pts:
        ix = int((p[0] - bounds[0]) / grid_size)
        iy = int((p[1] - bounds[1]) / grid_size)
        ix = min(ix, nx - 1)
        iy = min(iy, ny - 1)
        counts[iy, ix] += 1

    flat = counts.ravel().astype(float)
    n_quadrats = len(flat)
    expected = n / n_quadrats if n_quadrats > 0 else 0

    if expected > 0:
        chi2 = float(np.sum((flat - expected) ** 2 / expected))
        # Approximate p-value using chi2 with df = n_quadrats - 1
        # Simple approximation: use normal approx for large df
        df = n_quadrats - 1
        if df > 0:
            z = (chi2 - df) / np.sqrt(2 * df)
            p_val = float(1 - 0.5 * (1 + math.erf(z / np.sqrt(2))))
        else:
            p_val = 1.0
    else:
        chi2 = 0.0
        p_val = 1.0

    mean_count = flat.mean()
    var_count = flat.var()
    VMR = float(var_count / mean_count) if mean_count > 0 else 0.0

    return {
        'counts': counts,
        'chi2': chi2,
        'p_uniform': p_val,
        'VMR': VMR,
        'n_quadrats': n_quadrats,
    }


def density_map(positions, field_size=512, sigma=20, resolution=None):
    """Create a smoothed spatial density map from cell positions.

    Places a Gaussian kernel at each cell position and sums them to create
    a continuous density estimate. Useful for finding accumulation regions,
    tracking density changes over time, and detecting mounds.

    Args:
        positions: (N, 2) array of (y, x) coordinates.
        field_size: int or (height, width) tuple defining the map extent.
        sigma: float, Gaussian smoothing radius in pixels. Larger values
            produce smoother density maps. Match to expected cluster size.
        resolution: int or None. Output map resolution. If None, equals
            field_size. Use smaller values (e.g. 64) for speed.

    Returns:
        dict with:
            map: 2D array of density values (smoothed cell count per pixel).
            peak_density: float, maximum density value.
            total_mass: float, sum of all density values (proportional to N).
            field_size: (height, width) used.
    """
    from scipy.ndimage import gaussian_filter

    positions = np.asarray(positions, dtype=float)

    if isinstance(field_size, (int, float)):
        h, w = int(field_size), int(field_size)
    else:
        h, w = int(field_size[0]), int(field_size[1])

    if resolution is None:
        res_h, res_w = h, w
    elif isinstance(resolution, (int, float)):
        res_h = res_w = int(resolution)
    else:
        res_h, res_w = int(resolution[0]), int(resolution[1])

    # Accumulate point counts on a grid
    raw = np.zeros((res_h, res_w), dtype=float)
    scale_y = res_h / h if h > 0 else 1
    scale_x = res_w / w if w > 0 else 1

    for pos in positions:
        gy = int(pos[0] * scale_y)
        gx = int(pos[1] * scale_x)
        if 0 <= gy < res_h and 0 <= gx < res_w:
            raw[gy, gx] += 1

    # Smooth with Gaussian
    sigma_scaled = sigma * min(scale_y, scale_x)
    smoothed = gaussian_filter(raw, sigma=max(sigma_scaled, 0.5))

    return {
        'map': smoothed,
        'peak_density': float(smoothed.max()),
        'total_mass': float(smoothed.sum()),
        'field_size': (h, w),
    }


def detect_density_peaks(dmap, min_distance=50, threshold_percentile=90,
                          min_peak_value=None):
    """Find significant accumulation regions (peaks) in a density map.

    Use with density_map() output or any 2D density/intensity array.
    Good for detecting aggregation mounds, colony centers, or hotspots.

    Args:
        dmap: 2D array of density values (from density_map()['map'] or image).
        min_distance: int, minimum distance between peaks in pixels.
        threshold_percentile: float, only peaks above this percentile are kept.
        min_peak_value: float or None. Absolute minimum peak value. If None,
            computed from threshold_percentile.

    Returns:
        dict with:
            peaks: (M, 2) array of peak (y, x) coordinates.
            n_peaks: int.
            peak_values: list of float, density value at each peak.
            threshold: float, the threshold used.
    """
    from skimage.feature import peak_local_max

    dmap = np.asarray(dmap, dtype=float)

    if dmap.size == 0 or dmap.max() == 0:
        return {
            'peaks': np.zeros((0, 2)),
            'n_peaks': 0,
            'peak_values': [],
            'threshold': 0.0,
        }

    if min_peak_value is None:
        min_peak_value = float(np.percentile(dmap, threshold_percentile))

    peaks = peak_local_max(dmap, min_distance=min_distance,
                           threshold_abs=min_peak_value)

    peak_values = [float(dmap[p[0], p[1]]) for p in peaks]

    return {
        'peaks': peaks if len(peaks) > 0 else np.zeros((0, 2)),
        'n_peaks': len(peaks),
        'peak_values': peak_values,
        'threshold': float(min_peak_value),
    }


def classify_trapped(cell_positions, trap_positions, trap_radius=5.0,
                     position_axis=0):
    """Classify cells as trapped or free based on proximity to trap positions.

    In microfluidic devices, traps are at fixed positions (typically X coordinates).
    Cells within ``trap_radius`` of any trap position along ``position_axis``
    are classified as trapped; all others are free.

    Args:
        cell_positions: Array-like of shape (n, 2) — cell (x, y) world coords.
        trap_positions: Array-like of shape (m,) — 1D trap positions along
            the relevant axis, OR (m, 2) for 2D trap positions.
        trap_radius: Maximum distance to a trap to be classified as trapped (um).
        position_axis: Which axis to use for 1D trap matching (0=x, 1=y).
            Ignored if trap_positions is 2D.

    Returns:
        dict with:
            trapped: list of indices of trapped cells.
            free: list of indices of free cells.
            n_trapped: int count.
            n_free: int count.
            distances: 1D array of min distance to nearest trap for each cell.
    """
    cells = np.asarray(cell_positions, dtype=float)
    traps = np.asarray(trap_positions, dtype=float)

    if cells.ndim != 2 or cells.shape[0] == 0:
        return {'trapped': [], 'free': [], 'n_trapped': 0, 'n_free': 0,
                'distances': np.array([])}

    if traps.size == 0:
        idx = list(range(len(cells)))
        return {'trapped': [], 'free': idx, 'n_trapped': 0,
                'n_free': len(idx),
                'distances': np.full(len(cells), np.inf)}

    # 1D trap positions: compare along one axis
    if traps.ndim == 1:
        cell_axis = cells[:, position_axis]
        # Distance of each cell to nearest trap position on that axis
        dists = np.abs(cell_axis[:, None] - traps[None, :])
        min_dists = np.min(dists, axis=1)
    else:
        # 2D trap positions: Euclidean distance
        from scipy.spatial.distance import cdist
        d = cdist(cells[:, :2], traps[:, :2])
        min_dists = np.min(d, axis=1)

    trapped = [int(i) for i in range(len(cells)) if min_dists[i] <= trap_radius]
    free = [int(i) for i in range(len(cells)) if min_dists[i] > trap_radius]

    return {
        'trapped': trapped,
        'free': free,
        'n_trapped': len(trapped),
        'n_free': len(free),
        'distances': min_dists,
    }
