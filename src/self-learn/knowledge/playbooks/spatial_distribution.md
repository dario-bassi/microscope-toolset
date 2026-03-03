# Spatial Distribution Analysis Playbook

## When to use
- Colony spacing analysis (bacterial, yeast)
- Cell distribution uniformity in tissue
- Aggregation detection (Dictyostelium, immune cells)
- Quality control of cell seeding in wells/chambers

## Quick assessment: Clark-Evans index

The simplest test — is the distribution random, clustered, or regular?

```python
from src.analysis.spatial import clark_evans_index

# centroids: (n, 2) array of (x, y) coordinates
result = clark_evans_index(centroids, area=fov_area_um2)
R = result['R']
# R < 0.8 → clustered
# 0.8 ≤ R ≤ 1.2 → random (CSR)
# R > 1.2 → dispersed/regular
```

## Multi-scale analysis: Ripley's K/L

Detects clustering at different spatial scales.

```python
from src.analysis.spatial import ripleys_l

radii = np.linspace(5, 100, 20)  # test distances in µm
result = ripleys_l(centroids, radii, area=fov_area_um2)

# L(r) - r > 0 → clustering at scale r
# L(r) - r < 0 → dispersion at scale r
# L(r) - r ≈ 0 → random at scale r
```

**Interpretation**: If L-r peaks at r=30 µm, cells cluster at ~30 µm scale
(e.g., cell clusters of ~30 µm diameter).

## Territory analysis: Voronoi

How much space does each cell control?

```python
from src.analysis.spatial import voronoi_areas

result = voronoi_areas(centroids, bounds=(0, 0, 512, 512))
cv = result['cv']  # coefficient of variation
# Low CV (<0.3) → regular spacing
# High CV (>0.6) → clustered (some cells have huge territories, others tiny)
```

## Grid uniformity: quadrat counts

Divide FOV into grid, count cells per square.

```python
from src.analysis.spatial import quadrat_count

result = quadrat_count(centroids, grid_size=50,
                        bounds=(0, 0, 512, 512))
vmr = result['VMR']  # variance-to-mean ratio
# VMR ≈ 1 → Poisson random
# VMR > 1 → overdispersed (clustered)
# VMR < 1 → underdispersed (regular)
```

## Nearest-neighbor analysis

```python
from src.analysis.spatial import nearest_neighbor_distances

result = nearest_neighbor_distances(centroids)
mean_nn = result['mean']  # mean nearest-neighbor distance
# Compare to expected: 0.5 / sqrt(density)
# If mean_nn << expected → clustered
```

## Common workflows

### Colony spacing on agar plate
1. Detect colonies at 10x (BF dark spots or fluorescent spots)
2. Convert centroids to world coordinates
3. `clark_evans_index()` for overall pattern
4. `ripleys_l()` to identify clustering scale
5. Report: R-index, mean inter-colony distance, number of clusters

### Cell seeding uniformity check
1. Count cells per quadrat (`quadrat_count()`)
2. Report VMR — should be ~1 for uniform seeding
3. Identify empty quadrats (seeding failures)
4. `voronoi_areas()` to find territory heterogeneity

### Immune cell aggregation
1. Track immune cells over time
2. At each timepoint: `clark_evans_index()` on centroids
3. Plot R vs time — decreasing R = progressive aggregation
4. `ripleys_k()` to identify the aggregation scale

## Pitfalls
- **Edge effects**: Cells near FOV edge have artificially large NN distances.
  Use `area` parameter explicitly for Clark-Evans.
- **Small sample size**: <20 cells → statistical tests unreliable
- **Multiple FOVs**: Pool centroids from tiled acquisition with world coordinates
- **Non-square FOV**: Pass explicit `area` to avoid bounding box overestimate
