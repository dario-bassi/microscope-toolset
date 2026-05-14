# Multi-Position Survey

> **When to use:** When the sample spans more than one FOV and objects must be detected and catalogued across a multi-tile grid.

> **Note**: Code examples below use conceptual pseudocode. All multi-position
> loops MUST use `MDASequence(stage_positions=[...])` + `run_events()` in practice.

## Workflow

```
grid_positions() -> snap at each -> detect -> pixel_to_world -> deduplicate -> store
```

## Step 1: Generate Grid Positions

```python
fov_um = core.getImageWidth() * core.getPixelSizeUm()
positions = grid_positions(center=(x0, y0), nx=3, ny=3, fov_um=fov_um, overlap=0.1)
```

Query pixel_size and FOV from hardware — they vary by objective and camera.

## Step 2: Snap and Detect at Each Position

```python
detections_world = []
for (x, y) in positions:
    core.setXYPosition(x, y)
    img = snap()
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    for (r, c) in detect_cells(img):
        wx = x + (c - cx) * pixel_size
        wy = y + (r - cy) * pixel_size
        detections_world.append((wx, wy))
```

## Step 3: Deduplicate Overlapping Detections

Objects near tile borders appear in multiple FOVs. Use hierarchical clustering:

```python
from scipy.cluster.hierarchy import fcluster, linkage
coords = np.array(detections_world)
if len(coords) > 1:
    Z = linkage(coords, method='single', metric='euclidean')
    labels = fcluster(Z, t=min_dist, criterion='distance')
    unique = [coords[labels == l].mean(axis=0) for l in np.unique(labels)]
```

**min_dist**: ~30um for 10x, ~8um for 40x. Rule of thumb: half the cell diameter.

## Step 4: Store Interesting Positions

```python
interesting = sorted(unique, key=lambda p: density_at(p), reverse=True)
```

These become targets for follow-up at higher magnification.

## Coordinate System

- **Pixel**: (row, col), origin top-left of image.
- **World**: (x_um, y_um), stage units. Always deduplicate in world coordinates.
- `pixel_to_world(row, col, stage_x, stage_y, pixel_size)` handles conversion.

## Common Pitfalls

- **No overlap**: Cells at tile edges get missed. Use 10-15% overlap.
- **No deduplication**: Border cells appear twice. Always cluster.
- **Pixel vs world**: Two detections at pixel (500,500) in adjacent tiles are different cells.
- **Stage settling**: After `setXYPosition`, verify with a test snap if needed.

## See also

- [[Core/Strategies/Multichannel scan]] — same positions × multiple channels per tile.
- [[Core/Strategies/Multi-scale morphometry]] — survey at 10x, zoom at 40x.
- [[Core/Pitfalls/FOV vs well coverage]] — sizing the tile grid.
- [[Core/Approach/Coordinate systems]] — world vs tile-pixel coordinates.
- `self_learn.workflows.batch` — production implementation.

## Literature

- [[Papers/Kandel 2023]] — FAST: when you only intend to visit a fraction of the grid, pick the informative points with an online information-gain score (Expected Reduction in Distortion) and order the batch through a route optimiser. Dead time between arbitrary stage moves is as important to optimise as which points to pick; applies directly to whole-slide / whole-well tile scheduling.
