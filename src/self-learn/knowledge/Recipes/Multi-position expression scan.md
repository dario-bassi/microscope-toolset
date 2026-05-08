# Multi-Position Expression Scan Playbook

## When to Use
- Tissue or cell population spans multiple FOVs
- Need to find spatially localized expression hotspots
- Comparing marker intensity across regions

## Workflow

### 1. 10x Overview Tiled Scan
```python
from src.core.workflows.batch import tile_and_analyze, measure_nuclear_expression, identify_hotspot

# tile_and_analyze uses MDA internally — no manual snap loops
result = tile_and_analyze(
    core,
    analyze_fn=lambda img: measure_nuclear_expression(img, threshold=30, min_area=20),
    grid=(2, 2),
    channel='nucleus-channel',
    group='Fake',
    overlap=0.0,
    center=(512, 512),  # center of sample
)
```

### 2. Identify Hotspot
```python
hotspot = identify_hotspot(result['per_tile'], key='nuclear_mean')
print(f"Hotspot at {hotspot['hotspot_positions']}, fold change: {hotspot['fold_change']:.2f}x")
```

### 3. 20x Boundary Profiling
Use `multi_position_measure()` along the expression boundary:
```python
from src.core.workflows.batch import multi_position_measure

boundary_positions = [(x, boundary_y) for x in range(start, end, step)]
boundary = multi_position_measure(
    core, boundary_positions,
    measure_fn=lambda img: measure_nuclear_expression(img, threshold=30),
    channel='nucleus-channel',
)
```

### 4. Precise Hotspot Localization
Intensity-weighted centroid is more accurate than quadrant center:
```python
positions = np.array([t['position'] for t in result['per_tile']])
intensities = np.array([t['nuclear_mean'] for t in result['per_tile']])
centroid = (intensities @ positions) / intensities.sum()
```

## Key Lessons (ch463 = 9/10)
- **Tile centers, not corners**: For 2x2 grid with FOV=512, centers at (256,256), (768,256), (256,768), (768,768)
- **pixel_size=0 fallback**: `tile_and_analyze()` defaults to 1.0 when pixel_size <= 0
- **Fold change from pixel data**: Always measure from actual pixel intensities, not assumed parameters
- **Grid center matters**: Set `center` parameter to sample center, not stage origin
- **Start with wide scan**: Don't assume sample origin at (0,0) — scout first if unsure
