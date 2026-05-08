# Multi-Scale Morphometry Workflow

## When to Use
Tissue pathology, nuclear size surveys, dysplasia screening, any challenge
requiring population-level measurements at cellular resolution.

## Workflow: Survey → Tile → Measure → Classify

### Phase 1: 10x Survey (ALWAYS FIRST)
```python
set_objective(core, 10)  # low-mag overview

# Resolve channel names from the config (don't hardcode — they vary per system)
cfg = get_config(core)
bf   = snap(core, channel=cfg.channels['brightfield'])
nuc  = snap(core, channel=cfg.channels['nucleus'])
mem  = snap(core, channel=cfg.channels['membrane'])
```
**Purpose**: Count total objects, map spatial distribution, identify regions of interest.
Save and visually inspect ALL channels before proceeding.

### Phase 2: High-Mag Tiling (20x or 40x)
```python
# Compute tile positions
from src.core.workflows.scanning import grid_positions
positions = grid_positions(world_width, world_height, fov_size, overlap_frac=0.1)

# Or manual grid for known FOV
# 20x: pixel_size=0.5, FOV=256µm → 4 tiles for 512µm area
positions = [(128,128), (384,128), (128,384), (384,384)]

# Acquire via MDA
events = [MDAEvent(x_pos=x, y_pos=y, channel={"config": ch_nucleus})
          for x, y in positions]
frames = run_events(core, events)
```

### Phase 3: Segment + Measure
```python
from src.core.analysis.morphometry import segment_nuclei, measure_objects, to_world_coords

all_measurements = []
for (img, event), (sx, sy) in zip(frames, positions):
    seg = segment_nuclei(img, method='otsu', min_area=50)
    meas = measure_objects(seg['props'], pixel_size=pixel_size,
                           edge_margin=10, image_shape=img.shape)
    meas = to_world_coords(meas, sx, sy, pixel_size)
    all_measurements.extend(meas)
```

### Phase 4: Population Statistics + Outlier Detection
```python
from src.core.analysis.morphometry import population_stats, identify_outliers

stats = population_stats(all_measurements, key='diameter_um')
outliers = identify_outliers(all_measurements, key='diameter_um',
                             method='zscore', threshold=2.0)
```

### Phase 5: High-Mag Confirmation (optional)
Zoom to 40x or 100x on detected outliers for detailed morphology.

## Magnification Selection Guide

| Mag | Pixel Size | FOV | Best For |
|-----|-----------|-----|----------|
| 10x | 1.0 µm/px | 512 µm | Survey, count, spatial distribution |
| 20x | 0.5 µm/px | 256 µm | Nuclear size measurement (sweet spot) |
| 40x | 0.25 µm/px | 128 µm | Detailed morphology, subcellular |
| 100x | 0.1 µm/px | 51.2 µm | Single-cell detail, organelles |

**Rule of thumb**: Use the lowest magnification where objects are >20 pixels across.
Nuclear diameter ~25 µm → at 20x = 50 px (excellent). At 10x = 25 px (OK for counting).

## Dysplasia Detection

Nuclear enlargement criteria:
- **Mean + 2σ**: Robust for normal distributions. Identifies extreme outliers.
- **IQR method**: Q75 + 1.5×IQR. More robust to skewed distributions.
- **Ratio to median**: > 1.5× median diameter. Simple, biologically motivated.

For pathology screening, report:
1. Total cell count
2. Mean ± SD of nuclear diameter
3. Number and locations of enlarged nuclei
4. Size ratio (dysplastic / normal mean)
5. Dysplastic fraction

## Pixel Size

On a real microscope, query `core.getPixelSizeUm()` — it's a property of the current objective/camera combo, not a formula. The values above are examples for a 10 µm/px-at-1x reference system.

## Pitfalls
- **Skipping 10x overview**: The #1 recurring mistake. ALWAYS survey first.
- **Edge nuclei**: Partial objects at tile boundaries inflate measurements. Use edge_margin.
- **Duplicate counting**: If tiles overlap, use deduplicate_cells() or distance-based filtering.
- **Threshold drift between tiles**: Use the same threshold for all tiles (compute from first tile or use fixed).
- **Magnification mismatch**: Ensure pixel_size matches the actual objective used.

## Literature

- [[Papers/Shi 2024]] — SmartLLSM extends multi-scale survey→zoom from magnification switching to modality switching: a low-dose widefield scout identifies rare cellular events with a YOLOv5 detector, and the instrument autonomously escalates each hit to a 3D/4D lattice light-sheet acquisition. The photon-budget asymmetry between scout and capture is much larger than 10×→40× on the same objective, making the survey-first discipline non-negotiable.
