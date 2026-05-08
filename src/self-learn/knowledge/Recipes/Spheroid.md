# Spheroid Multi-Scale Structural Characterization

**Assumes:** 3D tumor spheroid, multi-scale acquisition across 3 objective states, BF + viability stain. Paired with `src/recipes/spheroid_drug_penetration.py` (`measure_penetration_depth` — radial viability profile + plateau-method penetration depth).

## When to Use
3D tumor spheroid with concentric layers (proliferating rim, quiescent mantle, necrotic core). Multi-scale characterization across 3 objective magnifications.

## Key Properties
- Spheroid has 3 layers: outer proliferating rim, middle quiescent mantle, inner necrotic core
- BF shows: bright halo at edge (phase contrast), lighter rim, darker mantle, darkest core
- Calcein (live-cell marker): bright dots = live cells, forming a ring (absent in core)
- PI (dead-cell marker): bright dots = dead cells, concentrated in core

## Workflow

### Phase 1: Low-Mag Z-scan (10x)
1. Scan Z across a range bracketing the equator (step size ~10-15µm)
2. Find equatorial plane = maximum BF std (sharpest focus)
3. Measure overall diameter from BF radial profile
4. Diameter = 2x radius at halo peak in BF azimuthal profile

### Phase 2: Mid-Mag Fluorescence (20x)
1. Switch objective to 20x, re-center stage if needed
2. Snap calcein and PI channels at equatorial Z
3. Measure necrotic core diameter from calcein gap (radial profile: where calcein rises above 25% of max)
4. Convert measurements to consistent pixel scale using magnification ratio
5. Core boundary verified by PI signal outer extent

### Phase 3: High-Mag Cell Counting (40x)
1. Switch to 40x, re-center stage
2. Combined detection: max(calcein, PI) > threshold
3. Watershed splitting for touching cells: distance_transform + peak_local_max(min_distance=8) + watershed
4. Classify: PI signal > threshold → dead, else → live
5. Filter by area (cells are large at high mag; check typical cell area from a few examples)

## Key Code Pattern
```python
# Objective switching — check available labels/states first
# core.setState('Objective', state_index) if set_objective() fails
core.waitForDevice('Objective')

# After switching objectives, verify pixel size and re-center if needed
pixel_size = core.getPixelSizeUm()
```

## Pitfalls
- set_objective() may not parse state labels correctly — try core.setState('Objective', N) directly
- At high mag, cells are LARGE, not dots. Adjust area filters accordingly
- Necrotic core: calcein gap method is more reliable than PI extent (PI cells are sparse/scattered)
- Radial profile smoothing: use 3px bin width minimum for clean azimuthal averages
- Halo artifact in BF: peak at spheroid edge can confuse thresholding — use radial profile
- Watershed can over-segment at high mag (cells with heterogeneous brightness get split). Try simple threshold + CC + size filter first
- Classify live/dead by which channel each component is brightest in
