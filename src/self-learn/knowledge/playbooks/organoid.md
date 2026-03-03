# Organoid Z-Stack Morphometry

## When to Use
- 3D organoids with hollow lumen structure (intestinal, brain, kidney organoids)
- Annular cross-section: outer wall + central lumen
- Multi-channel: BF for structure, membrane marker for wall, DAPI for nuclei

## Key Properties
- Organoid is a **hollow sphere**: dark ring (wall) around bright center (lumen) in BF
- Equatorial plane = maximum filled cross-section area
- Wall thickness varies — measure from both axes and average
- Organoid may be larger than FOV at high mag — verify before switching objectives

## Workflow

### Phase 1: Coarse Z-Scan (10x)
1. Start at 10x — organoid likely fits in FOV (~500 um diameter)
2. Coarse Z-scan: -100 to +100 um, step 10 um (21 planes)
3. At each Z: threshold BF (`img < p90 - 20`), fill holes, measure largest CC area
4. Equatorial plane = Z with maximum filled area
5. If organoid > FOV at 10x, stay at 10x for all measurements

### Phase 2: Fine Z-Stack
1. Fine scan: ±20 um around coarse equatorial, step 2 um
2. Refine equatorial Z to ±2 um precision
3. Record equatorial Z for final measurements

### Phase 3: Multi-Channel at Equatorial
1. Snap all available channels at equatorial Z
2. BF: structural segmentation (outer boundary + lumen)
3. Membrane channel: ring analysis for wall thickness (independent cross-check)
4. DAPI/nucleus channel: nuclei counting in wall region

### Phase 4: Measurements
1. **Outer boundary**: threshold BF → fill holes → largest CC → regionprops
2. **Lumen**: organoid_filled AND NOT wall_mask → largest CC → regionprops
3. **Diameter**: report major and minor axes, plus mean
4. **Wall thickness**: `((outer_major - inner_major) + (outer_minor - inner_minor)) / 4`
5. **Ring analysis**: `measure_ring()` on membrane channel for independent wall measurement
6. **Eccentricity**: from regionprops on outer boundary

### Phase 5: Nuclei Counting
1. Threshold nucleus channel (absolute threshold, e.g., `> 15`)
2. Binary opening to clean noise
3. Label connected components, filter by area (10 < area < 3000 px)
4. Count only nuclei within the organoid wall region (not in lumen)
5. Distance filter: nuclei centroid should be far enough from organoid center

### Phase 6: Volume from Z-Stack
1. Full Z-scan with finer step (5 um) for volume integration
2. At each Z: segment → measure cross-section area
3. Volume = Σ(area_i × pixel_size² × z_step)
4. Z-height: range of Z-planes where area > 10% of maximum

## Key Code Pattern
```python
from useq import MDAEvent
from src.hardware.core import run_events
from src.analysis.ring import measure_ring

# Z-scan via MDA events
z_positions = np.arange(-100, 105, 10)
events = [MDAEvent(z_pos=float(z), channel={'config': 'brightfield'}, exposure=50.0)
          for z in z_positions]
frames = run_events(core, events)

# BF segmentation at each Z
for (img, event), z in zip(frames, z_positions):
    bg = np.percentile(img, 90)
    wall = img < (bg - 20)
    wall = ndimage.binary_opening(wall, iterations=1)
    filled = ndimage.binary_fill_holes(wall)
    # Largest CC area → equatorial detection

# Ring analysis on membrane channel
ring = measure_ring(membrane_img, pixel_size=pixel_size)
wall_thickness = ring['wall_thickness_um']
```

## Pitfalls
- **Magnification**: organoid (~250 um diameter) may not fit in 20x FOV (256 um). Check before switching
- **Lumen vs noise**: fill_holes on wall_mask to get organoid_filled, then subtract wall to get lumen
- **BF threshold**: `p90 - 20` works well for dark-wall-on-bright-background
- **Nuclei in lumen**: some nuclei may be in the lumen (apoptotic debris) — only count wall nuclei
- **Volume**: use Riemann sum over Z-stack, NOT 4/3πr³ (organoid may not be spherical)
- **JSON serialization**: numpy int64/float64 → cast to Python int()/float() before submitting

### CRITICAL: BF overestimates boundaries (ch460 lesson, cost 4 points)
**BF dark ring includes out-of-focus shadow/phase contrast halo extending ~20px
beyond the actual organoid boundary.** This inflates baseline diameter and
compresses expansion percentage.

**For quantitative morphometry, use DAPI or membrane channel:**
- Membrane-channel (E-cadherin): sharp cell-cell boundaries = true organoid edge
- DAPI (nuclei ring): nuclear positions define wall boundaries precisely
- BF is fine for: finding the organoid, coarse Z-scan, detecting presence

**Implication for perturbation studies:**
If baseline is overestimated, expansion % is artificially compressed.
ch460 R1: BF gave 7% expansion (GT was 29%). ch460 R2: Membrane gave 24.4% (9/10).

### Forskolin Swelling Assay Protocol (ch460=9/10)
1. Perfusion device: `core.setProperty("Perfusion", "Label", "Off")` → baseline
2. Baseline: 5 frames, `measure_ring()` on membrane-channel
3. Drug: `core.setProperty("Perfusion", "Label", "Drug")` → 30 frames
4. Washout: `core.setProperty("Perfusion", "Label", "Off")` → 20 frames
5. Report: outer/lumen/wall at baseline+drug+recovery, expansion %, swelling rate, recovery fraction
6. Expected: ~25-30% outer expansion, wall thinning ~15-20%, recovery partial
