# Motile Organism Tracking Playbook

## When to use
Organisms that swim/crawl freely: Volvox, Paramecium, C. elegans, zebrafish larvae,
bacteria (motility assays), sperm. Any challenge requiring closed-loop stage tracking.

## General Workflow

### 1. Find the organism
- Snap 10x BF overview at origin
- If not at origin: `spiral_search()` from `src/workflows/stage_tracking`
- For transparent organisms: look for local variance/texture, not just intensity

### 2. Track with stage_tracking module
```python
from src.workflows.stage_tracking import make_tracker, track_target

def detect_fn(image):
    """Custom detection for your organism. Returns list of (cx, cy) pixel coords."""
    # Threshold, morphology, largest CC, centroid
    ...
    return [(cx, cy)]

tracker = make_tracker(initial_wx, initial_wy)
result = track_target(core, tracker, n_steps=60,
    channel='brightfield', detect_fn=detect_fn,
    max_match_dist=200, velocity_smoothing=0.4)
```

### 3. Measure while tracking
- Use `on_step` callback to snap additional channels between tracking steps
- Re-center frequently (every 5-10 frames) to prevent drift-off
- Record timestamps for speed calculation

### 4. snap() is correct for tracking
MDA sequences are fixed plans. Closed-loop tracking REQUIRES per-frame feedback:
snap -> detect -> decide where to move -> move -> snap again.
This is NOT a violation of the "use MDA" rule.

## Speed Measurement

**Use re-centering positions over longer intervals, NOT per-frame positions.**
- Per-frame centroids have sub-pixel jitter -> overestimates speed
- Re-centering positions (every 5-10 frames) average out detection noise
- For trajectory smoothing: `savgol_filter` on x/y coordinates

**Report**: path speed (total_path / total_time) and net speed (displacement / time).
Path speed > net speed indicates curved or oscillating trajectory.

## Rotation Detection (Volvox, embryos, etc.)

**Polar transform method (best):**
1. For each frame, find organism center (center_of_mass of mask)
2. Extract angular intensity profile at ~60% of organism radius
3. Cross-correlate successive profiles via FFT to find angular shift per frame
4. Cumulative angle over time -> divide 360 / cumulative rotation = period

**Pitfalls:**
- FFT on intensity at a fixed angle detects CELL SPACING frequency, not rotation frequency
- Frame-to-frame cross-correlation of full images is not sensitive enough
- Need >1 full rotation for reliable period measurement (>2 preferred)

## Organism-Specific Notes

### Volvox carteri (ch424)
- Spherical colony, ~100-500 um diameter
- BF: colony appears darker than background (threshold below median)
- Nucleus-channel: somatic cells visible as green dots (chlorophyll)
- Membrane-channel: colony boundary + daughter colonies
- Swimming speed: ~10-20 um/s (simulation); real: 100-500 um/s
- Rotation: ~10s period; driven by flagellar beating
- Somatic cell counting: `peak_local_max(min_distance=3, threshold_abs=10)` within colony mask
  - Count is very stable across rotation angles (~158, range 146-168)
  - This is 2D projection count; total ~2x visible for opaque sphere

### C. elegans (ch412)
- See c_elegans.md playbook
- Track PHARYNX (brightest GFP blob), not body COM
- Body undulation adds ~45% noise to speed estimates

### Bacteria (motility assay)
- Very small at low mag (1-3 px at 10x)
- Need high frame rate for speed; Hungarian matching for multi-target tracking
- Short intervals (0.15-0.2s) improve tracking accuracy

## Key Modules
- `src/workflows/stage_tracking.py`: track_target, spiral_search, make_tracker
- `src/analysis/temporal.py`: fft_spectrum, measure_periodic_rate (for rotation/oscillation)
- `src/detection/cells.py`: detect_cells (general-purpose; write custom detect_fn for specific organisms)
