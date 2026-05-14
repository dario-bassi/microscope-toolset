# Pitfall: Run-and-Tumble Tracking

> **When to use:** When motility tracking produces far more tracks than objects, or tumble frequency is much higher than expected.

When analyzing trajectories of organisms (or motile cells) that alternate between directed motion and brief reorientation events, two compounding methodology errors can turn a well-detected population into a thoroughly wrong frequency measurement.

## Failure modes

### 1. Track fragmentation (far more tracks than moving objects)
- Hungarian matching broke tracks at direction-change events (tumbles confused the matcher)
- Detection dropouts during re-orientation created gaps
- Default `max_gap=0` meant any missing detection terminated a track
- **Fix**: allow small gap tolerance (`max_gap=1` or `2`) in track building

### 2. Tumble overclassification (apparent frequency higher than true frequency)
- Angle threshold too low relative to detection noise
- At the pixel scale of mid-magnification imaging with short dt, ±1 px detection noise can manifest as ~15–20° apparent angle changes
- Without temporal filtering, single-frame direction noise falsely triggered tumble classification
- **Fix**: set `angle_threshold` above the noise angle, and require consecutive frames (`min_tumble_frames=2`)

### 3. Cascading downstream errors
Tumble overclassification → inflated frequency, depressed run-fraction. Track fragmentation → inflated n_tracked. The symptoms compound — a calibration error in one metric warps the others.

## Root cause

The two parameters (`angle_threshold`, `min_tumble_frames`) were tuned for a different imaging regime than the one being used. The rule: when combining angle-based classification with noisy centroid measurements, require the threshold to exceed the 95th-percentile noise angle, and require temporal persistence to reject single-frame noise events.

## Lessons that transfer

1. **Domain-appropriate thresholds.** Pick the angle threshold from the organism's known reorientation statistics (e.g., typical mean turn angle), not from a generic default. Different organisms need different values.
2. **Temporal filtering against noise.** Require ≥2 consecutive candidate frames for any event classified from per-frame comparisons. Trades a small recall cost for a large false-positive reduction.
3. **Always validate track counts** against expected unique-object counts. A track-to-detection ratio >2 means fragmentation is eating the analysis.
4. **Gap tolerance.** Brief detection dropouts during state transitions (tumbling, dividing, z-drifting) are normal. Allow 1-frame gaps in track building.

Organism-specific parameter tuning: start with `angle_threshold` at twice the expected single-frame noise angle and require `min_tumble_frames=2`.
