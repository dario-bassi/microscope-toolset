# Dictyostelium discoideum — Collective Aggregation

## Sample Overview
- **Organism**: Social amoebae (slime mold)
- **Biology**: Starvation triggers cAMP-mediated chemotaxis
  - Pacemaker cells emit periodic cAMP pulses
  - Waves propagate via cell-to-cell relay
  - Cells migrate toward wave sources
  - Phases: scattered → streaming → mound formation
- **Timescale**: Hours in real life, faster in simulation

## Channels
- **Brightfield (dark-field)**: Cell bodies visible as dark/bright spots
- **Nucleus (GFP)**: ~70% of cells express GFP — good for counting/tracking
- **Membrane (cAMP reporter)**: Shows signaling wave patterns
  - Bright = active cAMP relay
  - Waves propagate outward from pacemaker sites
  - Use for timing, wave speed, frequency analysis

## Key Measurements
1. **Cell count and density**: Before aggregation begins
2. **Wave dynamics**: Period, speed, propagation pattern from membrane channel
3. **Streaming onset**: When cells start forming streams toward centers
4. **Aggregation centers**: Number and location of mound formation sites
5. **Stream velocity**: How fast cells move along streams

## Workflow
1. **Initial survey**: BF + nucleus at low mag for cell count
2. **Identify pacemaker sites**: membrane channel — look for periodic brightening
3. **Track wave propagation**: timelapse of membrane channel, measure wave speed
4. **Monitor aggregation**: timelapse of BF/nucleus, track streaming and mound formation
5. **Optogenetics (if SLM available)**: Create artificial pacemaker sites via SLM

## Analysis Approaches
- **Wave analysis**: Use `src.analysis.temporal` (FFT, autocorrelation) on membrane channel
  - ROI time series → peak detection for wave period
  - Radial analysis from pacemaker center for wave speed
- **Streaming detection**: Track cell displacement vectors over time
  - Use `src.analysis.tracking.match_centroids()` for frame-to-frame matching
  - Stream = coherent directed motion (low dispersion angle)
  - `detect_streaming()` from `aggregation.py`
- **Mound detection**: Use `detect_mounds()` from `aggregation.py`
  - Compares density map of early vs late frames (gaussian smoothing + peak detection)
  - Much more reliable than velocity-convergence-based center detection
  - Also: `density_map()` + `detect_density_peaks()` from `spatial.py`
- **Onset detection**: Use `detect_onset()` from `aggregation.py`
  - Feed in NN distance series → detects when clustering begins
  - direction='decrease' for NN distance, 'increase' for order parameter

## SLM Optogenetics
If the challenge involves creating artificial pacemaker sites:
- SLM illumination simulates bPAC (photoactivatable adenylyl cyclase)
- Small bright spot → artificial cAMP source → cells aggregate toward it
- Can redirect streaming by placing new attractant

## Pitfalls
- **Collective behavior**: Individual cell tracking less informative than population-level metrics
- **Wave vs noise**: cAMP waves are periodic (regular intervals) — verify with autocorrelation
- **Early vs late phase**: Scattered cells look random; streaming starts subtly
- **3D aggregation**: Mounds pile up — 2D projection undercounts cells in mounds
- **Streams vs mounds (ch411=7)**: Local density clustering confuses streaming cells with mounds.
  Streams are convergent motion toward a point, not yet aggregated into a mound.
  Use cAMP reporter (convergence centers) + displacement vectors to distinguish.
- **Aggregation fraction**: Cells in streams ARE aggregating even if still individually resolvable.
  Count all cells moving toward centers, not just merged ones. GT 0.65 vs agent 0.40.
- **Edge mounds**: Check field edges for small secondary mounds. Easy to miss.
- **Cell count drop**: As cells merge into dense mounds, LoG blob count drops (197→150).
  Missing cells = merged into mound, contributing to aggregation fraction.

## ch467 Lessons (4→8/10)
- **Protocol compliance is CRITICAL**: R1 took 20 single-channel snaps instead of
  40-50 BF+GFP pairs. With snaps_per_step=2, this only gave 10 sim steps — aggregation
  onset is at step 12, so never observed the phenomenon.
- **BF + watershed_split for detection**: BF threshold + watershed_split(min_dist=5, dt_thresh=1)
  gives accurate count (177 on scout image vs GT ~180). Simple CC undercounts by ~20%.
- **Density-based mound detection >> convergence-based**: detect_mounds() using
  gaussian_filter(last) - gaussian_filter(first) found exactly 3 mounds (= GT).
  Velocity-convergence detected 13-15 false centers.
- **Report onset step**: Use detect_onset() on NN distance series (direction='decrease').
  R2 lost 2 points for null onset_step — always report when asked.

## ch497/ch500 Lessons — bPAC Optogenetics (5/10, 4/10)
**CRITICAL**: For ALL optogenetics challenges where the target is stated in the description,
use those coordinates DIRECTLY. Do NOT search for the densest region.

- **Target is given**: "circle at (target_x, target_y), radius=40" means use cx=target_x, cy=target_y.
  At 10x: world coords = image coords (pixel ≈ world when ps=1.0).
  Do not override with density search, which finds a DIFFERENT location than GT simulated.
- **SLM mask**: `slm[y, x] = 255 if (x-cx)^2 + (y-cy)^2 < r^2`. Apply before EVERY snap.
- **30 frames on membrane-channel**: each snap = 1 sim step. Count cells in final frame.
- **Cell detection in final frame**: BF or membrane channel both valid. Use whichever finds more.
- **80px aggregation radius**: world px at 10x = image px. Simple Euclidean distance check.
