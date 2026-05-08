# Single-Particle Tracking (SPT) Diffusion Analysis

**Assumes:** 100x oil objective for single-particle resolution, high frame rate (≥10 Hz), fluorescent point sources. The pixel-scale numbers below are the simulator's (world 12.8×12.8 µm / camera 512×512); on a real prep, pull both from `cfg.pixel_size_um` and stage-coordinate conversion.

## Overview
SPT traces individual fluorescent particles to measure membrane receptor dynamics.
Three motion types:
- **FREE** (Brownian): linear MSD → slope = 4D, α ≈ 1.0
- **CONFINED**: plateau MSD → r²_corral = (3/4) × plateau, α < 0.7
- **DIRECTED**: parabolic MSD → quadratic term, α > 1.3

## Pixel Scale (100x, sim-specific)
- World: 128×128 px = 12.8×12.8 µm (1 world px = 0.1 µm)
- Camera: 512×512 px = 8 camera px per world px
- **1 camera px = 0.0125 µm = 0.125 world px**
- MSD in µm²: `MSD_cam_px² × (0.0125)²`

## σ Step Size Sanity Check
ALWAYS verify before linking:
```
σ_step (cam px) = sqrt(2*D*dt) / px_scale_um
```
- D=0.1 µm²/s, dt=0.1s, px=0.0125 → σ = sqrt(0.02)/0.0125 = 11 cam px
- max_link should be 2-3 × σ = 25-35 cam px
- If σ >> max_link: FREE particles can't be tracked (design flaw)

## Blob Detection Threshold — CRITICAL
Threshold affects what you see and how stable detection is:
- **threshold=0.4**: 1-19 spots/frame (heavy "blinking"), frames with 1 spot break linking
- **threshold=0.3**: 12-30 spots/frame (stable), almost no gaps
- Use **threshold=0.3** as default for SPT
- Validate: frames should have stable counts (min/max ratio < 3x)

## Acquisition
```python
seq = MDASequence(
    time_plan={"loops": 50, "interval": 0.1},
    channels=[{"config": "spt-channel", "group": "Fake"}]
)
frames = []
run_events(core, list(seq), on_frame=lambda img, e: frames.append(
    img[:,:,0].astype(float) if img.ndim==3 else img.astype(float)
))
# Each snap advances simulation by 100ms (fixed_dt=0.1s)
```

## Spot Detection
```python
# Normalize to 0-1 FIRST, then apply threshold
f_norm = frame / max(frame.max(), 1e-6)
blobs = blob_log(f_norm, min_sigma=0.8, max_sigma=3, threshold=0.3)
spots = blobs[:, :2]  # (row, col) in camera px
```

## Track Linking (Fixed Hungarian)
Key insight: reject cost must be BETWEEN valid match cost and "far" cost.
```python
REJECT_COST = max_dist + 1  # slightly above threshold (not max_dist * 1.5!)
FAR_COST = REJECT_COST * 3  # very high cost for impossible matches

# Cost matrix (n_active × n_spots)
cost_aug = np.full((n_active + n_spots, n_active + n_spots), FAR_COST)
cost_aug[:n_active, :n_spots] = distance_matrix
# Reject options
for i in range(n_active):
    cost_aug[i, n_spots + i] = REJECT_COST   # track → no match
for j in range(n_spots):
    cost_aug[n_active + j, j] = REJECT_COST  # spot → new track

row_ind, col_ind = linear_sum_assignment(cost_aug)
```

## MSD Computation
```python
# For track xy (N×2 camera px array):
for tau in range(1, N//2):
    diffs = xy[tau:] - xy[:-tau]
    msd_um2[tau-1] = np.mean(diffs**2) * CAMERA_PX_TO_UM**2
lags = np.arange(1, N//2) * DT
```

## Classification
```python
# Log-log fit for anomalous diffusion exponent
alpha = np.polyfit(np.log(lags), np.log(msd_um2), 1)[0]
if alpha < 0.7:
    motion = 'CONFINED'
    plateau = msd_um2[max_lag//2:].mean()
    r_corral = sqrt(3 * plateau / 4)  # µm
elif alpha > 1.3:
    motion = 'DIRECTED'
    # MSD = 4D*t + v²*t² (fit parabolic)
else:
    motion = 'FREE'
    D = linear_slope[:5] / 4  # µm²/s from first 5 lags
```

## Common Pitfalls

### Wrong pixel scale (ch517 lesson)
- D=1.0 µm²/s gave σ=36 cam px → exceeded max_link=24 → free particles UNTRACKED
- Always check: σ_step << max_link

### Simulation step size bug (ch520 lesson)
- GT computation via `setup_spt_microscope()` overwrote global bridge
- Particles appeared stationary (static sim, no auto_step)
- Fix: use `SPTSim()` directly for GT, not `setup_spt_microscope()`

### Heavy blinking at threshold=0.4 (ch522 lesson)
- Frames with only 1-2 spots cause linking failures for all tracks
- Fix: **use threshold=0.3** → stable 12-30 spots/frame with minimal blinking
- Validate: check min/max spot count ratio before linking

### Wrong reject cost in Hungarian (ch522 bug)
- Using `REJECT_COST = max_dist * 1.5 = 45` while valid matches are at 6 px
- This made "reject" CHEAPER than "match wrong particle at 6 px" → 0 tracks formed
- Fix: `REJECT_COST = max_dist + 1` so valid close matches always win

### D_free lower than expected
- If measured D << expected: check threshold stability first
- If both frames have 20+ spots, mean step should match σ prediction
- If step mean is 5 px but expected 11 px: simulation may have different D than stated
- Report bug to virtual-env if physics don't match description

### Only confined particles tracked
- Confined particles (small steps) always stay within max_link
- Free particles (large steps) sometimes link to wrong particle
- Result: overcount confined, undercount free
- Use larger max_link (2.5σ) and gap tolerance

### Directed particles rarely detected
- Need alpha > 1.3 reliably
- Directed: parabolic MSD, constant velocity
- Often misclassified as FREE if short track

## Expected Values (membrane receptors)
- D_free: 0.05-0.15 µm²/s (slow receptor complexes in lipid bilayer)
- D_confined: 0.01-0.05 µm²/s (inside corral)
- r_confinement: 0.2-0.6 µm (corral size, lipid raft or cytoskeletal barrier)
- fraction_free: ~50%
- fraction_confined: ~30%
- fraction_directed: ~20%

### Particle FOV exit problem (ch522 lesson)
- At 100x: FOV = 64×64 world px (center of 128×128 world)
- Particles initialized across entire world → may start OUTSIDE FOV
- When particles exit/enter FOV, nearest-neighbor linker confuses departing + arriving particles
- Result: apparent steps MUCH smaller than true steps (4.7 vs 15.9 cam px)
- **True particle steps must be ≥ max_link** to be tracked (this is physics, not a bug)
- Fix (by virtual-env): initialize particles inside FOV region only
- Verification: check that median nearest-neighbor step matches σ_expected
  - If step << σ_expected: FOV exit confusion or wrong objective

### FISH deleted cell detection
- Standard percentile threshold (P10) may miss the 2-3 truly deleted nuclei
- Deleted nuclei have near-ZERO fish signal (not just low signal)
- Better approach: look for nuclei with 0 spots at 40x (truly empty)
- Or: GMM clustering to find a distinct "empty" population
