# Perturbation Kinetics Playbook

## When to Use
Any experiment with baseline → perturbation → (optional recovery) design.
Examples: cold shock, drug washout, temperature shift, osmotic shock,
photobleaching (FRAP), optogenetic activation/deactivation.

## General Protocol

### Phase 1: Baseline
- 5-10 frames at control conditions (stable signal)
- Establish baseline mean and variability
- Use this to set FIXED foreground threshold for entire experiment

### Phase 2: Perturbation
- Apply stimulus (temperature change, drug, light)
- Acquire timelapse until signal reaches steady state
- **Take MORE frames than you think you need** (≥40)
- 1s intervals for fast responses, 5-10s for slow
- Watch for plateau — stop only when stable for ≥5 frames

### Phase 3: Recovery (if applicable)
- Reverse stimulus (rewarming, washout, light off)
- Continue acquiring until recovery plateaus or timeout
- ≥30 frames typical for recovery phase

## Analysis

### Key Metrics
```python
from src.core.analysis.treatment import compute_half_time, recovery_fraction, temporal_response

# 1. Max response (% change)
baseline_I = np.mean(baseline_values)
equil_I = np.mean(perturb_values[-5:])
max_change_pct = abs(baseline_I - equil_I) / baseline_I * 100

# 2. Half-time
ht = compute_half_time(cold_times, cold_intensities, baseline_value=baseline_I)
half_time = ht['half_time']  # seconds

# 3. Recovery fraction
rec_frac = recovery_fraction(baseline_I, equil_I, recovery_30s_I)
recovery_pct = rec_frac * 100
```

### Intensity Tracking — Use `track_mean_intensity()`
```python
from src.core.analysis.temporal import track_mean_intensity

# Option A: Whole-image mean (PREFERRED for half-time measurements)
result = track_mean_intensity(timelapse_stack, method='whole_image')
intensities = result['values']  # 1D array, one per frame

# Option B: Fixed-threshold foreground mean (higher SNR, sparse cells)
result = track_mean_intensity(timelapse_stack, method='foreground_fixed')
# Threshold auto-set from first 5 frames; or pass threshold=10.0

# Option C: Adaptive Otsu per frame (tracks dim signal, but masks real change)
result = track_mean_intensity(timelapse_stack, method='foreground_adaptive')
```
- **whole_image**: avoids threshold-crossing nonlinearity. Best for half-time.
- **foreground_fixed**: excludes background → higher SNR, but introduces step-down
  artifact when dim pixels cross below threshold (see pitfalls).
- **foreground_adaptive**: per-frame Otsu. Tracks even dim signal but may mask
  true decrease (adaptive thresholds adjust to signal).

### Temperature Control
```python
# Temperature states: 0=20°C, 1=4°C, 2=25°C, 3=30°C, 4=37°C, 5=42°C
core.setState("Temperature", 1)   # 4°C
core.setState("Temperature", 4)   # 37°C
```

## Sample-Specific Notes

### Actin Cytoskeleton (Cold Shock / Cytochalasin D)
- **Phalloidin-fluorescent or LifeAct-GFP** labels F-actin (stress fibers)
- Cold (4°C) disrupts dynamic actin assembly → fibers depolymerize
- Cytochalasin D inhibits polymerization → similar effect
- **Baseline**: organized stress fibers, high structured intensity
- **Perturbation**: diffuse signal, loss of fiber structure
- **Recovery**: fibers re-form (minutes to hours at 37°C)
- Detect with: `src/analysis/cytoskeleton.py` (fiber orientation, count)
- Intensity tracking: whole-image mean tracks depolymerization well
- Half-time: typically 2-10 min for cold-induced depolymerization

### Stress Granule Formation (Arsenite / Heat / Osmotic Stress)
- See `Stress granule formation.md` playbook (detailed)
- Key: use noise-floor thresholding, not arbitrary multipliers

### NF-kB Nuclear Translocation (TNFα / LPS / Drug Stimulation)
- NF-kB-GFP reporter: cytoplasmic at rest (dark nuclei), nuclear after stimulus
- **Segment nuclei from DAPI**, NOT from GFP (GFP distribution changes!)
- Cytoplasm mask: dilate nuclei (25x25 structuring element), then `segment_cytoplasm()`
- `compute_nc_ratio(gfp_img, nuc_labeled, cyto_mask)` → N:C ratio per frame
- Baseline N:C ~0.5 (cytoplasmic), peak ~2-5 (nuclear accumulation)
- Half-time: use `compute_half_time(times, nc_values, baseline_value=baseline_nc)`
- Import typically 3-15 min; export (washout) 15-60 min
- Drug delivery: `core.setState("Perfusion", 4)`, washout: `setState("Perfusion", 0)`
- Key: snap() loops (not MDA) when device state changes between phases

### FRAP (Fluorescence Recovery After Photobleaching)
- `src/analysis/frap.py`: normalize_frap, fit_frap_recovery, mobile_fraction
- Bleach in spot → measure recovery → diffusion coefficient
- Key: normalize to pre-bleach AND post-bleach immediately

## Pitfalls
- **Threshold-crossing nonlinearity** (ch553=8/10, cost ~2 pts on half-time):
  As signal dims, pixels cross below threshold and suddenly vanish from the
  foreground mean instead of gradually decreasing. This makes apparent half-time
  FASTER than reality. For half-time: prefer whole-image mean or use a very low
  threshold to minimize this artifact.
- **Adaptive threshold masks real change**: Fixed threshold reveals true dynamics
- **Not enough frames**: Run until plateau, not just "about 40 frames"
- **Wrong channel**: Use the specific marker (phalloidin for actin, not BF)
- **Forgetting to measure recovery**: Even if not explicitly asked, recovery data adds value
- **numpy type serialization**: Cast to Python float() before submitting
- **Washout time ≠ last frame**: When challenge asks "N:C ratio 30s after washout",
  interpolate to exactly 30s — do NOT use the last frame (which may be at 79s if
  frames take 2.6s each). Use `np.interp(30.0, export_times, export_nc)`.
- **Peak includes lag**: Drug effect continues briefly after washout. Report the
  actual peak across ALL frames (import + early export), not just the import phase.
- **Frame timing drift**: Network snap + compute takes ~2.5s per frame, not 1s.
  Record actual wall-clock times for each frame, not assumed intervals.
- **PSF compression of N:C ratio** (ch555=7/10): At 20x, PSF spreading dilutes
  concentrated nuclear signal into surrounding cytoplasm pixels, compressing the
  measured N:C dynamic range by ~2x vs the true biochemical ratio. This affects
  ALL derived quantities (max_nc, half_time, washout_nc). To mitigate: use 40x
  (PSF~0.5px instead of ~2px), or apply deconvolution before measuring N:C.
