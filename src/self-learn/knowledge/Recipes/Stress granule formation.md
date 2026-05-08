# Stress Granule Formation Playbook

## When to use
G3BP1-GFP or similar RBP (RNA-binding protein) markers, stress response
assays under arsenite, oxidative stress, heat shock, or osmotic stress.
Any challenge involving puncta formation kinetics.

## Biology
Stress granules (SGs) are cytoplasmic condensates that form when cells
are stressed. They sequester mRNAs and translation factors. Key phases:
1. **Baseline**: diffuse cytoplasmic fluorescence (no puncta)
2. **Stress onset**: puncta begin forming (nucleation)
3. **Growth**: puncta increase in number and size
4. **Steady state**: puncta count plateaus (equilibrium)
5. **Recovery** (if stress removed): puncta dissolve

## Workflow

### 1. Snap overview at 10x
```python
from src.core.hardware.core import snap, snap_all_channels, set_objective_verified
set_objective_verified(core, 10)
channels = snap_all_channels(core)
```
- Identify which channel has SG signal (GFP/YFP for G3BP1)
- Identify cell body channel (brightfield or DAPI)

### 2. Choose magnification
- 20x for population-level counting (stress fraction, onset time)
- 40x for per-cell foci counting and morphology

### 3. Acquire timelapse with MDA
```python
from useq import MDASequence
from src.core.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 40, "interval": 30.0},  # 40 frames, 30s apart
    channels=[
        {"config": "GFP", "group": "Fake"},
        {"config": "brightfield", "group": "Fake"},
    ],
)
results = run_events(core, list(seq))
```
- **Interval**: 15-60s (SGs form over 5-30 min)
- **Duration**: at least 2x expected onset time
- Both fluorescence AND brightfield for cell detection

### 4. Detect foci per frame
```python
from src.core.analysis.condensate import detect_foci, foci_per_cell
from src.core.detection.threshold import estimate_noise_floor

# Measure noise floor from BASELINE frames
noise = estimate_noise_floor(baseline_frame, bg_fraction=0.7)
# Use 1.3-2.0 sigma threshold (NOT arbitrary multipliers)

foci = detect_foci(frame,
    min_sigma=1.0, max_sigma=4.0,
    threshold=0.03,  # adjust based on signal
    background_multiplier=None)  # prefer noise-floor approach
```

### 5. Classify cells as SG-positive
```python
from src.core.analysis.condensate import stress_response_index

# Count foci per cell
per_cell = foci_per_cell(foci_list, cell_labels)

# SG-positive: ≥3 foci per cell (standard threshold)
sri = stress_response_index(per_cell, threshold_foci=3)
# Returns: stressed_fraction, per_cell_counts, n_stressed, n_total
```

### 6. Measure kinetics
- **Onset time**: first frame where stressed_fraction > 0.1 (or 2× baseline)
- **Half-time**: use `compute_half_time()` from `src/analysis/treatment`
- **Max response**: peak stressed_fraction
- Use `detect_changepoint(method='pelt')` for onset detection

## Key Metrics
| Metric | Method | Notes |
|--------|--------|-------|
| Foci count/cell | `foci_per_cell()` | Per-frame |
| Stressed fraction | `stress_response_index()` | SG+ cells / total |
| Onset time | Changepoint or threshold | First frame > baseline + 2σ |
| Half-time | `compute_half_time()` | Time to 50% max response |
| Mean foci area | `detect_foci()` → area field | In px² or µm² |

## Common Pitfalls

### Threshold too strict (ch548: -2 pts)
- Using `background_multiplier=1.8` missed 15% of SG+ cells
- **Fix**: Use `estimate_noise_floor()` → `threshold_1_3sigma` for sensitive detection
- Verify by visual inspection of detected vs missed foci

### Confusing diffuse signal with foci
- Before stress: high mean intensity but NO puncta
- After stress: puncta appear (local maxima above background)
- Detection threshold must distinguish puncta from diffuse cytoplasm

### Not enough baseline frames
- Acquire 5+ frames before stress onset for reliable baseline
- Use these to calibrate noise floor and SG-negative fraction

### Wrong magnification
- At 10x, individual foci may be sub-pixel → undercount
- Use 20x or 40x for foci counting
- 10x OK for population-level stressed fraction

## Key Modules
- `src/analysis/condensate.py`: detect_foci, foci_per_cell, stress_response_index
- `src/detection/threshold.py`: estimate_noise_floor
- `src/analysis/treatment.py`: compute_half_time, recovery_fraction
- `src/analysis/events.py`: detect_changepoint
- `src/analysis/temporal.py`: track_mean_intensity
