# FRAP (Fluorescence Recovery After Photobleaching) Playbook

**Assumes:** SLM-equipped scope for local photobleaching, fluorescent mobile species, timelapse long enough to capture recovery. Paired with `src/recipes/frap_background_correction.py` (`analyze_frap` — bleach-corrected mobile fraction + frame-index half-life, with `expected_frames` truncation guard from sprint #15).

## Principle
Bleach a small region with high-intensity light (via SLM), then monitor fluorescence
recovery as unbleached molecules diffuse in. Extract:
- **Half-time (t½)**: time to recover 50% of lost fluorescence
- **Mobile fraction**: % of molecules that recover (some may be immobile/bound)
- **Diffusion coefficient**: D = w² / (4 × t½) where w = bleach spot radius

## Step-by-Step Protocol

### 1. Setup and scouting
```python
from src.core.hardware.core import snap, make_slm_circle, run_events
from src.core.hardware.config import get_config

# Scout channels — find the fluorescent channel to bleach
from src.core.workflows.scouting import channel_scout
scout = channel_scout(core)
# Choose channel with fluorescence signal
```

### 2. Pre-bleach baseline (5-10 frames)
```python
from useq import MDASequence, MDAEvent

# Snap pre-bleach images to establish baseline intensity
baseline_events = [MDAEvent(channel={"config": fluor_channel}) for _ in range(5)]
baseline_frames = run_events(core, baseline_events)

# Define bleach ROI (circle at chosen location)
cx, cy = 256, 256  # center of FOV, or on a specific structure
radius_px = 20     # bleach spot radius
yy, xx = np.ogrid[:512, :512]
roi_mask = (xx - cx)**2 + (yy - cy)**2 <= radius_px**2

# Measure pre-bleach intensity in ROI
pre_bleach = np.mean([f[0][roi_mask].mean() for f in baseline_frames])
```

### 3. Bleaching pulse (SLM)
```python
from useq import SLMImage
slm_mask = make_slm_circle((cx, cy), radius_px, size=512)
slm = SLMImage(data=slm_mask, device="SLM")

# Single bleach event with high exposure
bleach_event = MDAEvent(
    channel={"config": fluor_channel},
    slm_image=slm,
    exposure=500,  # high exposure for bleaching
)
bleach_result = run_events(core, [bleach_event])
```

### 4. Recovery timelapse
```python
# Monitor recovery with regular intervals (NO SLM)
interval_s = 0.5  # adjust based on expected t½
n_recovery = 40   # enough frames to see full recovery

recovery_events = [
    MDAEvent(
        channel={"config": fluor_channel},
        min_start_time=i * interval_s,
    )
    for i in range(n_recovery)
]
recovery_frames = run_events(core, recovery_events)

# Extract ROI intensity over time
recovery_intensities = [f[0][roi_mask].mean() for f in recovery_frames]
times = np.array([i * interval_s for i in range(n_recovery)])
```

### 5. Fit recovery curve
```python
from src.core.analysis.kinetics import fit_exponential_recovery

result = fit_exponential_recovery(
    times, recovery_intensities,
    max_plateau=pre_bleach * 1.05  # constrain to pre-bleach level
)

half_life = result['half_life']
mobile_fraction = result['recovery_pct'] / 100
I_inf = result['I_inf']
```

### 6. Calculate diffusion coefficient
```python
cfg = get_config(core)
radius_um = radius_px * cfg.pixel_size_um
D = (radius_um ** 2) / (4 * half_life)  # um²/s
```

## Key Parameters

| Parameter | Typical range | Notes |
|-----------|--------------|-------|
| Bleach depth | 50-80% of pre-bleach | Too shallow = noisy recovery; too deep = may damage |
| Recovery interval | 0.1-2s | Match to expected t½ (sample 20+ points during recovery) |
| Mobile fraction | 0.5-1.0 for cytoplasm | <0.5 suggests significant immobile fraction |
| D (cytoplasmic) | 1-50 um²/s | Small proteins ~10-30, large ~1-5, lipid bilayer ~0.1-1 |

## Common Pitfalls
- **Analysis ROI = bleach ROI**: Edge pixels are only partially bleached (Gaussian profile), recover faster, and skew the fit toward faster kinetics and lower mobile fraction. **Use analysis ROI at 50-70% of bleach radius** for accurate t½ and mobile fraction.
- **Bleach too large**: Recovery takes forever, neighboring structures affected
- **Bleach too small**: Diffusion fills spot before you can measure
- **Photobleaching during recovery**: Use low exposure for monitoring, not the bleach exposure
- **Fit plateau too high**: Use `max_plateau=pre_bleach` constraint
- **Wrong time scale**: If recovery is faster than your frame rate, increase exposure or reduce ROI size

## Existing Tools
- `src/hardware/core.py`: `make_slm_circle()`, `run_events()` with SLM support
- `src/analysis/kinetics.py`: `fit_exponential_recovery()` with max_plateau constraint
- `src/analysis/calcium.py`: `extract_roi_traces()` for multi-ROI analysis
- `src/workflows/optogenetics.py`: `slm_stimulation_experiment()` as 3-phase template
- `src/recipes/frap_background_correction.py`: `analyze_frap_with_guards()` — opt-in wrapper that runs `run_presubmit_guards` on the `corrected_mobile_fraction` submission. Use this when the rendered stack you submit may differ from the stack you analysed (re-acquisition, truncation, ROI drift). Returns the same dict as `analyze_frap` plus `presubmit_severity` + `presubmit_result`. Sprint #22 worked example.
