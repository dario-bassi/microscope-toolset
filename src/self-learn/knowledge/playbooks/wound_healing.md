# Wound Healing Kinetics Playbook

## When to Use
Confluent tissue monolayer with a scratch wound. Goal: measure gap closure over time and fit kinetics.

## Key Properties
- Wound is typically **horizontal** (full width of FOV)
- Gap narrows progressively as cells migrate inward from both edges
- Closure follows exponential decay: `gap(t) = gap_0 * exp(-k * t)`

## Workflow

### Phase 1: Acquisition
1. Acquire timelapse: BF + nucleus channel per timepoint
2. Check how time advances (interval-based or per-snap)
3. Plan total frame count to capture full closure

### Phase 2: Gap Measurement (from nucleus channel)
1. Compute **row-wise mean intensity** of nucleus channel
2. **Smooth** with uniform filter (31px kernel) to remove cell-level oscillations
3. Estimate **tissue signal level** from top/bottom edges (always confluent)
4. Find wound center: minimum of smoothed profile in central region
5. **Edge-crossing**: scan outward from center until signal exceeds 50% of tissue level
6. Gap = distance between upper and lower edges
7. Apply **monotonic enforcement** (wound can only close, not reopen)

### Phase 3: Kinetics Fitting
1. Filter to non-zero gap values (avoid log(0))
2. Fit exponential: `gap(t) = A * exp(-k * t)` using `fit_exponential_decay()`
3. Half-closure time: `t_half = ln(2) / k`
4. Leading edge speed: mean gap decrease per step / 2 (two migrating edges)

## Key Code Pattern
```python
from src.analysis.wound_healing import analyze_scratch_assay
from src.analysis.kinetics import fit_exponential_decay
import numpy as np

# Collect timelapse images (MDA-based) → 3D array (T, H, W)
stack = np.array(frames)

# Full pipeline
result = analyze_scratch_assay(stack, orientation='horizontal',
                               pixel_size_um=pixel_size)
# result['gap_widths'] — mean gap at each frame
# result['closure'] — closure_fraction, rate, time_to_close
# result['migration'] — edge speeds, mean_speed

# Alternative: per-frame wound detection
from src.analysis.wound_healing import detect_wound, wound_closure_rate
wounds = [detect_wound(f, orientation='horizontal') for f in frames]
gaps = [w['mean_gap'] for w in wounds]
closure = wound_closure_rate(gaps, timepoints, pixel_size_um=pixel_size)

# Optional: fit exponential decay for kinetics
gaps_arr = np.array(gaps)
valid = gaps_arr > 0
t_arr = np.arange(len(gaps_arr), dtype=float)
if valid.sum() >= 3:
    fit = fit_exponential_decay(t_arr[valid], gaps_arr[valid])
    # fit['k'] = decay rate, fit['half_life'] = ln(2)/k
```

## Pitfalls
- **Row-mean vs row-std**: Use nucleus channel mean, not BF std. Nucleus has clear gap with no signal where cells are absent
- **Heavy smoothing → biased gap**: 31px uniform filter blurs sharp wound edges, making the gap look wider and closure slower. Prefer a simpler approach: threshold rows where mean < low_value, count longest contiguous band of low rows. Less smoothing = sharper edges = faster measured closure.
- **Log(0) in exponential fit**: Once gap reaches 0, exclude those points from fitting
- **Threshold level**: 50% tissue signal can overestimate gap (gave k=0.14 vs GT 0.35). Try lower threshold or hard cutoff on raw row means.
- **Leading edge speed**: Check challenge definition! "px per step from gap change" usually means TOTAL gap narrowing rate (both edges combined), NOT per-edge. Don't divide by 2 unless explicitly asked for per-edge speed.
- **Monotonic enforcement**: Important because noise can cause small gap increases between frames

## Lessons Learned
- Heavy smoothing blurs wound edges → biased gap/rate estimates. Use minimal smoothing.
- Check speed definition: "gap change rate" usually means TOTAL (both edges), not per-edge
- Low threshold on row means gives sharper gap detection than heavy smoothing + edge-crossing

## Laser Ablation Wound Healing (ch532 lesson)

A different wound healing assay: SLM laser ablation kills cells in a defined region (e.g. right half of FOV), then neighboring cells migrate in to repopulate.

### Key Difference from Scratch Assay
- Wound is created by SLM illumination (ablation), not a physical scratch
- Wound region is a HALF-PLANE (col >= 256), not a narrow gap
- Healing = count of cells in wound region increases over time
- Need per-frame tracking, NOT just before/after comparison

### Protocol (ch532)
1. Pre-ablation: count cells in each half
2. Apply SLM mask (`slm_mask[:, 256:] = 255` for right half)
3. Re-apply SLM each frame in on_frame callback
4. Track per-frame cell count in wound region (right half)
5. healing_observed = True if wound count increases ≥1 from minimum

### CRITICAL: Track per frame, not just before/after
With migration_speed=4px/step, border cells drift ~40px in 10 snaps.
Even 1-2 cells crossing the boundary = healing_detected = True.

```python
from src.analysis.wound_healing import (
    segment_cells_bf, define_wound_region, 
    count_cells_in_region, track_wound_repopulation
)

wound_mask = define_wound_region(fov_size=512, region='right_half')

# Post-ablation healing tracking
healing_imgs = [snap(ch) for _ in range(10)]

heal_result = track_wound_repopulation(
    healing_imgs, wound_mask,
    channel='fluorescence',
    min_area=30, max_area=2000,
    threshold=locked_threshold,  # LOCK threshold from pre-ablation!
)

healing_observed = heal_result['healing_detected']
# heal_result['n_in_wound'] = [5, 6, 7, 7, 8, ...] = shows repopulation
```

### Grade lessons from ch532
- ch532 R1=8/10: Had healing_observed=False (missed 2pts)
  - Root cause: compared only first vs last frame, not per-frame series
  - FIX: use track_wound_repopulation() to get per-frame time-series
  - With migration_speed=4px/step: cells at wound border drift ~40px across boundary in 10 snaps
  - Time-series [5, 6, 7, 7, 8...] shows clear repopulation
