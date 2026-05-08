# Drug Response Timelapse Playbook

## When to Use
Single-field timelapse where a drug is applied mid-experiment. Observe
intensity/structure changes over time and identify the drug from phenotype.

## Critical: Multi-Scale First

**ALWAYS snap a low-mag overview BEFORE starting the timelapse.**

At high magnification, the FOV may only cover a few cells. If the task asks for
total cell count + per-cell detail, overview first is mandatory.

```python
# Step 1: Low-mag overview for cell count + positions
from useq import MDAEvent
from src.core.hardware.core import run_events

overview_events = [MDAEvent(channel={"config": overview_channel}, index={"t": 0})]
overview = run_events(core, overview_events)
# Count cells, record positions

# Step 2: Switch to high-mag for temporal detail (or stay at low-mag for global tracking)
```

## Workflow

### Phase 1: Overview + Cell Count
1. Snap low-mag overview (BF or nucleus channel)
2. Count total cells, record their positions
3. Decide magnification for timelapse: low-mag for intensity, high-mag for structure

### Phase 2: Timelapse Acquisition
Use MDA events:
```python
events = [MDAEvent(channel={"config": primary_channel}, index={"t": t})
          for t in range(n_frames)]
frames = run_events(core, events, on_frame=track_callback)
```

Use the PRIMARY channel for the timelapse. Snap other channels at start/end only.

### Phase 3: Analysis
1. **Intensity tracking**: Mean foreground intensity per frame
2. **Onset detection**: First frame where intensity drops below X% of baseline
3. **Structure counting**: Use `count_structures()` with FIXED baseline threshold
4. **Drug classification**: Match phenotype to drug profile

## Intensity Tracking Method Selection (CRITICAL)

Choose the right method for what you're measuring:

| Method | When to Use | When NOT to Use |
|--------|-------------|-----------------|
| `whole_image` mean | **Half-time measurement**, recovery curves | When background dominates |
| `foreground_fixed` | Counting, segmentation-based metrics | **Half-time** (threshold artifact) |
| `foreground_adaptive` | Relative comparisons within frame | **Temporal comparison** (masks change) |

**Why `foreground_fixed` is wrong for half-time**: As intensity drops, pixels cross below
the fixed threshold and vanish from the mean. This creates a step-down artifact that makes
the half-time appear ~15% faster than ground truth (ch553 lesson: 15.8s vs 19.0s).

For half-time, use `whole_image` mean or `track_mean_intensity(method='whole_image')`.

## Analysis Patterns

### Intensity Ratio (post/pre)
```python
# Foreground-only mean (excludes background)
intensities = [np.mean(img[img > bg_threshold].astype(float)) for img in frames]
pre = np.mean(intensities[:onset])
post = np.mean(intensities[-5:])
ratio = post / pre
```

### Onset Detection
```python
baseline = np.mean(intensities[:3])
diffs = np.diff(intensities)
onset = next((i for i in range(len(diffs)) if diffs[i] < -threshold), 3)
```

### Structure Counting with Fixed Threshold
```python
from scipy.ndimage import label

# Get baseline threshold ONCE from the first frame (lock the same threshold
# across timepoints — drift in per-frame threshold confounds the kinetic).
threshold = np.percentile(frames[0][frames[0] > bg], 70)

def count_structures(frame, threshold):
    _, n = label(frame > threshold)
    return n

baseline_count = count_structures(frames[0], threshold)
final_count = count_structures(frames[-1], threshold)
```

## General Drug Classification Strategy

When classifying an unknown drug from phenotype, use BOTH intensity loss AND
structural change. When they disagree, trust the structural change (more specific).

**Severity ranking approach:**
- Rank drugs by expected severity (from literature/experiment description)
- Match measured loss to the closest severity tier
- Use multiple metrics (intensity, structure count, morphology) for confirmation

## N:C Ratio Measurement (Translocation Assays)

For nuclear translocation assays (NF-kB, NFAT, etc.):

```python
from src.core.analysis.nuclear_cytoplasmic import compute_nc_ratio
# Auto guard_band from pixel size — prevents PSF compression
result = compute_nc_ratio(img, nuc_mask, cyto_mask, pixel_size_um=ps)
```

**PSF compression at low magnification** (ch555 lesson):
- At 20x (0.5 µm/px), PSF spreading dilutes nuclear signal into cytoplasm
- Measured max_nc can be compressed by ~2x (6.0 vs GT 11.9)
- Use `pixel_size_um=` parameter for automatic guard_band
- Or switch to 40x for quantitative N:C measurements

**Exact-time interpolation** (ch554 lesson):
```python
from src.core.analysis.treatment import value_at_time
washout_nc = value_at_time(timepoints, nc_ratios, target_time=30.0)
```
Don't use the last frame value if the target time differs from frame timing.

## Key Lessons

1. **Intensity ratio is robust** — straightforward to measure accurately with foreground masking
2. **Structure counts need multi-scale** — limited FOV at high-mag is the #1 error source
3. **Fixed thresholds for temporal comparison** — adaptive thresholds track signal down,
   masking true structural loss
4. **Drug ID from phenotype**: Use BOTH intensity loss AND structural change for classification
5. **whole_image for half-time** — foreground_fixed thresholding introduces step-down artifact
6. **PSF guard_band for N:C** — use `pixel_size_um=` in compute_nc_ratio() at ≤20x
7. **value_at_time for exact targets** — interpolate, don't use nearest frame
