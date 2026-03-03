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
from src.hardware.core import run_events

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
from src.analysis.morphology import count_structures

# Get baseline threshold ONCE from the first frame
threshold = np.percentile(frames[0][frames[0] > bg], 70)

# Apply SAME threshold to all frames
baseline_count = count_structures(frames[0], threshold=threshold)['count']
final_count = count_structures(frames[-1], threshold=threshold)['count']
```

## General Drug Classification Strategy

When classifying an unknown drug from phenotype, use BOTH intensity loss AND
structural change. When they disagree, trust the structural change (more specific).

**Severity ranking approach:**
- Rank drugs by expected severity (from literature/experiment description)
- Match measured loss to the closest severity tier
- Use multiple metrics (intensity, structure count, morphology) for confirmation

## Key Lessons

1. **Intensity ratio is robust** — straightforward to measure accurately with foreground masking
2. **Structure counts need multi-scale** — limited FOV at high-mag is the #1 error source
3. **Fixed thresholds for temporal comparison** — adaptive thresholds track signal down,
   masking true structural loss
4. **Drug ID from phenotype**: Use BOTH intensity loss AND structural change for classification

## ⚠️ Real Microscope Considerations

### Drug Perfusion Equilibration Time
**Simulation assumption:** Drug applied instantly; cells exposed immediately.
**Real hardware:** Perfusion system has equilibration lag:
- Tube dead volume: 5–20 μL
- Flow rate: 50–200 μL/min typical
- **Equilibration time: 30–60 seconds** before drug reaches cells at full concentration

**Impact on onset detection:**
- Measured "onset frame" will be significantly later than actual drug addition time
- Intensity/structure changes appear to occur 30–60s into timelapse, not at frame 0

**Adjustments for real hardware:**
- **Start timelapse BEFORE drug addition** → capture baseline 30–60s before drug reaches cells
- Use frame-to-frame diffs to detect actual onset (sharp transition from flat baseline)
- Calculate dose-response kinetics using drug addition time as t=0, not timelapse start
- If drug system has known dead volume V and flow rate Q:
  ```python
  equilibration_time = V / Q  # in seconds
  onset_frame_expected = equilibration_time / frame_interval
  ```
- Validate equilibration time on pilot run with fluorescent tracer dye

### Concentration Gradient During Equilibration
**Real hardware:** Drug concentration in chamber rises gradually during equilibration:
- Initial exposure: partial drug effect
- Full concentration: complete drug effect
- Edge cells may see drug sooner than center (if inlet near edge)

**Impact on measurements:**
- Intensity drop is multi-phase: slow rise (0–30s), then rapid drop (30–60s)
- Structure counts may show gradual decline rather than step-function

**Adjustments for real hardware:**
- Monitor early timeframes (0–30s) separately as "ramp-up" phase
- Use only frames after t_equilibration for reliable drug response classification
- For multi-point dose-response: standardize by using only plateau-phase measurements

### Baseline Drift During Equilibration
**Real hardware:** Even without drug, focus drift, temperature drift, and photobleaching occur during 30–60s equilibration.

**Adjustments:**
- Acquire 10–20 baseline frames BEFORE drug addition to measure baseline drift
- Fit linear/exponential drift to baseline frames
- Subtract fitted drift from post-drug frames to isolate drug effect
- Example:
  ```python
  # Measure pre-drug baseline drift
  baseline_trend = np.polyfit(range(n_baseline), baseline_intensities, 1)
  drift_per_frame = baseline_trend[0]

  # Correct post-drug measurements
  corrected = post_drug_intensities - (drift_per_frame * np.arange(len(post_drug_intensities)))
  ```

### Flow Artifact and Mechanical Stress
**Real hardware:** Perfusion flow can cause:
- Cell displacement (cells move with flow)
- Osmotic stress (hyposmotic solutions cause cell swelling)
- Mechanical stress from shear forces
- May confound drug response measurements

**Adjustments:**
- Use low flow rate (20–50 μL/min) if possible
- Include control: isotonic buffer without drug to measure baseline mechanical response
- Use gentle infusion protocols (syringe pump, not peristaltic pump if possible)
- Allow 5–10 min stabilization after flow starts before assessing drug response
