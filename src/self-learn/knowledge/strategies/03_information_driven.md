# Information-Driven Acquisition

> **Note**: Code examples below use conceptual pseudocode. All multi-frame
> loops MUST use `run_events(core, generator(), on_frame=callback)` in practice.

Acquire only the data that changes your conclusions. Every photon should
contribute information; every snap should reduce uncertainty.

---

## Core Philosophy

Traditional microscopy: "Acquire everything, analyze later."
Information-driven: "Analyze as you go, acquire only what's needed."

The agent has a finite photon budget (phototoxicity), time budget (snaps per
challenge), and computational budget. Spend them where they matter most.

---

## DIA to DDA Targeting

Borrowed from mass spectrometry: Data-Independent Acquisition (survey) followed
by Data-Dependent Acquisition (targeted).

### Survey Phase (DIA)
- Low magnification (10x), brightfield
- Quick scan of available positions / wells
- Goal: find regions of interest, estimate sample density
- Cheap: BF is non-destructive, 10x covers large FOV

### Targeting Phase (DDA)
- High magnification (40x) on selected ROIs
- Specific fluorescence channels as needed
- Goal: precise measurements on features identified in survey
- Expensive: fluorescence bleaches, 40x is slow, small FOV

### Example: Tissue Triage

```python
# Phase 1: Survey all wells at 10x BF
well_data = {}
for well in range(num_wells):
    move_to_well(core, well)
    core.setConfig(group, "BF")
    core.snapImage()
    img_bf = core.getImage().copy()
    well_data[well] = {"bf_std": np.std(img_bf), "bf_mean": np.mean(img_bf)}

# Phase 2: Classify wells based on survey
# (wound = low std region, drug = check fluorescence, overgrowth = high density)

# Phase 3: Targeted fluorescence only where needed
for well in wells_needing_fluorescence:
    move_to_well(core, well)
    core.setConfig(group, "nucleus")
    core.snapImage()
    img_nuc = core.getImage().copy()
    # detailed analysis...
```

---

## Hysteresis Triggering

When transitioning between acquisition modes, use hysteresis to prevent
flickering.

### The Problem
Without hysteresis, noise near a threshold causes rapid mode switching:
```
Frame 1: metric=49 -> low-res mode
Frame 2: metric=51 -> high-res mode    # switch!
Frame 3: metric=49 -> low-res mode     # switch back!
Frame 4: metric=50 -> ???              # unstable
```

### The Solution
Use separate thresholds for activating and deactivating:

```python
HIGH_THRESHOLD = 55   # switch to high-res above this
LOW_THRESHOLD = 45    # switch to low-res below this
current_mode = "low_res"

for frame in acquisition:
    metric = compute_metric(frame)

    if current_mode == "low_res" and metric > HIGH_THRESHOLD:
        current_mode = "high_res"
        switch_to_high_res(core)
    elif current_mode == "high_res" and metric < LOW_THRESHOLD:
        current_mode = "low_res"
        switch_to_low_res(core)
    # Between 45 and 55: stay in current mode
```

Apply this to:
- Autofocus triggering (focus metric drop)
- Channel switching (signal level)
- Objective changes (feature density)
- SLM activation (distance to target)

---

## Photon Budgeting

Every photon costs something: bleaching, phototoxicity, time. Budget them.

### Cost Hierarchy (cheapest to most expensive)

1. **Brightfield** -- essentially free, no bleaching
2. **Phase contrast / DIC** -- also label-free, minimal damage
3. **Low-power fluorescence** -- some bleaching, acceptable for survey
4. **High-power fluorescence** -- significant bleaching, use sparingly
5. **UV excitation** -- most damaging, avoid unless essential

### Budget Allocation Strategy

```
Total budget: N snaps

Allocation:
  Survey (BF):           30% of budget  -- find sample, plan
  Primary measurement:   50% of budget  -- main experiment
  Verification:          10% of budget  -- sanity checks
  Reserve:               10% of budget  -- error recovery
```

### Example: 30-Snap Budget

```python
# Survey: 3 snaps (BF at 10x)
# Switch objective: 0 snaps (free)
# Main experiment: 15 snaps (fluorescence time-lapse at 40x)
# Verification: 3 snaps (overlay checks)
# Reserve: 9 snaps (retry if detection fails)
```

---

## Stopping Criteria

Know when you have enough data. Do not keep acquiring out of habit.

### Statistical Sufficiency
- Cell count: stop when coefficient of variation < 5% across frames
- Intensity ratio: stop when confidence interval is narrow enough
- Speed measurement: stop when running average stabilizes

### Convergence Detection

```python
def has_converged(values, window=5, cv_threshold=0.02):
    """Check if recent measurements have stabilized."""
    if len(values) < window:
        return False
    recent = values[-window:]
    mean = np.mean(recent)
    if mean == 0:
        return True
    cv = np.std(recent) / abs(mean)
    return cv < cv_threshold
```

### Event Completion
- Tracking: stop when object leaves FOV or stops moving
- Wound healing: stop when gap is closed
- Phototaxis: stop when colony reaches target
- Photoconversion: stop when converted cells are tracked long enough

### Budget Exhaustion
- Always have a hard maximum frame count
- Warn when 80% of budget is used
- Reserve last 10% for verification and recovery

---

## Information Gain Estimation

Before each snap, estimate the expected information gain:

### High Information Gain (snap is valuable)
- First frame of a new channel (reveals new features)
- Frame after a perturbation (SLM, drug, media change)
- Frame at a new position (unexplored region)
- Frame during rapid dynamics (fast-changing system)

### Low Information Gain (skip if budget is tight)
- Repeat of a stable measurement (system at steady state)
- Redundant channel (BF and phase contrast of same field)
- Dense time sampling of slow process (cell division at 1s interval)
- Empty region (already confirmed no cells here)

### Decision Framework

```python
def should_snap(state):
    """Estimate whether next snap is worth the cost."""
    # Always snap if we have no data
    if state.frame_count == 0:
        return True

    # Always snap after perturbation
    if state.just_applied_slm or state.just_moved_stage:
        return True

    # Skip if measurements have converged
    if has_converged(state.measurements):
        return False

    # Skip if budget is low and gain is marginal
    budget_remaining = state.max_frames - state.frame_count
    if budget_remaining < 5 and not state.in_critical_phase:
        return False

    return True
```

---

## Anti-Patterns

- **Acquiring all channels every frame**: Snap only channels needed for
  the current decision. BF for tracking, fluorescence for quantification.
- **Fixed frame rate for variable dynamics**: Slow down when nothing happens,
  speed up during events.
- **Completing a grid scan of empty wells**: Check each well before committing
  to full acquisition. Skip empty ones.
- **Fluorescence survey**: Use BF for survey. Fluorescence is for measurement.
- **Not reserving budget for recovery**: If detection fails at frame 28 of 30,
  you have no room to retry.
