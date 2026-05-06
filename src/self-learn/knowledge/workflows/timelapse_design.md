# Timelapse Design

## Backward Design

Start with WHAT to measure, derive HOW to acquire:

```
biological process -> temporal resolution -> frame rate -> total duration -> exposure budget
```

## Step 1: Identify the Temporal Process

| Process             | Timescale        | Typical Frame Rate  |
|---------------------|------------------|---------------------|
| Calcium transients  | 100ms - 2s       | 5-20 fps            |
| Cell migration      | 5-30 min/frame   | 1 frame/5-10 min    |
| Cell division       | 30 min - 2 hours | 1 frame/5-15 min    |
| Wound healing       | hours            | 1 frame/15-30 min   |
| Phototaxis          | seconds/step     | 1 frame/step        |

## Step 2: Frame Rate (Nyquist)

Frame interval <= fastest_change_period / 2. Use 3-5x Nyquist if budget allows.

## Step 3: Total Duration

`total_duration = expected_process_time * 1.3` (add 30% buffer for variance).

## Step 4: Exposure Budget (Phototoxicity)

```
total_dose = n_frames * exposure * n_channels * intensity
```

Reduce dose: minimum exposure for sufficient SNR, fewer channels, longer intervals.

```python
n_frames = int(total_duration / frame_interval)
exposure_budget = max_total_dose / (n_frames * n_channels)
```

## Step 5: Drift Correction

**Focus metric** (Laplacian variance on nucleus channel):
```python
def focus_score(img):
    return laplace(img.astype(float)).var()
```

**Drift rate**: Observe first 5-7 frames, fit linear model:
```python
drift_rate = np.polyfit(time_points, z_positions, 1)[0]  # um/frame
```

**Proactive correction** via adaptive generator (Z corrected per-frame):
```python
from useq import MDAEvent
from src.hardware.core import run_events

def drift_corrected_gen(n_frames, z_start, drift_rate, channel, group):
    for i in range(n_frames):
        yield MDAEvent(
            z_pos=z_start + drift_rate * i,
            channel={"config": channel, "group": group},
            index={"t": i},
        )

results = run_events(core, drift_corrected_gen(n_frames, z_start, drift_rate, "BF", group))
```

## Complete Example

```python
from useq import MDASequence
from src.hardware.core import run_events

# Track cell division over 2 hours, 2 channels, drift-corrected Z
n_frames = 16  # 120min / 10min interval
seq = MDASequence(
    time_plan={"loops": n_frames, "interval": 600.0},  # 10 min
    channels=[{"config": "BF", "group": group}, {"config": "nucleus", "group": group}],
    axis_order="tpc",
)
results = run_events(core, list(seq))
```

## Common Pitfalls

- **Undersampling**: If in doubt, sample faster first, then reduce.
- **Phototoxicity**: Dead cells do not divide. Monitor cell health.
- **Ignoring drift**: Always correct for timelapses > 20 minutes.
- **No baseline**: Acquire frames BEFORE stimulus to establish reference.
