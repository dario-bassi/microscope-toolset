# Playbook: Timelapse & Division Detection

## When to Use
- Time-series imaging to track cell dynamics
- Tasks: detect divisions, track cell count over time, monitor morphology changes

## Step 1: Understand the Timing

- Check how time advances: per-snap, per-interval, or continuous (real-time)
- Know the expected timescale of the biology (cell cycle ~24h, drug onset ~minutes)
- Plan frame count and interval to capture the dynamics of interest
- **Multi-channel cost**: Each channel snap consumes time. Plan accordingly.

## Step 2: Acquire Timelapse

### Basic MDASequence (fixed timing)
```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 20, "interval": 2.0},
    channels=[{"config": "brightfield", "group": "Fake"}],
)
results = run_events(core, list(seq))
```

### Drift-corrected timelapse (Z-drift)
```python
from src.workflows.autofocus import drift_corrected_timelapse
from src.hardware.core import run_events

gen, on_frame, state = drift_corrected_timelapse(
    n_frames=15, interval=1.0, initial_drift_rate=0.4,
    channels=['membrane-channel'], focus_method='brenner')
results = run_events(core, gen(), on_frame=on_frame)
# state['focus_scores'] tracks focus quality per frame
```

### Adaptive timelapse (with per-frame analysis)
```python
from useq import MDAEvent
from src.hardware.core import run_events
from src.detection.cells import detect_cells

shared = {"counts": []}

def on_frame(img, event):
    cells = detect_cells(img)
    shared["counts"].append(len(cells))

def gen():
    for t in range(30):
        yield MDAEvent(
            channel={"config": "brightfield", "group": "Fake"},
            min_start_time=t * 2.0,  # 2s intervals
        )

results = run_events(core, gen(), on_frame=on_frame)
```

## Step 3: Track Cell Count

Count cells per frame with CONSISTENT threshold:
```python
import math
counts = []  # [n_t0, n_t1, ..., n_tN]
for img, ev in results:
    cells = detect_cells(img, threshold_sigma=2.0)
    counts.append(len(cells))

# Doubling time from linear regression on log2(count) vs time
import numpy as np
t_arr = np.arange(len(counts), dtype=float)
log2_counts = [math.log2(c) for c in counts if c > 0]
slope, intercept = np.polyfit(t_arr[:len(log2_counts)], log2_counts, 1)
doubling_time = 1.0 / slope  # in frame units
```

## Step 4: Visual Verification

Save key frames and examine with vision:
```python
from PIL import Image
for i in [0, len(results)//2, -1]:
    img, _ = results[i]
    Image.fromarray(img).save(f"/tmp/timelapse_frame_{i}.png")
# Read with Read tool to visually verify
```

## Common Pitfalls

- **CONSISTENT thresholds**: Use the same threshold for all frames. Adaptive thresholds
  mask true signal changes.
- **Frame budget**: Don't waste frames on preview snaps during timelapse.
- **Drift between frames**: Use drift_corrected_timelapse for Z-drift, or
  phase_correlate/fft_cross_correlate from src.hardware.drift for XY drift.
- **Multi-channel cost**: Each channel snap advances time. Use BF-only for
  counting, add fluorescence only when needed.
- **Non-monotonic counts**: If re-counting from scratch each frame gives
  oscillating counts (24→60→55), use tracking (Hungarian matching) to
  maintain consistent identity across frames.
- **min_start_time**: run_events() respects MDAEvent.min_start_time for
  proper timing intervals in real-time experiments.
