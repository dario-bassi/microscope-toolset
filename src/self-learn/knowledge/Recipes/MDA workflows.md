# MDA Workflows Playbook

## When to Use MDA vs Snap

### Use MDASequence / run_events() when:
- Acquiring N frames over time (timelapse, tracking, kinetics)
- Multi-channel per timepoint (BF + fluorescence)
- Multi-position visits
- Z-stacks
- Any fixed, reproducible acquisition protocol
- Timed acquisitions (min_start_time on events)

### Use snap() only when:
- Single-frame preview/scouting (1 frame)
- Quick hardware check before experiment

### Use generator + run_events() when:
- Timelapse with mid-experiment decisions (adaptive exposure, early stopping)
- Closed-loop SLM control
- Drift-corrected timelapse

## Key Code Patterns

### MDASequence timelapse
```python
from useq import MDASequence
from src.core.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 15, "interval": 1.0},
    channels=[{"config": "brightfield", "group": "Fake"}],
)
results = run_events(core, list(seq))
# results is list of (image, MDAEvent) tuples
```

### Multi-channel acquisition
```python
seq = MDASequence(
    channels=[
        {"config": "brightfield", "exposure": 50},
        {"config": "nucleus-channel", "exposure": 100},
    ],
)
results = run_events(core, list(seq))
bf_img = results[0][0]
nuc_img = results[1][0]
```

### Multi-position scan
```python
seq = MDASequence(
    stage_positions=[{"x": 100, "y": 200}, {"x": 300, "y": 400}],
    channels=[{"config": "brightfield", "group": "Fake"}],
)
results = run_events(core, list(seq))
```

### Adaptive timelapse with per-frame analysis
```python
from useq import MDAEvent
from src.core.hardware.core import run_events

shared = {"counts": [], "stop": False}

def on_frame(img, event):
    cells = detect_cells(img)
    shared["counts"].append(len(cells))
    if len(cells) > 100:
        shared["stop"] = True

def gen():
    for t in range(50):
        if shared["stop"]:
            return
        yield MDAEvent(
            channel={"config": "brightfield", "group": "Fake"},
            min_start_time=t * 2.0,
        )

results = run_events(core, gen(), on_frame=on_frame)
```

### Drift-corrected timelapse (Z-drift)
```python
from src.core.workflows.autofocus import drift_corrected_timelapse
from src.core.hardware.core import run_events

gen, on_frame, state = drift_corrected_timelapse(
    n_frames=15, interval=1.0, initial_drift_rate=0.4,
    channels=['membrane-channel'], focus_method='brenner')
results = run_events(core, gen(), on_frame=on_frame)
```

### Autofocus (MDA-native)
```python
from src.core.workflows.autofocus import autofocus_mda
from src.core.hardware.core import run_events

gen, on_frame, state = autofocus_mda(z_range=10.0, n_coarse=11, n_fine=11)
results = run_events(core, gen(), on_frame=on_frame)
best_z = state['best_z']
```

### Speed measurement from MDA timelapse (ch418 pattern)
```python
from useq import MDASequence
from src.core.hardware.core import run_events
from src.core.detection.cells import count_bacteria_bf
from src.core.analysis.tracking import population_speeds

# Collect frames
centroids_per_frame, timestamps = [], []
def on_frame(img, event):
    _, cents = count_bacteria_bf(img, return_centroids=True, magnification=40)
    centroids_per_frame.append(cents)
    timestamps.append(time.time())

seq = MDASequence(
    time_plan={"loops": 20, "interval": 0.15},
    channels=[{"config": "brightfield", "group": "Fake"}],
)
run_events(core, list(seq), on_frame=on_frame)

# Compute speeds
result = population_speeds(
    centroids_per_frame, timestamps,
    pixel_size=0.25,         # um/px at 40x
    max_displacement=100,    # px, filter outliers
    time_scale=5.0,          # for realtime sim, 1.0 for real microscopes
)
print(f"Speed: {result['mean_speed']:.1f} ± {result['std_speed']:.1f} um/s")
```

## Execution Pattern

**run_events(core, events, on_frame=)** — The single standard path.
Delegates to `core.mda.run()` via the MDA engine (works identically for
local CMMCorePlus and remote pymmcore-proxy). frameReady signals deliver
frames. Intercepts `CustomAction('switch_objective')` events locally.

## Pitfalls
- Channel config names vary by instrument — check `core.getAvailableConfigs(group)`
- Always discover channel_group from `core.getAvailableConfigGroups()`
- `interval=0` means as-fast-as-possible
- Channel in MDAEvent: `{"config": name, "group": group_name}` (dict format)
- min_start_time is relative to start of acquisition (seconds)
- Multi-channel timelapse: each channel snap is a separate event
