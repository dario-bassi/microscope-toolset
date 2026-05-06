# Pattern: MDA-Based Acquisition

## Why Use MDA
- MDA produces consistent, reproducible acquisition sequences
- MDA works on real microscopes — snap loops require manual timing
- MDA handles multi-position and multi-channel natively
- pymmcore-plus MDASequence is the standard

## When to Use MDA vs snap()
- **Use MDA**: timelapse, multi-position, multi-channel, grid scans, Z-stacks
- **Use snap()**: single-image captures, quick preview, SLM closed-loop (need per-frame control)

## Quick MDA Timelapse Template

```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 15, "interval": 1.0},
    channels=[{"config": "BF", "exposure": 50}],
)
results = run_events(core, list(seq))
```

## Quick MDA with on_frame Callback

```python
from useq import MDASequence
from src.hardware.core import run_events

frames = []
def on_frame(img, event):
    frames.append(img.copy())

seq = MDASequence(
    time_plan={"loops": 20, "interval": 0.15},
    channels=[{"config": "brightfield", "group": "Fake"}],
)
run_events(core, list(seq), on_frame=on_frame)
```

## Adaptive Acquisition Template

```python
from useq import MDAEvent
from src.hardware.core import run_events

shared = {"counts": [], "stop": False}

def on_frame(img, event):
    cells = detect(img)
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

run_events(core, gen(), on_frame=on_frame)
```

## Speed Measurement Pattern (ch418)

```python
from src.detection.cells import count_bacteria_bf
from src.analysis.tracking import population_speeds

centroids_per_frame, timestamps = [], []
def on_frame(img, event):
    _, cents = count_bacteria_bf(img, return_centroids=True, magnification=40)
    centroids_per_frame.append(cents)
    timestamps.append(time.time())

# ... run MDA with on_frame ...

result = population_speeds(centroids_per_frame, timestamps,
                           pixel_size=0.25, max_displacement=100)
```

## Multi-Phase Experiment Template

```python
from src.workflows.experiment import phase_timelapse, baseline_treatment
from src.hardware.core import run_events

# Define phases
phases = baseline_treatment(
    n_baseline=10, n_treatment=15, interval=1.0,
    treatment_devices={'Temperature': '42'},
    wait_after_treatment=5.0)

channels = [{'config': 'brightfield', 'group': 'Fake'}]

# Execute
frames = []
def on_frame(img, event):
    frames.append(img.copy())

events = list(phase_timelapse(phases, channels))
run_events(core, events, on_frame=on_frame)
```

## Multi-Position Tiling Template

```python
from src.workflows.batch import tile_and_analyze

def analyze_tile(img):
    return {'cell_count': detect_cells(img), ...}

result = tile_and_analyze(core, analyze_tile, grid=(2,2),
                          channel='brightfield', group='Fake')
# result['aggregate'] has mean/total across tiles
```

## Practical Notes
- MDA events iterate LOCALLY — only the actual snap/move calls go to the server
- Primary execution: `run_events(core, events, on_frame=callback)` from `src.hardware.core`
- For SLM control: use MDAEvent with `slm_image=SLMImage(data=mask, device="SLM")`
- Channel must be passed as dict in MDAEvent: `channel={"config": name}`
- Check available config groups with `core.getAvailableConfigGroups()` first
- For realtime challenges with `--time-scale N`: biological_speed = displacement * pixel_size / (dt_wall * N)
