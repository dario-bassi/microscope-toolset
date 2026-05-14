# Pattern: MDA-Based Acquisition

> **TL;DR** — Use MDA for all multi-frame acquisitions; `snap()` only for single exploratory frames.
> Simplest timelapse: `timelapse(core, n_frames, interval_s, channel)`.
> Device changes between phases: separate `MDASequence` calls with state changes between them.
> SLM per-frame control: generator that sets the mask between yields.
> **Never** embed `SLMImage` inside `MDAEvent` when using pymmcore-proxy — serialisation fails.

> **Import note:** Install with `pip install -e .` or `uv sync`, then import as `self_learn.*`

## Why Use MDA
- MDA produces consistent, reproducible acquisition sequences
- MDA works on real microscopes — snap loops require manual timing
- MDA handles multi-position and multi-channel natively
- pymmcore-plus MDASequence is the standard

## Decision Flowchart: MDA vs snap()

```
Is it a single image?  → snap()
Is it a pure timelapse (no device changes)?  → MDA (timelapse() or MDASequence)
Is it multi-position static imaging?  → MDA (stage_positions=[...])
Is it multi-channel at one position?  → MDA (channels=[...])
Is it a Z-stack?  → MDA (z_plan={...})
Does it need device state changes MID-experiment?
  → Use SEPARATE MDASequence per phase (see Multi-Phase Pattern below)
  → Device change happens BETWEEN phases, not during
Is it closed-loop SLM/device with per-frame changes?
  → MDA generator (device control between yields — see Generator Pattern)
  → Only use snap() if you truly need the image before deciding the next step
```

**Key rule**: Use MDA for ALL multi-frame acquisitions. Generators handle
device changes between frames (SLM, Perfusion, Temperature). Only fall back
to snap() when you need per-frame decision-making that can't be pre-planned.

**Common mistake**: Using snap() loops for pure timelapses. If you're writing a `for t in range(N): snap()` with fixed timing, that's an MDA timelapse; use `timelapse(core, N, interval, channel)` instead.

## Quick Timelapse (simplest — use this first)

```python
from self_learn.hardware.core import timelapse

# Returns 3D stack (T, H, W) directly — no boilerplate
stack = timelapse(core, n_frames=15, interval_s=1.0, channel='BF')
```

## MDA Timelapse (when you need more control)

```python
from useq import MDASequence
from self_learn.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 15, "interval": 1.0},
    channels=[{"config": "BF", "exposure": 50}],
)
results = run_events(core, list(seq))
```

## Quick MDA with on_frame Callback

```python
from useq import MDASequence
from self_learn.hardware.core import run_events

frames = []
def on_frame(img, event):
    frames.append(img.copy())

seq = MDASequence(
    time_plan={"loops": 20, "interval": 0.15},
    channels=[{"config": "brightfield"}],
)
run_events(core, list(seq), on_frame=on_frame)
```

## Adaptive Acquisition Template

```python
from useq import MDAEvent
from self_learn.hardware.core import run_events

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
            channel={"config": "brightfield"},
            min_start_time=t * 2.0,
        )

run_events(core, gen(), on_frame=on_frame)
```

## Speed Measurement Pattern

```python
from self_learn.detection.cells import count_bacteria_bf
from self_learn.analysis.tracking import population_speeds

centroids_per_frame, timestamps = [], []
def on_frame(img, event):
    _, cents = count_bacteria_bf(img, return_centroids=True, magnification=40)
    centroids_per_frame.append(cents)
    timestamps.append(time.time())

# ... run MDA with on_frame ...

result = population_speeds(centroids_per_frame, timestamps,
                           pixel_size=0.25, max_displacement=100)
```

## Multi-Phase MDA with Device Changes (PREFERRED)

Use separate MDASequence per phase with device changes between them.
This is the correct pattern for Perfusion, Temperature, SLM mode changes.

```python
from useq import MDASequence
from self_learn.hardware.core import run_events

all_frames = []
def on_frame(img, event):
    all_frames.append(img.copy())

channel = {"config": ch_nucleus}  # name comes from reading getAvailableConfigs() + picking the right one by looking at the images — no fuzzy-matcher

# Phase 1: Baseline
seq1 = MDASequence(time_plan={"loops": 10, "interval": 1.0}, channels=[channel])
run_events(core, list(seq1), on_frame=on_frame)

# Device change (between MDA phases)
core.setState(perfusion_device, drug_state)  # switch to drug

# Phase 2: Treatment
seq2 = MDASequence(time_plan={"loops": 25, "interval": 2.0}, channels=[channel])
run_events(core, list(seq2), on_frame=on_frame)

# Device change
core.setState(perfusion_device, washout_state)  # switch to washout

# Phase 3: Recovery
seq3 = MDASequence(time_plan={"loops": 15, "interval": 2.0}, channels=[channel])
run_events(core, list(seq3), on_frame=on_frame)
```

Works for: Perfusion drug/washout, Temperature shifts, SLM mode changes.
Each phase gets proper MDA timing. Device changes happen between phases.

## Multi-Phase Experiment Template (workflow module)

```python
from self_learn.workflows.experiment import phase_timelapse, baseline_treatment
from self_learn.hardware.core import run_events

# Define phases
phases = baseline_treatment(
    n_baseline=10, n_treatment=15, interval=1.0,
    treatment_devices={'Temperature': '42'},
    wait_after_treatment=5.0)

channels = [{'config': 'brightfield'}]

# Execute
frames = []
def on_frame(img, event):
    frames.append(img.copy())

events = list(phase_timelapse(phases, channels))
run_events(core, events, on_frame=on_frame)
```

## Multi-Position Tiling Template

```python
from self_learn.workflows.batch import tile_and_analyze

def analyze_tile(img):
    return {'cell_count': detect_cells(img), ...}

result = tile_and_analyze(core, analyze_tile, grid=(2,2),
                          channel='brightfield')
# result['aggregate'] has mean/total across tiles
```

## MDA Generator for Device Control Between Frames (PREFERRED)

Use a generator when you need device state changes between frames.
The generator yields MDAEvents and controls devices between yields:

```python
from useq import MDAEvent
from self_learn.hardware.core import run_events

def experiment():
    ch = {"config": ch_nucleus}

    # Phase 1: Baseline (SLM off)
    core.setSLMImage("SLM", clear_mask)
    core.displaySLMImage("SLM")
    for t in range(10):
        yield MDAEvent(channel=ch, exposure=50,
                      metadata={'phase': 'baseline', 't': t})

    # Phase 2: SLM stimulation
    core.setSLMImage("SLM", stim_mask)
    core.displaySLMImage("SLM")
    for t in range(5):
        yield MDAEvent(channel=ch, exposure=50,
                      metadata={'phase': 'stim', 't': t})

    # Phase 3: Recovery (SLM off)
    core.setSLMImage("SLM", clear_mask)
    core.displaySLMImage("SLM")
    for t in range(10):
        yield MDAEvent(channel=ch, exposure=50,
                      metadata={'phase': 'recovery', 't': t})

run_events(core, experiment(), on_frame=on_frame)
```

Key: `run_events()` processes generators lazily — side effects between
yields are preserved. This works for SLM, Perfusion, Temperature, etc.

## When snap() Loops ARE Still Appropriate

Only use raw snap() loops for:
- Single exploratory snaps (overview, focus check)
- Per-frame closed-loop feedback where you need the image BEFORE deciding
  what to do next (and can't express this as a generator)

## Practical Notes
- MDA events iterate LOCALLY — only the actual snap/move calls go to the server
- Primary execution: `run_events(core, events, on_frame=callback)` from `hardware.core`
- **Never use SLMImage in MDA events** — it can't be serialized over pymmcore-proxy.
  Instead, control SLM manually between generator yields (see example above).
- Channel must be passed as dict in MDAEvent: `channel={"config": name}`
- Check available config groups with `core.getAvailableConfigGroups()` first
- Generators are processed lazily by run_events() — side effects between yields preserved
