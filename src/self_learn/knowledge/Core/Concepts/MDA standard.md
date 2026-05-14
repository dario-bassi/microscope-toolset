# MDASequence — Standard Fixed Acquisitions

MDASequence defines pre-planned, non-adaptive acquisition sequences.
All plans are declarative and can be freely combined.

```python
from useq import MDASequence, MDAEvent
```

## Plan Types

```python
# Timelapse
seq = MDASequence(time_plan={"loops": 10, "interval": 1.0})

# Z-stack
seq = MDASequence(z_plan={"range": 10, "step": 0.5})  # 21 slices

# Multi-position
seq = MDASequence(stage_positions=[
    {"x": 100, "y": 200},
    {"x": 300, "y": 400, "z": 10.0},  # optional per-position Z
])

# Multi-channel
seq = MDASequence(channels=[
    {"config": "DAPI", "exposure": 20},
    {"config": "GFP", "exposure": 50},
])
```

## Combined Acquisition

```python
seq = MDASequence(
    time_plan={"loops": 5, "interval": 2.0},
    z_plan={"range": 6, "step": 1.0},
    stage_positions=[{"x": 0, "y": 0}, {"x": 100, "y": 100}],
    channels=[{"config": "DAPI", "exposure": 20}, {"config": "GFP", "exposure": 50}],
)
# Default axis order (outer→inner): T → P → C → Z
# Customize with: axis_order="tpzc"
```

## Running and Listening

```python
core.mda.run(seq)  # blocks until complete

# Per-frame callback
def on_frame(image, event):
    print(f"t={event.index.get('t')}, ch={event.channel.config}")

core.mda.events.frameReady.connect(on_frame)
core.mda.run(seq)
core.mda.events.frameReady.disconnect(on_frame)
```

## When to Use MDASequence vs Generators

- **MDASequence**: fixed plans known before acquisition starts
- **Generators**: adaptive decisions, conditional logic, closed-loop control
- Embed in a generator with `yield from seq`

## See also

- [[Core/Concepts/MDA generators]] — when the plan isn't fixed.
- [[Core/Concepts/MDA engine]] — custom hardware actions inside MDA events.
- [[Core/Concepts/Event-driven acquisition]] — poll/burst state machines as a special case of generators.
- [[Core/Approach/MDA solve pattern]] — canonical `MDASequence + run_events` usage.
- `self_learn.hardware.core: run_events` — project wrapper; delegates to `core.mda.run()`.

## Literature

- [[Papers/Pinkard 2021]] — the Python-native programmable-acquisition substrate (acquisition events + acquisition hooks + image processors) that `useq.MDAEvent` + `MDAEngine` overrides + `on_frame` callbacks generalise. The MDA standard documented here is the descendant of Pycro-Manager's three abstractions.
- [[Papers/Edelstein 2010]] — the C++ µManager Core (MMCore) is the device-abstraction layer underneath everything in this note. `useq.MDAEvent` describes *what* to acquire; the µManager Core knows *how* to drive arbitrary motorised microscopes to do it. Every `core.setConfig` / `setState` / `setXYPosition` / `snapImage` call inside an MDA run crosses the µManager device-API boundary.

## References

- useq-schema — https://github.com/pymmcore-plus/useq-schema — the MDASequence spec.
