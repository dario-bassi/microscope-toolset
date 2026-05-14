# Generator-Based Adaptive Acquisition

For anything MDASequence cannot express — conditional logic, adaptive decisions,
closed-loop control — use Python generators that yield MDAEvent objects.

## Basic Pattern

```python
from useq import MDAEvent

def my_experiment():
    yield MDAEvent(channel={"config": "BF"}, exposure=20)
    yield MDAEvent(x_pos=100, y_pos=200, channel={"config": "GFP"}, exposure=50)
    if some_condition:
        yield MDAEvent(channel={"config": "DAPI"}, exposure=30)

core.mda.run(my_experiment())
```

## Embedding MDASequence in a Generator

```python
from useq import MDASequence, MDAEvent

def survey_then_zoom():
    # Phase 1: survey scan using a standard sequence
    survey = MDASequence(
        stage_positions=[{"x": i * 500, "y": 0} for i in range(5)],
        channels=[{"config": "BF", "exposure": 10}],
    )
    yield from survey  # embed standard sequence

    # Phase 2: analyze and yield targeted events
    for pos in find_interesting_positions():
        yield MDAEvent(x_pos=pos["x"], y_pos=pos["y"],
                       channel={"config": "GFP"}, exposure=100)
```

## OADA — On-line Adaptive Decision Architecture

Use a plain generator with a shared-state dict and a `frameReady` callback.
The engine delivers each frame (updating shared state) before requesting the
next event from the generator.

```python
from useq import MDAEvent

shared = {"last_intensity": 0.0, "measurements": []}

def on_frame(image, event):
    shared["last_intensity"] = float(image.mean())
    shared["measurements"].append(shared["last_intensity"])

def adaptive_gen():
    for step in range(50):
        yield MDAEvent(channel={"config": "BF"}, min_start_time=0.5)
        if shared["last_intensity"] > 1000:
            return  # early stop

core.mda.events.frameReady.connect(on_frame)
try:
    core.mda.run(adaptive_gen())
finally:
    core.mda.events.frameReady.disconnect(on_frame)
```

## Common Adaptive Patterns

1. **Survey-and-zoom**: scan FOVs at low mag, revisit interesting ones at high mag
2. **Focus-hunt**: yield Z events, find best focus, continue at that Z
3. **Dose-response**: adjust SLM/illumination based on measured fluorescence
4. **Early termination**: stop when signal reaches threshold

## Key Rules

- Generators are lazy — events produced one at a time
- Engine processes each yielded event before requesting the next
- `yield from` flattens a sub-sequence into the parent generator
- State between yields lives in normal Python variables
- Connect to `frameReady` to capture images for decision logic

## See also

- [[Core/Concepts/MDA standard]] — the non-adaptive counterpart.
- [[Core/Concepts/Event-driven acquisition]] — poll/burst as a generator pattern.
- [[Core/Strategies/Adaptive acquisition]] — decide where to image next from the frame you just saw.
- [[Core/Strategies/Feedback control]] — closed-loop interventions.
- `self_learn.hardware.core: run_events` — project wrapper around `core.mda.run()` with `on_frame` kwarg.

## References

Topic candidates for verification in [[Papers candidates]] — automated smart-microscopy protocols, AI-guided adaptive acquisition.
