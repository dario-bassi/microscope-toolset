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

```python
from workflows.oada import adaptive_generator, FeedbackState

def on_decide(image, event, state: FeedbackState):
    intensity = image.mean()
    if intensity > 1000:
        state.stop = True                     # stop acquisition
    state.extra_events.append(                # inject events dynamically
        MDAEvent(channel={"config": "GFP"}, exposure=200))
    state.next_exposure = 100 if intensity < 500 else 20  # adjust exposure
    state.measurements.append({"mean": float(intensity)})

base_events = MDASequence(time_plan={"loops": 50, "interval": 0.5})
gen = adaptive_generator(base_events, on_decide=on_decide)
core.mda.run(gen)
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
