# The OADA Loop

**Observe -- Analyze -- Decide -- Act**

The core adaptive acquisition cycle. Every closed-loop microscopy experiment
is an instance of this pattern. Master this and you can handle any challenge
that requires real-time feedback.

---

## The Basic Loop

```
+----------+     +---------+     +--------+     +------+
| OBSERVE  | --> | ANALYZE | --> | DECIDE | --> | ACT  |
| (snap)   |     | (process|     | (logic)|     | (hw) |
+----------+     +---------+     +--------+     +------+
      ^                                             |
      |                                             |
      +---------------------------------------------+
```

1. **Observe**: Acquire one or more images (snap, MDA event)
2. **Analyze**: Process image(s) to extract measurements
3. **Decide**: Apply decision logic to measurements
4. **Act**: Modify hardware state (stage, SLM, exposure, objective, channel)

Then repeat.

---

## Code Pattern

The canonical implementation using an adaptive generator with `run_events()`:

```python
from useq import MDAEvent
from self_learn.hardware.core import run_events

def adaptive_experiment(core, group, max_steps=30):
    """Generic OADA loop skeleton — MDA generator pattern."""
    results = []
    state = {"done": False}

    def on_frame(img, event):
        # --- ANALYZE ---
        measurement = analyze_frame(img)
        results.append(measurement)

        # --- DECIDE ---
        action = decide_next_action(measurement, results)
        if action == "done":
            state["done"] = True
        else:
            # --- ACT --- (hardware changes take effect before next event)
            execute_action(core, action)

    def gen():
        for step in range(max_steps):
            if state["done"]:
                return
            yield MDAEvent(channel={"config": "BF", "group": group})

    run_events(core, gen(), on_frame=on_frame)
    return results
```

For SLM-based closed-loop control, attach the mask directly to the
event via `MDAEvent.slm_image`. `run_events()` applies it before each
snap — no manual `setSLMImage()` / `displaySLMImage()` calls needed.

```python
from useq import MDAEvent, SLMImage
from self_learn.hardware.core import run_events

def slm_oada(core, target_pos, max_steps=30, threshold=5):
    """Closed-loop SLM steering using the same generator pattern as above."""
    state = {"done": False, "current_mask": create_slm_mask(target_pos)}

    def on_frame(img, event):
        current_pos = detect_object(img)
        distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        if distance < threshold:
            state["done"] = True
        else:
            # update mask for the next event
            state["current_mask"] = create_slm_mask(current_pos)

    def gen():
        for _ in range(max_steps):
            if state["done"]:
                return
            slm = SLMImage(data=state["current_mask"].astype(np.uint8), device="SLM")
            yield MDAEvent(channel={"config": "BF"}, slm_image=slm)

    run_events(core, gen(), on_frame=on_frame)
```

**Critical**: `slm_image` on `MDAEvent` is applied automatically before each snap by `run_events()`.
No need to call `setSLMImage()` manually.

---

## Hierarchy of Feedback

OADA operates at multiple timescales:

### Per-Frame Feedback (fastest)
- Adjust exposure based on image brightness
- Update SLM mask based on object position
- Refocus if sharpness drops below threshold
- Skip channels if signal is absent

### Per-Tile Feedback
- Skip empty tiles during grid scans
- Increase resolution in regions with features
- Change objective for detailed sub-regions

### Per-Phase Feedback
- Switch from survey (10x) to detail (40x) after finding ROIs
- Change from time-lapse to event-triggered after detecting onset
- Stop experiment when convergence criteria are met

### Per-Experiment Feedback
- Adjust parameters for next well based on previous well
- Update detection thresholds based on control wells
- Modify acquisition plan based on sample condition

---

## Common OADA Patterns

### 1. Adaptive Exposure

```python
# OBSERVE
core.snapImage()
img = core.getImage()

# ANALYZE
brightness = np.percentile(img, 95)

# DECIDE
if brightness < 50:
    new_exposure = current_exposure * 2
elif brightness > 200:
    new_exposure = current_exposure / 2
else:
    new_exposure = current_exposure  # good enough

# ACT
core.setExposure(new_exposure)
```

### 2. Smart Grid Scanning

```python
# Survey at low mag
positions = generate_grid(core, overlap=0.1)
interesting = []

for pos in positions:
    core.setXYPosition(pos[0], pos[1])
    core.snapImage()
    img = core.getImage()

    # ANALYZE: is there anything here?
    if np.std(img) > empty_threshold:
        interesting.append(pos)

# Switch to high mag only for interesting positions
set_objective(core, 40)
for pos in interesting:
    core.setXYPosition(pos[0], pos[1])
    # detailed acquisition...
```

### 3. Event-Triggered Acquisition

```python
# Low frame rate until event detected
while not event_detected:
    core.snapImage()
    img = core.getImage()
    event_detected = check_for_event(img)
    time.sleep(long_interval)

# Switch to high frame rate
for frame in range(burst_frames):
    core.snapImage()
    img = core.getImage()
    record(img)
    time.sleep(short_interval)
```

### 4. Convergence-Based Stopping

```python
measurements = []
for step in range(max_steps):
    core.snapImage()
    val = measure(core.getImage())
    measurements.append(val)

    # Stop when measurement stabilizes
    if len(measurements) >= 5:
        recent = measurements[-5:]
        if np.std(recent) / np.mean(recent) < 0.02:  # CV < 2%
            break
```

---

## State Management

The OADA loop must track state across iterations:

```python
class OADAState:
    def __init__(self):
        self.frame_count = 0
        self.measurements = []
        self.actions_taken = []
        self.current_phase = "survey"

    def update(self, measurement, action):
        self.frame_count += 1
        self.measurements.append(measurement)
        self.actions_taken.append(action)
```

Key state to track:
- Frame counter (budget management)
- Running measurements (for trend detection)
- Current phase (survey vs detail vs verification)
- Hardware state (current objective, channel, position)

---

## Pitfalls

- **Forgetting to copy images**: `core.getImage()` returns a buffer that gets
  overwritten on next snap. Always `.copy()`.
- **SLM not applied before snap**: The mask must be set before each snap call.
- **Budget snaps wisely**: Avoid unnecessary acquisitions for "previewing" —
  every snap has a cost (time, phototoxicity, bleaching).
- **Infinite loops**: Always have a `max_steps` guard. Budget your snaps.
- **State not reset between phases**: Clear accumulators when switching modes.
