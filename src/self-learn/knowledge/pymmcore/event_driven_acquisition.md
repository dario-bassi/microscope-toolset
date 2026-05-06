# Event-Driven Acquisition — Poll/Burst State Machine

For monitoring experiments that react to transient events: poll at low rate,
burst at high rate when something happens.

## State Machine

```
IDLE → POLLING → EVENT_DETECTED → BURST → COOLDOWN → POLLING
```

## Core API

```python
from workflows.event_driven import make_event_state, run_event_loop

state = make_event_state(
    poll_interval=2.0,       # seconds between polls
    burst_interval=0.1,      # seconds between burst frames
    burst_duration=5.0,      # seconds of burst acquisition
    cooldown_duration=3.0,   # seconds before re-arming
    trigger_up=500,          # threshold to START burst
    trigger_down=300,        # threshold to STOP burst (hysteresis)
)
```

## Hysteresis — Preventing Mode Flicker

```python
# BAD: single threshold → rapid toggling around noisy signal
if signal > 500: burst()

# GOOD: separate up/down thresholds with gap
# Enter burst when signal > trigger_up (500)
# Exit burst when signal < trigger_down (300)
```

## Running the Loop

```python
def detect_event(image):
    return image.mean()  # metric compared against trigger thresholds

def on_burst_frame(image, event, state):
    state.measurements.append(image.copy())

run_event_loop(core, channel="GFP", detect_fn=detect_event,
               burst_callback=on_burst_frame, state=state, max_duration=120.0)
```

## Manual Generator Pattern

```python
from useq import MDAEvent

def event_driven_generator(core):
    mode, cooldown = "POLLING", 0
    while True:
        yield MDAEvent(channel={"config": "GFP"}, exposure=20)
        img = get_last_image()
        if mode == "POLLING" and img.mean() > 500:
            mode = "BURST"
            for _ in range(50):
                yield MDAEvent(channel={"config": "GFP"}, exposure=10)
                if get_last_image().mean() < 300: break  # hysteresis
            mode, cooldown = "COOLDOWN", 5
        elif mode == "COOLDOWN":
            cooldown -= 1
            if cooldown <= 0: mode = "POLLING"
```

## Example: Calcium Wave Monitoring

```python
state = make_event_state(
    poll_interval=2.0, burst_interval=0.1, burst_duration=10.0,
    cooldown_duration=5.0, trigger_up=200, trigger_down=120,
)
run_event_loop(core, channel="GFP", detect_fn=detect_calcium, state=state)
```

## Key Points

- Poll/burst avoids photobleaching and data overload vs continuous acquisition
- Hysteresis (separate up/down thresholds) prevents rapid mode switching
- Cooldown prevents re-triggering on the decay phase of the same event
- Combine with SLM for closed-loop optogenetics (detect wave, apply barrier)
