---
title: Event-driven acquisition for content-enriched microscopy
authors: Dora Mahecic, Willi L. Stepp, Chen Zhang, Juliette Griffié, Martin Weigert, Suliana Manley
year: 2022
venue: Nature Methods 19(10):1262-1267
doi: 10.1038/s41592-022-01589-x
url: https://www.nature.com/articles/s41592-022-01589-x
researched: 2026-04-24
---

## Abstract

A common goal of fluorescence microscopy is to collect data on specific biological events. Yet, the event-specific content that can be collected from a sample is limited, especially for rare or stochastic processes. This is due in part to photobleaching and phototoxicity, which constrain imaging speed and duration. We developed an event-driven acquisition framework, in which neural-network-based recognition of specific biological events triggers real-time control in an instant structured illumination microscope. Our setup adapts acquisitions on-the-fly by switching between a slow imaging rate while detecting the onset of events, and a fast imaging rate during their progression. Thus, we capture mitochondrial and bacterial divisions at imaging rates that match their dynamic timescales, while extending overall imaging durations. Because event-driven acquisition allows the microscope to respond specifically to complex biological events, it acquires data enriched in relevant content.

## Smart microscopy principle

Event-driven acquisition is the paradigm where the microscope continuously watches its own frames with a detector (a small neural net, in this paper) and uses the detector's output to modulate acquisition rate in real time. The instrument runs **slow during quiescent baseline** (low dose, long duration) and **fast during the event of interest** (high temporal resolution when it matters). This turns the photon/time budget into a content-matched resource: stochastic or rare events get sampled densely, quiescent intervals cheaply.

The contribution is the closed loop: detection → rate change → continued detection. The microscope is no longer a passive data source; it is a content-aware sampler. The trigger model is swappable (CNN for mitochondrial fission, different CNN for bacterial division, and so on).

## Implementation on pymmcore-plus

The control loop maps cleanly onto a `useq` event generator with an `on_frame` callback that updates a shared state. The generator inspects the state at each step and chooses the next event's interval. Two modes — slow poll and fast burst — with hysteresis on the trigger so a noisy detector doesn't cause mode flicker.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

POLL_INTERVAL_S = 2.0       # slow acquisition between events
BURST_INTERVAL_S = 0.1      # fast acquisition during an event
TRIGGER_UP = 0.8            # detector probability to enter burst
TRIGGER_DOWN = 0.3          # detector probability to leave burst

def event_driven_acquisition(core, detector, channel="GFP", max_frames=2000):
    """Run an event-driven acquisition with a user-provided detector.

    detector(image) -> float in [0, 1]. Values above TRIGGER_UP enter burst
    mode; values below TRIGGER_DOWN return to poll mode.
    """
    state = {"mode": "poll", "prob": 0.0, "frames_in_burst": 0}

    def on_frame(img, event, meta=None):
        p = float(detector(img))
        state["prob"] = p
        if state["mode"] == "poll" and p > TRIGGER_UP:
            state["mode"] = "burst"
            state["frames_in_burst"] = 0
        elif state["mode"] == "burst" and p < TRIGGER_DOWN:
            state["mode"] = "poll"
        if state["mode"] == "burst":
            state["frames_in_burst"] += 1

    def gen():
        t = 0.0
        for i in range(max_frames):
            interval = BURST_INTERVAL_S if state["mode"] == "burst" else POLL_INTERVAL_S
            t += interval
            yield MDAEvent(
                channel={"config": channel},
                min_start_time=t,
                index={"t": i},
                metadata={"mode": state["mode"], "p": state["prob"]},
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` is the dispatcher.
- [[../../../src/core/workflows/]] — an event-driven workflow module does not yet exist; this paper motivates creating one (`src/core/workflows/event_driven.py`) that wraps the state machine above with configurable poll/burst intervals and a pluggable detector interface.
- The detector is user-supplied: a classical blob detector, a small CNN, or even an LLM-vision callback — the microscope side is detector-agnostic.

Calibration on a real prep: trigger thresholds should be set empirically from a baseline run (record detector output on known quiescent and known active frames, pick `TRIGGER_UP ≈ mean(active) − 1σ`, `TRIGGER_DOWN ≈ mean(quiescent) + 1σ`).

## Cited by

- [[Core/Concepts/Event-driven acquisition]] — concept note on the poll/burst state machine pattern.
- [[Core/Strategies/Adaptive acquisition]] — generator-driven adaptive acquisition, of which event-driven is a special case.
