---
title: Smart hybrid microscopy for cell-friendly detection of rare events
authors: Willi L. Stepp, Giorgio Tortarolo, Juan C. Landoni, Emine Berna Durmus, Santiago N. Rodriguez Alvarez, Kyle M. Douglass, Martin Weigert, Suliana Manley
year: 2026
venue: Nature Communications 17(1):1423
doi: 10.1038/s41467-025-68168-4
url: https://www.nature.com/articles/s41467-025-68168-4
researched: 2026-04-24
---

## Abstract

Fluorescence microscopy offers unparalleled access to the spatial organization and dynamics of biological events in living samples, yet capturing rare processes over extended durations remains challenging due to trade-offs between exposure to excitation light and sample health. Here, we introduce hybrid-EDA, an event-driven acquisition (EDA) framework that combines the gentleness and contextual richness of phase-contrast with the functional specificity of fluorescence. We develop surveillance for events of interest in label-free microscopy using dynamics-informed neural networks that trigger smart fluorescence acquisitions upon detection. This allows us to dramatically reduce phototoxic damage while obtaining specific and functional information from fluorescence when beneficial. We demonstrate how hybrid-EDA enables improved imaging acquisitions of organelle contacts and mitochondrial divisions.

## Smart microscopy principle

Hybrid-EDA generalises the Mahecic 2022 poll/burst paradigm from **rate-switching within one modality** to **modality-switching between two modalities**. The microscope runs continuously in phase-contrast — a label-free, photon-free contrast mode — and a dynamics-informed neural network watches the phase-contrast stream for the onset of rare events (organelle contacts, mitochondrial fission). Only when the detector fires does the instrument switch into fluorescence acquisition, where it can read out functional / molecular content the label-free channel doesn't carry.

The decisive design choice is that the **surveillance modality consumes effectively zero photon budget on the cell**, so total fluorescence dose collapses to whatever the events themselves cost — and the experiment can run as long as the cells live, not as long as the fluorophores survive. The "dynamics-informed" detector matters because single-frame phase-contrast doesn't contain enough information to predict an upcoming fission event; a small recurrent / temporal CNN over a short rolling window does. This paper is the canonical reference for **two-state modality EDA** (label-free → fluorescence) when the rare-event detector can be learned from label-free dynamics alone.

## Implementation on pymmcore-plus

The state machine is the same `IDLE → POLLING → EVENT → BURST → COOLDOWN` as in [[Papers/Mahecic 2022]], but the state transition switches a `Channel` config group rather than the inter-event interval. On a real micro-manager rig this is a `setConfig("Channel", "BF")` ⇄ `setConfig("Channel", "GFP")` swap, and the burst is a short multi-channel sub-MDA.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
from collections import deque

POLL_CHANNEL  = {"config": "PhaseContrast", "group": "Channel"}
BURST_CHANNELS = [
    {"config": "GFP",  "group": "Channel", "exposure": 50},
    {"config": "RFP",  "group": "Channel", "exposure": 50},
]
WINDOW_FRAMES = 8     # frames fed to the temporal detector
TRIGGER_UP   = 0.85   # detector probability that opens the burst
BURST_FRAMES = 20     # number of fluorescence frames per detected event
COOLDOWN     = 30     # poll frames before re-arming, prevents re-triggering decay

def hybrid_eda(core, temporal_detector, max_frames=5000):
    """Modality-switching EDA: phase-contrast surveillance, fluorescence on event.

    temporal_detector(window) -> p in [0, 1] over a deque of recent BF frames.
    """
    state = {"mode": "POLL", "cooldown": 0, "burst_left": 0,
             "window": deque(maxlen=WINDOW_FRAMES)}

    def on_frame(img, event, meta=None):
        if event.channel == POLL_CHANNEL:  # only label-free frames feed the detector
            state["window"].append(img)
            if state["mode"] == "POLL" and state["cooldown"] == 0 \
               and len(state["window"]) == WINDOW_FRAMES:
                if float(temporal_detector(list(state["window"]))) > TRIGGER_UP:
                    state["mode"] = "BURST"
                    state["burst_left"] = BURST_FRAMES

    def gen():
        for i in range(max_frames):
            if state["mode"] == "BURST":
                for ch in BURST_CHANNELS:
                    yield MDAEvent(channel=ch, index={"t": i})
                state["burst_left"] -= 1
                if state["burst_left"] <= 0:
                    state["mode"] = "POLL"
                    state["cooldown"] = COOLDOWN
            else:
                yield MDAEvent(channel=POLL_CHANNEL, index={"t": i})
                if state["cooldown"] > 0:
                    state["cooldown"] -= 1

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `src/core/hardware/core.py` — `run_events` is the dispatcher.
- `src/core/workflows/` — a generic `event_driven.py` module exposing the state machine with a pluggable detector + pluggable burst-channel list would cover both Mahecic-style rate switching (single channel, two intervals) and Stepp-style modality switching (different channels per state). This is currently a candidate for future work; existing `workflows/rate_control.py` covers only the dynamics-rate axis.
- The detector is the substantive contribution: a small temporal CNN over a rolling label-free window. On a real prep, the network is trained once on a labelled set of events from a separate fluorescence ground-truth run, then deployed in the BF-only loop. The microscope side is detector-agnostic and stays unchanged.

Calibration: collect a baseline pure-BF run, score all windows with the detector, set `TRIGGER_UP` from the detector's score distribution on known event vs. quiescent windows. Hysteresis on `COOLDOWN` is critical — a fired event has multi-frame after-effects in the dynamics window that would otherwise re-trigger the burst on the same event.

## Cited by

- [[Core/Concepts/Event-driven acquisition]] — concept note on the poll/burst state machine; this paper is the reference for the modality-switching variant (label-free → fluorescence) sitting alongside Mahecic 2022 (rate-switching) and Alvelid 2022 (widefield → STED).
- [[Core/Strategies/Gentle imaging]] — strategy note on minimising photon dose; this paper is the canonical example of using a label-free surveillance modality so the fluorescence budget is spent only on detected events.
