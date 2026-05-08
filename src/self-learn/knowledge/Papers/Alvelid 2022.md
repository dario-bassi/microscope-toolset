---
title: Event-triggered STED imaging
authors: Jonatan Alvelid, Martina Damenti, Chiara Sgattoni, Ilaria Testa
year: 2022
venue: Nature Methods 19(10):1268-1275
doi: 10.1038/s41592-022-01588-y
url: https://www.nature.com/articles/s41592-022-01588-y
researched: 2026-04-24
---

## Abstract

Monitoring the proteins and lipids that mediate all cellular processes requires imaging methods with increased spatial and temporal resolution. STED (stimulated emission depletion) nanoscopy enables fast imaging of nanoscale structures in living cells but is limited by photobleaching. Here, we present event-triggered STED, an automated multiscale method capable of rapidly initiating two-dimensional (2D) and 3D STED imaging after detecting cellular events such as protein recruitment, vesicle trafficking and second messengers activity using biosensors. STED is applied in the vicinity of detected events to maximize the temporal resolution. We imaged synaptic vesicle dynamics at up to 24 Hz, 40 ms after local calcium activity; endocytosis and exocytosis events at up to 11 Hz, 40 ms after local protein recruitment or pH changes; and the interaction between endosomal vesicles at up to 3 Hz, 70 ms after approaching one another.  Event-triggered STED extends the capabilities of live nanoscale imaging, enabling novel biological observations in real time.

## Smart microscopy principle

Event-triggered STED couples two imaging modalities into a single adaptive loop: a gentle widefield / confocal **scout** continuously watches for a biosensor signature (calcium spike, pH drop, protein recruitment), and as soon as the detector fires the instrument switches the same region to high-resolution STED nanoscopy — localised around the event and only for its duration. The photobleaching and phototoxicity that otherwise cap STED's usable duration are spent where and when they matter, not across a whole field of quiescent pixels.

The novel contribution over generic event-driven acquisition (Mahecic 2022, same issue) is **modality switching** instead of rate switching: the scout and the measurement use different optical paths, exposure regimes, and spatial scales, and the trigger reconfigures the microscope rather than just speeding it up. The pattern generalises beyond STED — any rare event where a cheap monitoring modality can gate an expensive high-content one (confocal → STED, widefield → lattice light-sheet, brightfield → fluorescence) follows the same closed loop.

## Implementation on pymmcore-plus

On pymmcore-plus the trigger loop is a generator with an `on_frame` callback that inspects each scout frame, finds candidate event coordinates, and injects high-resolution events at those coordinates into the stream. The modality switch is a `StateDevice` / objective / config-group transition — same mechanism as channel switching, just wrapping a larger configuration change (laser lines, pinhole, scan mode).

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

SCOUT_INTERVAL_S = 0.1       # fast widefield monitor (biosensor)
STED_BURST_FRAMES = 20       # high-res frames per event
STED_PIXEL_SIZE_UM = 0.02    # nanoscale sampling

def event_triggered_sted(core, detector, max_frames=5000,
                         scout_channel="GCaMP",
                         sted_channel="STED-STAR635"):
    """Run scout → detect → STED-burst → resume loop.

    detector(image) -> list of (x_um, y_um) candidate event coordinates
    in world units, or [] for a quiescent frame.
    """
    queue = []  # pending STED events injected by the detector

    def on_frame(img, event, meta=None):
        if (event.metadata or {}).get("phase") != "scout":
            return
        # detector returns world-unit coordinates of biosensor events
        for (x_um, y_um) in detector(img):
            queue.append((x_um, y_um))

    def gen():
        for i in range(max_frames):
            # Flush any pending STED bursts before the next scout frame
            while queue:
                x_um, y_um = queue.pop(0)
                for k in range(STED_BURST_FRAMES):
                    yield MDAEvent(
                        x_pos=x_um, y_pos=y_um,
                        channel={"config": sted_channel},
                        metadata={"phase": "sted", "k": k},
                    )
            yield MDAEvent(
                channel={"config": scout_channel},
                min_start_time=i * SCOUT_INTERVAL_S,
                metadata={"phase": "scout", "i": i},
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` is the dispatcher; modality-switching events are standard `MDAEvent`s with a different `channel` config group.
- [[../../../src/core/workflows/]] — event-driven poll/burst wrappers live here (see `event_driven.py` motivated by Mahecic 2022); a sibling `modality_switching.py` would generalise that state machine over (scout-modality, capture-modality) pairs rather than (slow-rate, fast-rate). Not yet written — candidate for future work.
- On a real instrument the "modality switch" typically reconfigures multiple devices (laser shutters, galvo vs piezo scan, pinhole, detector gain). Wrap them in a single pymmcore-plus config group so the switch is one `core.setConfig("Mode", "STED")` call, not a sequence the generator has to orchestrate.

Calibration discipline: detector coordinates are in scout-modality world units; STED may use a different pixel size and FOV. Verify the world-unit round-trip with a fixed fiducial before trusting the loop on a live prep, and keep the STED burst small (10-50 frames) — the whole premise is that STED is too photodamaging to run continuously.

## Cited by

- [[Core/Concepts/Event-driven acquisition]] — modality-switching variant of the poll/burst state machine: instead of changing frame rate, the trigger changes optical mode.
- [[Core/Strategies/Adaptive acquisition]] — extends survey→zoom / patrol→burst from scale-switching to modality-switching (widefield scout → STED capture).
