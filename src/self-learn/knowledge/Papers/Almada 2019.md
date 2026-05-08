---
title: Automating multimodal microscopy with NanoJ-Fluidics
authors: Pedro Almada, Pedro M. Pereira, Siân Culley, Ghislaine Caillol, Fanny Boroni-Rueda, Christina L. Dix, Guillaume Charras, Buzz Baum, Romain F. Laine, Christophe Leterrier, Ricardo Henriques
year: 2019
venue: Nature Communications 10:1223
doi: 10.1038/s41467-019-09231-9
url: https://www.nature.com/articles/s41467-019-09231-9
researched: 2026-04-24
---

## Abstract

Combining and multiplexing microscopy approaches is crucial to understand cellular events, but requires elaborate workflows. Here, we present a robust, open-source approach for treating, labelling and imaging live or fixed cells in automated sequences. NanoJ-Fluidics is based on low-cost Lego hardware controlled by ImageJ-based software, making high-content, multimodal imaging easy to implement on any microscope with high reproducibility. We demonstrate its capacity on event-driven, super-resolved live-to-fixed and multiplexed STORM/DNA-PAINT experiments.

## Smart microscopy principle

NanoJ-Fluidics closes the loop between acquisition and **sample state itself**: instead of only choosing when and where to image, the microscope also decides *what the sample chemistry should be* at each step — fixation, labelling-round N, wash, imaging buffer — and drives pumps to make it so. The paradigm-defining demonstration is live-to-fixed event capture: the microscope watches a live sample, and when a trigger condition fires, the fluidics automatically fixes the sample mid-experiment and switches to a super-resolution imaging protocol on the very same cells.

The second paradigm is multiplexed labelling cycles. In multiplexed STORM / DNA-PAINT, imaging capacity is no longer limited by the number of spectrally separable fluorophores: a pump exchanges the imaging oligo / elutes / adds the next probe, and the same channel images a different target each round. The acquisition is a sequence of (probe-on, image, probe-off) cycles rather than a one-shot snapshot.

The broader contribution is that the microscope's control surface now includes fluid-exchange events as first-class actions, treated on equal footing with stage moves, channel changes, and exposures. Protocol reproducibility jumps because pump timings are scripted rather than pipetted by hand.

## Implementation on pymmcore-plus

pymmcore-plus does not model fluidics natively, but Micro-Manager's device abstraction handles syringe-pump controllers as `StateDevice` or `GenericDevice` — set a config group (e.g. `"Fluidics"` with presets `"buffer"`, `"fix"`, `"probe_A"`, `"probe_B"`, `"wash"`), and fluid steps become ordinary `MDAEvent` transitions with a pump-settle delay. On a real rig the same pattern runs against any Micro-Manager-compatible pump; on a custom rig, expose the pump via a thin adapter and wire it as a core Property or use a callback generator.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

# Multiplexed DNA-PAINT-style acquisition: N probe rounds, each round
# exchanges fluid, settles, then images all positions in one channel.

PROBES = ["probe_A", "probe_B", "probe_C"]
POSITIONS = [(0, 0), (256, 0), (0, 256)]
SETTLE_S = 30.0   # buffer exchange + equilibration
WASH_S = 15.0     # elute between rounds

def multiplexed_rounds(probes, positions):
    t = 0.0
    for r, probe in enumerate(probes):
        # Fluid-exchange event: drive the pump to the probe preset.
        yield MDAEvent(
            min_start_time=t,
            properties=[("Fluidics", "Label", probe)],
            metadata={"phase": "exchange", "round": r, "probe": probe},
        )
        t += SETTLE_S
        # Image every position in this probe round.
        for p, (x, y) in enumerate(positions):
            yield MDAEvent(
                min_start_time=t,
                x_pos=x, y_pos=y,
                channel={"config": "Cy5"},
                index={"r": r, "p": p},
                metadata={"phase": "image", "round": r, "probe": probe},
            )
            t += 1.0
        # Elute before the next round.
        yield MDAEvent(
            min_start_time=t,
            properties=[("Fluidics", "Label", "wash")],
            metadata={"phase": "wash", "round": r},
        )
        t += WASH_S

run_events(core, list(multiplexed_rounds(PROBES, POSITIONS)))
```

Event-driven live-to-fixed variant: run an adaptive generator (as in the `Event-driven acquisition` concept) during the live phase; when the detector fires, yield a fluidics event that injects fixative, then switch to the super-resolution MDA block. The same `run_events` dispatcher handles all of it because the fluidics device is just another Micro-Manager state.

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` already drives `properties=[(group, preset)]` transitions; no new code needed for the generic case.
- `src/core/workflows/fluidics.py` — does not yet exist; this paper motivates a helper module that wraps (set-pump, wait-settle, snap) into a single `FluidStep` event-generator primitive and composes it with multi-position scans. Candidate for future work.
- Calibration discipline on a real rig: measure real buffer-exchange time on the specific tubing / flow-cell geometry; `SETTLE_S` is protocol-specific, not a library default.

## Cited by

- [[Core/Strategies/Multichannel scan]] — multiplexed labelling rounds extend the channel axis via fluidics; each round = new effective channel at the same excitation.
- [[Core/Strategies/Timelapse design]] — live-to-fixed protocols let a single timelapse span live dynamics and fixed-state super-resolution on the same cells.
