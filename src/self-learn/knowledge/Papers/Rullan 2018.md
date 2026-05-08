---
title: An Optogenetic Platform for Real-Time, Single-Cell Interrogation of Stochastic Transcriptional Regulation
authors: Marc Rullan, Dirk Benzinger, Gregor W. Schmidt, Andreas Milias-Argeitis, Mustafa Khammash
year: 2018
venue: Molecular Cell 70(4):745-756.e6
doi: 10.1016/j.molcel.2018.04.012
url: https://www.cell.com/molecular-cell/fulltext/S1097-2765(18)30308-3
researched: 2026-04-25
---

## Abstract

Transcription is a highly regulated and inherently stochastic process. The complexity of signal transduction and gene regulation makes it challenging to analyze how the dynamic activity of transcriptional regulators affects stochastic transcription. By combining a fast-acting, photo-regulatable transcription factor with nascent RNA quantification in live cells and an experimental setup for precise spatiotemporal delivery of light inputs, we constructed a platform for the real-time, single-cell interrogation of transcription in *Saccharomyces cerevisiae*. We show that transcriptional activation and deactivation are fast and memoryless. By analyzing the temporal activity of individual cells, we found that transcription occurs in bursts, whose duration and timing are modulated by transcription factor activity. Using our platform, we regulated transcription via light-driven feedback loops at the single-cell level. Feedback markedly reduced cell-to-cell variability and led to qualitative differences in cellular transcriptional dynamics. Our platform establishes a flexible method for studying transcriptional dynamics in single cells.

## Smart microscopy principle

Rullan et al. is the **canonical microscope-based cybergenetics paper**: a closed loop that runs at the gene-circuit scale, driven by per-cell live imaging of nascent transcription, with light delivered through a DMD onto each cell individually. The plant is a synthetic gene circuit (VP-EL222 light-activated transcription factor → 5×EL222 binding sites → PP7-tagged nascent RNA reporter); the sensor is a microscope counting diffraction-limited PP7-tdPCP-tdmRuby3 nuclear spots in real time; the actuator is a DLP DMD projecting per-cell blue-light patterns; the controller is a per-cell integral feedback law `I(t_k) = K_I · Σ e(t_n)` updated every 2 minutes. Hundreds of yeast cells, trapped one-cell-thick in a PDMS microfluidic chamber, run independent feedback loops in parallel.

The contribution is establishing that the **right scale for closed-loop gene-expression control is the single cell, not the population**. The same paper compares two regimes side by side: an "in-silico cell" controller that targets the *population mean* (one shared light input), and an "in-silico cell-specific" controller that targets each cell's own trajectory (per-cell light input). Per-cell control collapses cell-to-cell variability dramatically and produces qualitatively different transcription dynamics — population-average control cannot reach. This is the empirical foundation that later papers in the cluster build on: [[Papers/Lugagne 2024]] replaces integral with deep-MPC, [[Papers/Passmore 2025]] replaces gene expression with cell-behaviour outcomes, [[Papers/Hinderling 2025]] replaces the bespoke pipeline with an open Python substrate. Rullan 2018 is the earliest microscope-based instance of all three.

The practical implication for anyone building a closed-loop gene-circuit experiment: target the per-cell loop from day one. Population-mean controllers are easier to implement (one PMT or one cytometry sample suffices) but the variance-reduction story only appears at single-cell resolution. The microscope is the actuator, not just the sensor — the DMD pattern *is* the control signal.

## Implementation on pymmcore-plus

The Rullan loop maps onto a `run_events` generator with an `on_frame` callback that segments + tracks each yeast cell, counts PP7 nascent-RNA spots, updates a per-cell integral-error accumulator, and rebuilds the DMD mask so each cell receives its own light intensity at the next time point. Key design points: (1) per-cell state is the integral-error accumulator, indexed by tracker identity — identity flicker between frames corrupts the controller as much as it corrupts the segmentation; (2) the control interval (2 min in Rullan; matched to the EL222 dimerisation/dissociation timescales) is the budget for segment + count + integrate + repaint; (3) the integral gain `K_I` saturates at the LED's maximum brightness — clip the per-cell light value to avoid wind-up when a cell drifts out of focus and looks transiently dim.

```python
from useq import MDAEvent, SLMImage
import numpy as np
from src.core.hardware.core import run_events

CONTROL_INTERVAL_S = 120.0       # 2 min between updates (Rullan protocol)
K_I = 0.05                       # integral gain (a.u. light / spot-count error)
LIGHT_MAX = 255                  # DMD intensity ceiling (8-bit pattern)
MAX_FRAMES = 240                 # ~8 h of control


def cybergenetic_per_cell_control(
    core,
    segment_and_track,        # fn(image) -> dict cell_id -> mask + centroid
    count_nascent_rna,        # fn(image, mask) -> int (PP7 spot count)
    reference_per_cell,       # dict cell_id -> target spot count (or callable)
    build_dmd_pattern,        # fn(dict cell_id -> light_level, masks) -> np.uint8
    channel="brightfield",
):
    """Per-cell integral-feedback control of transcription via DMD.

    Each control step: image cells, count nascent RNA spots per cell,
    update each cell's integral-error accumulator, compute its next
    light intensity, rebuild the DMD pattern. Light is delivered by
    the next MDAEvent's SLMImage.
    """
    state = {
        "integral": {},                 # cell_id -> accumulated error
        "next_mask": np.zeros((1, 1), dtype=np.uint8),  # placeholder
    }

    def on_frame(img, event):
        cells = segment_and_track(img)
        light_per_cell = {}
        for cid, cinfo in cells.items():
            count = count_nascent_rna(img, cinfo["mask"])
            ref = reference_per_cell(cid) if callable(reference_per_cell) \
                  else reference_per_cell.get(cid, 0)
            err = ref - count
            acc = state["integral"].get(cid, 0.0) + err
            state["integral"][cid] = acc
            light = float(np.clip(K_I * acc, 0, LIGHT_MAX))
            light_per_cell[cid] = light
        state["next_mask"] = build_dmd_pattern(light_per_cell, cells)

    def gen():
        for t in range(MAX_FRAMES):
            yield MDAEvent(
                channel={"config": channel},
                slm_image=SLMImage(data=state["next_mask"], device="SLM"),
                min_start_time=t * CONTROL_INTERVAL_S,
                index={"t": t},
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` dispatches the events; `make_slm_circle` / `apply_slm` rasterise per-cell stim spots into a DMD frame. Rullan's "DLP LightCrafter 4500" maps directly onto any pymmcore-plus `genericSLM` device.
- `[[Core/Approach/OADA loop]]` — the generator + `on_frame` scaffold this snippet uses: `on_frame` updates per-cell integral-error state and stages the next DMD mask; the generator emits an `MDAEvent` carrying that mask before the next snap.
- `../../../src/core/workflows/stage_tracking.py` — supplies the per-cell identity tracker that the integral accumulator depends on. The same lesson as [[Papers/Hinderling 2025]]: when a cell flips identity, its integrator state flips with it, and the controller drives the wrong cell.
- `../../../src/core/workflows/optogenetics.py` — already exposes `target_cells` and stim-pattern helpers; the cybergenetics contribution to absorb here is the **per-cell controller-state container** (a `{cell_id: integrator}` dict that survives across frames). A clean cybergenetics-style API would be `run_per_cell_controller(core, segment_fn, measure_fn, control_law, build_pattern_fn, tracker, control_interval)` separating the reusable scaffold from the four sample-specific pieces (segmentation, measurement, control law, pattern synthesis).

Calibration on a real prep: (i) identify the control timescale — Rullan's 2 min update matches EL222 dimerisation/dissociation; faster TFs (LightOn, CRY2) tolerate sub-minute updates, slower outputs (mature fluorophore) need slower loops. (ii) Identify the *measurement* timescale separately — PP7 nascent RNA reads transcription on the same minute timescale as the TF; if you're reading mature fluorescence the lag adds 30-60 min and the integral controller will hunt unless gain is reduced. (iii) Single-cell trapping is the silent prerequisite: Rullan uses a one-cell-thick PDMS chamber so each cell stays in focus and gets reliably segmented across hours; without trapping, identity tracking fails and the controller diverges. (iv) Saturate the per-cell light intensity at the level where dose-response is still in the linear regime — beyond saturation the integral term winds up and the controller can't recover when the disturbance reverses.

The same scaffolding generalises to other transcriptional optogenetic toolkits (LightOn, LOV-bZIP, CRY2-CIB1) by swapping the EL222 binding sites for the corresponding promoter and re-tuning `K_I` against the new dose-response curve. The integral feedback law is the simplest possible cybergenetic controller; [[Papers/Lugagne 2024]] is the deep-MPC successor on the same scaffold.

## Cited by

- [[Core/Strategies/Feedback control]] — Rullan 2018 is the canonical microscope-based instance of the observe → compute → modify → snap loop applied to gene expression: per-cell integral controller, DMD actuator, nascent-RNA sensor, hundreds of cells in parallel.
- [[Core/Strategies/SLM optogenetics]] — DMD-targeted per-cell light delivery driving an optogenetic transcription factor; predates [[Papers/Passmore 2025]] and [[Papers/Hinderling 2025]] but uses the same actuator pattern.
- [[Core/Strategies/Closed-loop state device]] — the DMD pattern register is the state device; cybergenetic transcription control is the application that drove the architecture.
- [[Core/Strategies/Simultaneous SLM targeting]] — per-cell light intensities computed from per-cell measurements, projected as one DMD mask each control step. Hundreds of independent loops time-share one DMD.
- [[Papers/Lugagne 2024]] — deep-MPC successor in the same lineage (Khammash group). Replaces the integral law with a learned forward model + receding-horizon optimiser; same actuator (DMD), same sensor (per-cell fluorescence), same per-cell-loop architecture, but trajectory-tracks arbitrary dynamic references rather than holding a setpoint. Rullan 2018 is the architectural ancestor.
- [[Papers/Passmore 2025]] — outcome-driven sibling that closes the loop on **cell behaviour** (migration trajectory, N/C ratio) instead of **gene expression**; same observe → compute → re-pattern loop, different readout. The two papers together delineate the two halves of microscope-based optogenetic feedback: cybergenetics (gene circuits) and outcome-driven microscopy (cell behaviour).
- [[Papers/Hinderling 2025]] — open Python+Micro-Manager substrate that operationalises the Rullan-style per-cell loop for any optogenetic prep; FARO inherits the per-cell scoring + DMD pattern synthesis architecture and adds decoupled scheduler/analysis threads, interactive segmentation, and explicit retargeting across subcellular / single-cell / tissue scales.
- [[Papers/Fox 2022]] — MicroMator factors reactive microscopy into composable event-condition-action rules and demonstrates single-cell optogenetic control as a flagship use case (yeast, EL222) — implementing the Rullan paradigm as ECA rules in Python rather than a bespoke MATLAB pipeline. Sibling open-source substrate to Hinderling 2025 / Casas Moreno 2021/2023 / Roos 2024.
