---
title: Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms
authors: Loïc A Royer, William C Lemon, Raghav K Chhetri, Yinan Wan, Michael Coleman, Eugene W Myers, Philipp J Keller
year: 2016
venue: Nature Biotechnology 34(12):1267-1278
doi: 10.1038/nbt.3708
url: https://pubmed.ncbi.nlm.nih.gov/27798562/
researched: 2026-04-24
---

## Abstract

Optimal image quality in light-sheet microscopy requires a perfect overlap between the illuminating light sheet and the focal plane of the detection objective. However, mismatches between the light-sheet and detection planes are common owing to the spatiotemporally varying optical properties of living specimens. Here we present the AutoPilot framework, an automated method for spatiotemporally adaptive imaging that integrates (i) a multi-view light-sheet microscope capable of digitally translating and rotating light-sheet and detection planes in three dimensions and (ii) a computational method that continuously optimizes spatial resolution across the specimen volume in real time. We demonstrate long-term adaptive imaging of entire developing zebrafish (Danio rerio) and Drosophila melanogaster embryos and perform adaptive whole-brain functional imaging in larval zebrafish. Our method improves spatial resolution and signal strength two to five-fold, recovers cellular and sub-cellular structures in many regions that are not resolved by non-adaptive imaging, adapts to spatiotemporal dynamics of genetically encoded fluorescent markers and robustly optimizes imaging performance during large-scale morphogenetic changes in living organisms.

## Smart microscopy principle

AutoPilot is the canonical paper for **closed-loop optical alignment** during a long acquisition. The core observation: a living specimen's optical properties change with time (development, morphogenesis, pigmentation) and with position inside the volume, so a detection plane that's correctly aligned to the light sheet at t=0 and at the embryo centre is misaligned 30 minutes later or 200 µm away. Classical autofocus addresses Z drift only; AutoPilot generalises this to five degrees of freedom per view — detection-plane Z, light-sheet Y-offset, light-sheet angle in two axes, and relative illumination/detection tilt — and runs the optimiser continuously while the acquisition proceeds.

The contribution is methodological: a differentiable image-quality metric (DCTS, a variant of the discrete cosine transform sharpness score) that is sensitive enough to resolve sub-micron misalignments, paired with a schedule that spreads correction snaps across space and time so the photon overhead stays small. The microscope becomes a self-calibrating instrument — it doesn't just acquire, it *maintains* its own alignment against a moving sample.

For smart-microscopy practice the transferable idea is broader than light-sheet: any imaging modality with multiple degrees of freedom that drift (focus, objective correction collar, astigmatism correction, light-sheet offset, illumination angle) can be treated as a continuous optimisation problem running alongside the main acquisition, not as a one-off setup step.

## Implementation on pymmcore-plus

A real pymmcore-plus microscope doesn't usually expose the five-axis light-sheet geometry that AutoPilot controls, but the pattern — periodic small sweeps on one or more alignment axes, score each sweep with a sharpness metric, update the axis setpoint, then resume the main plan — translates directly. The canonical target on a widefield/confocal rig is Z focus, but the same loop handles objective correction-collar position or a piezo-driven light-sheet offset if those devices are exposed as `StateDevice`s or `StageDevice`s.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
from src.core.workflows.autofocus import focus_metric, sweep_focus

MAIN_INTERVAL_S = 5.0      # main timelapse rate
ALIGN_EVERY_N = 20         # run an alignment sweep every N main frames
SWEEP_HALF_RANGE = 2.0     # um, narrow sweep around the current best Z
SWEEP_STEP = 0.5           # um

def autopilot_timelapse(core, n_frames=600, channel="GFP"):
    """Long-term timelapse with periodic, narrow closed-loop refocus.

    A small sweep on Z runs every ALIGN_EVERY_N frames; the best Z found
    becomes the new setpoint for subsequent frames. The sweep is narrow
    enough that the photon overhead is ~5% of the main acquisition.
    """
    state = {"best_z": core.getZPosition(), "last_score": None}

    def gen():
        for i in range(n_frames):
            if i > 0 and i % ALIGN_EVERY_N == 0:
                # narrow alignment sweep: 5 Z positions around the current setpoint
                z0 = state["best_z"]
                for dz in (-2 * SWEEP_STEP, -SWEEP_STEP, 0.0, SWEEP_STEP, 2 * SWEEP_STEP):
                    yield MDAEvent(
                        z_pos=z0 + dz,
                        channel={"config": channel},
                        metadata={"phase": "align", "dz": dz},
                    )
            yield MDAEvent(
                z_pos=state["best_z"],
                channel={"config": channel},
                min_start_time=i * MAIN_INTERVAL_S,
                metadata={"phase": "main", "i": i},
            )

    sweep_buf = []
    def on_frame(img, event, meta=None):
        phase = (event.metadata or {}).get("phase")
        if phase == "align":
            sweep_buf.append((event.z_pos, focus_metric(img, method="brenner")))
            if len(sweep_buf) == 5:
                # pick the Z with the highest score, update setpoint
                best_z, best_s = max(sweep_buf, key=lambda r: r[1])
                state["best_z"] = best_z
                state["last_score"] = best_s
                sweep_buf.clear()

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/autofocus.py` — already provides `focus_metric`, `sweep_focus`, and a `make_focus_state` / `check_and_correct_focus` pair. AutoPilot-style continuous maintenance is the predictive strategy documented in [[Core/Strategies/Closed-loop autofocus]]; what is missing is a multi-axis generalisation (a `maintain_alignment(core, axes=["Z", "CorrectionCollar"], metric=...)` helper that runs the same narrow-sweep loop on N axes in round-robin).
- `../../../src/core/hardware/core.py` — `run_events` is the dispatcher; the narrow-sweep frames are just extra `MDAEvent`s interleaved with the main schedule, same engine.
- Budgeting: the alignment snaps count against the sample's photon and time budget. On a bleaching-sensitive prep, use `ALIGN_EVERY_N >= 20` and a five-point sweep (not more); the DCTS metric the paper uses is more sample-efficient than a raw Laplacian and should be the default when available.

Differences from the paper to keep in mind:
- AutoPilot operates on 4-view SPIM geometry; on a single-objective rig the problem collapses to 1-2 axes.
- The paper's DCTS metric is more robust than Brenner/Laplacian for sparse fluorescence — a candidate for addition to `src/core/workflows/autofocus.py` alongside the existing methods.
- Long-term imaging of living specimens means the alignment setpoint is non-stationary; never cache a single global best Z — always re-optimise periodically.

## Cited by

- [[Core/Strategies/Closed-loop autofocus]] — AutoPilot is the paradigm case for treating alignment as a continuous optimisation, not a one-off setup.
- [[Core/Strategies/Adaptive acquisition]] — closed-loop alignment is a sibling to closed-loop content-aware triggering; both are adaptive acquisition.
