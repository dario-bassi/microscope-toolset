---
title: Closed-loop optogenetic control of cell biology enables outcome-driven microscopy
authors: Josiah B. Passmore, Alfredo Rates, Jakob Schröder, Menno T. P. van Laarhoven, Vincent J. W. Hellebrekers, Henrik G. van Hoef, Antonius J. M. Geurts, Wendy van Straaten, Wilco Nijenhuis, Florian Berger, Carlas S. Smith, Ihor Smal, Lukas C. Kapitein
year: 2025
venue: Nature Communications 17:1087
doi: 10.1038/s41467-025-67848-5
url: https://www.nature.com/articles/s41467-025-67848-5
researched: 2026-04-24
---

## Abstract

Smart microscopy is transforming biological imaging by integrating real-time analysis with adaptive acquisition to enhance imaging efficiency. Whereas many emerging implementations are event-driven and focus on on-demand data acquisition to reduce phototoxicity, we here present 'outcome-driven' microscopy, a framework combining smart microscopy with optogenetics to control cell biological processes and achieve predefined outcomes. We validate this approach using light-based control of cell migration and nucleocytoplasmic transport, demonstrating robust spatiotemporal control of cellular behaviour in single cells and in cell populations.

## Smart microscopy principle

Outcome-driven microscopy extends the event-driven paradigm from *observing on demand* to *steering on demand*. Instead of the detector deciding when to acquire, a controller decides how to illuminate: each frame feeds an analysis step that compares the current cell state to a target outcome (a cell position, a nuclear/cytoplasmic ratio, a migration trajectory), then updates an SLM pattern that optogenetically pushes the sample toward that target. The loop closes at the biology, not at the acquisition rate.

The contribution is demonstrating that a microscope can act as a **controller of cell behaviour** rather than a passive observer. The authors validate this on two orthogonal axes: cell migration (spatial outcome — steer a cell along a chosen trajectory) and nucleocytoplasmic shuttling (compartmental outcome — drive a defined N/C ratio). Both use the same observe → measure → pattern-SLM → illuminate loop with different readouts and different optogenetic actuators.

The practical implication for anyone running optogenetics on a microscope: your SLM mask is a control signal, not a fixed stimulus. Treating it as feedback-driven — recomputed each frame from the latest image — is the difference between stimulating cells and controlling them.

## Implementation on pymmcore-plus

The loop is a straightforward `run_events` generator with an `on_frame` callback that measures the current outcome, computes an error versus the target, and emits a fresh `MDAEvent` carrying an updated `SLMImage`. Key design points: (1) mask and error update each frame — nothing is precomputed; (2) a saturating controller (clip the mask radius / stim exposure to physiological bounds) prevents integrator windup on sub-threshold opsins; (3) a termination criterion (tolerance on the outcome, or a frame budget) closes the loop cleanly.

```python
from useq import MDAEvent, SLMImage
from src.core.hardware.core import run_events, make_slm_circle

TARGET_XY = (256, 256)          # outcome: steer cell to image centre
TOLERANCE_PX = 15               # stop when within this of target
STIM_RADIUS_PX = 40             # SLM spot size for opto-actuation
MAX_FRAMES = 200

def outcome_driven_migration(core, detect_cell, channel="GFP"):
    """Close-loop steer a migratory cell toward TARGET_XY by moving the
    SLM stim spot to pull the cell forward each frame.

    Generator-+-on_frame skeleton (cf. [[Core/Approach/OADA loop]]).
    """
    initial_mask = make_slm_circle(TARGET_XY, STIM_RADIUS_PX, core=core)
    state = {"done": False, "next_mask": initial_mask, "last_xy": None}

    def on_frame(img, event):
        if state["done"]:
            return
        cx, cy = detect_cell(img)
        state["last_xy"] = (cx, cy)
        dx, dy = TARGET_XY[0] - cx, TARGET_XY[1] - cy
        if (dx * dx + dy * dy) ** 0.5 < TOLERANCE_PX:
            state["done"] = True
            return
        # Place stim spot one step ahead of the cell, clipped to FOV.
        step = 30  # px per frame toward target; calibrate on sample
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        sx = int(cx + step * dx / norm)
        sy = int(cy + step * dy / norm)
        state["next_mask"] = make_slm_circle((sx, sy), STIM_RADIUS_PX, core=core)

    def gen():
        for _ in range(MAX_FRAMES):
            if state["done"]:
                return
            yield MDAEvent(
                channel={"config": channel},
                slm_image=SLMImage(data=state["next_mask"], device="SLM"),
            )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events`, `make_slm_circle`, `apply_slm`.
- `[[Core/Approach/OADA loop]]` — the generator + `on_frame` scaffold this snippet uses: `on_frame` updates the next mask in shared state, the generator emits an `MDAEvent` carrying that mask before the next snap.
- An `outcome_controller` module does not yet exist. A clean API would be: `run_outcome_loop(core, measure_fn, control_fn, target, tolerance, max_frames)` where `control_fn(state, target) -> SLMImage` is user-supplied. That separates the reusable loop scaffolding from the sample-specific measurement and control law.

Calibration on a real prep: the `step` parameter (how far ahead of the cell to place the stim spot) and `STIM_RADIUS_PX` (how many cells get activated) both need empirical tuning — too small and the cell doesn't turn, too large and nearby cells respond. A pre-experiment calibration run sweeping radius × exposure × step, scoring against observed migration per frame, is worth the frames.

The same scaffolding retargets to nucleocytoplasmic transport by swapping `detect_cell` for a N/C-ratio measurement and `TARGET_XY` for a target ratio — the loop doesn't change, only the readout and the error term.

## Cited by

- [[Core/Strategies/SLM optogenetics]] — closed-loop SLM patterns; this paper is the paradigm reference for feedback-driven stimulation.
- [[Core/Strategies/Feedback control]] — observe → compute → modify → snap loop, instantiated for optogenetic actuation.
- [[Core/Strategies/Closed-loop state device]] — the SLM pattern register is the state device; outcome-driven microscopy is the application.
