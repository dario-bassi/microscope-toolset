---
title: Deep learning for single-shot autofocus microscopy
authors: Henry Pinkard, Zachary Phillips, Arman Babakhani, Daniel A. Fletcher, Laura Waller
year: 2019
venue: Optica 6(6):794-797
doi: 10.1364/OPTICA.6.000794
url: https://opg.optica.org/optica/fulltext.cfm?uri=optica-6-6-794&id=413486
researched: 2026-04-24
---

## Abstract

Maintaining an in-focus image over long time scales is an essential and nontrivial task for a variety of microscopy applications. Here, we describe a fast, robust autofocusing method compatible with a wide range of existing microscopes. It requires only the addition of one or a few off-axis illumination sources (e.g., LEDs), and can predict the focus correction from a single image with this illumination. We designed a neural network architecture, the fully connected Fourier neural network (FCFNN), that exploits an understanding of the physics of the illumination to make accurate predictions with 2–3 orders of magnitude fewer learned parameters and less memory usage than existing state-of-the-art architectures, allowing it to be trained without any specialized hardware. We provide an open-source implementation of our method, to enable fast, inexpensive autofocus compatible with a variety of microscopes.

## Smart microscopy principle

Single-shot deep-learning autofocus replaces the classical Z-sweep (snap N planes, score each, pick the peak) with **one acquisition whose pixel content already encodes the signed defocus**. By illuminating the sample off-axis with a coherent LED, a defocused image carries directional fringe information that a trained neural network can regress into a µm-valued `Δz` correction. A single frame gives the sign *and* magnitude of the focus error — the microscope knows which way to move and how far, without ever taking a stack.

The smart-microscopy contribution is to move autofocus from **measurement (many snaps → fit a curve)** to **inference (one snap → predicted correction)**. This collapses the photon/time budget of autofocus by an order of magnitude and makes it cheap enough to run before every position of a multi-position timelapse — enabling focus correction at timescales where a Z-sweep would be prohibitive. The FCFNN architecture embeds the illumination physics so the trained model is small, CPU-inferrable (~50 ms per 2048² image), and trainable on a desktop without a GPU.

## Implementation on pymmcore-plus

On a pymmcore-plus rig the pattern is: (i) configure the off-axis LED as a coherent illumination channel (usually a programmable LED array exposed as a `StateDevice`, or a single LED on a shutter); (ii) at every position / timepoint, snap one frame under that channel and pass it to a pre-trained `predict_defocus(img) -> dz_um` callback; (iii) move the Z stage by `dz_um` and proceed with the main acquisition. The model itself is an inference-only callback — training is offline from paired focal stacks (see paper's Fig. 1a).

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

# Pre-trained FCFNN or equivalent single-shot focus model.
# Contract: predict_defocus(img_coherent) -> signed Δz in µm.
predict_defocus = load_focus_model("fcfnn_widefield_20x.pt")

AF_CHANNEL = "LED_offaxis"   # coherent off-axis illumination
MAIN_CHANNEL = "GFP"

def single_shot_af_timelapse(core, positions, n_frames=200, dt_s=30.0):
    """Timelapse with single-shot deep-learning refocus before each main frame."""
    state = {"z_by_pos": {i: core.getZPosition() for i, _ in enumerate(positions)}}

    def on_frame(img, event, meta=None):
        phase = (event.metadata or {}).get("phase")
        if phase == "af":
            pos_idx = event.metadata["pos_idx"]
            dz = float(predict_defocus(img))
            state["z_by_pos"][pos_idx] += dz   # update cached focus

    def gen():
        for i in range(n_frames):
            t = i * dt_s
            for pos_idx, (x, y) in enumerate(positions):
                # 1. single-shot AF frame under off-axis coherent illumination
                yield MDAEvent(
                    x_pos=x, y_pos=y,
                    z_pos=state["z_by_pos"][pos_idx],
                    channel={"config": AF_CHANNEL},
                    min_start_time=t,
                    metadata={"phase": "af", "pos_idx": pos_idx},
                )
                # 2. main science frame at the corrected Z (read updated state)
                yield MDAEvent(
                    x_pos=x, y_pos=y,
                    z_pos=state["z_by_pos"][pos_idx],
                    channel={"config": MAIN_CHANNEL},
                    min_start_time=t,
                    metadata={"phase": "main", "pos_idx": pos_idx, "i": i},
                )

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/autofocus.py` — currently implements the classical reactive / threshold / predictive strategies (Brenner-metric sweeps). A single-shot ML predictor is a **fourth strategy** that slots in beside them: same `make_focus_state` / `check_and_correct_focus` surface, but `check_and_correct` returns the model's prediction instead of sweeping. Would add `predict_defocus(img)` as the canonical callback contract.
- `../../../src/core/hardware/core.py` — `run_events` dispatches both AF and main frames; no engine changes needed.
- Missing piece on most rigs: the off-axis LED. A programmable LED dome (as used in the paper) is ideal; a single off-axis LED on a shutter works for a fixed magnification. Without the coherent off-axis frame, the FCFNN doesn't apply — but the *interface* (single-frame → Δz) still works for other single-shot schemes (e.g. defocused-intensity regression on widefield fluorescence, at lower accuracy).

Calibration: the model is sample-class specific (the paper shows a model trained on white-blood-cells does not transfer to tissue sections without retraining). On a new sample type, collect ~400 focal stacks with 1 µm spacing over ±30 µm once, train offline (~1.5 h CPU or 30 min GPU per the paper), then deploy the inference callback — it amortises across the entire timelapse campaign.

## Cited by

- [[Core/Strategies/Closed-loop autofocus]] — adds single-shot ML inference as a fourth autofocus strategy alongside reactive / threshold-triggered / predictive sweep-based methods; collapses the per-position autofocus snap budget from N (sweep) to 1 (predict).
