---
title: "Resolution enhancement with a task-assisted GAN to guide optical nanoscopy image analysis and acquisition"
authors: Catherine Bouchard, Theresa Wiesner, Andréanne Deschênes, Anthony Bilodeau, Benoît Turcotte, Christian Gagné, Flavie Lavoie-Cardinal
year: 2023
venue: Nature Machine Intelligence 5(8):830-844
doi: 10.1038/s42256-023-00689-3
url: https://www.nature.com/articles/s42256-023-00689-3
researched: 2026-04-24
---

## Abstract

Super-resolution fluorescence microscopy methods enable the characterization of nanostructures in living and fixed biological tissues. However, they require the adjustment of multiple imaging parameters while attempting to satisfy conflicting objectives, such as maximizing spatial and temporal resolution while minimizing light exposure. To overcome the limitations imposed by these trade-offs, post-acquisition algorithmic approaches have been proposed for resolution enhancement and image-quality improvement. Here we introduce the task-assisted generative adversarial network (TA-GAN), which incorporates an auxiliary task (for example, segmentation, localization) closely related to the observed biological nanostructure characterization. We evaluate how the TA-GAN improves generative accuracy over unassisted methods, using images acquired with different modalities such as confocal, bright-field, stimulated emission depletion and structured illumination microscopy. The TA-GAN is incorporated directly into the acquisition pipeline of the microscope to predict the nanometric content of the field of view without requiring the acquisition of a super-resolved image. This information is used to automatically select the imaging modality and regions of interest, optimizing the acquisition sequence by reducing light exposure. Data-driven microscopy methods like the TA-GAN will enable the observation of dynamic molecular processes with spatial and temporal resolutions that surpass the limits currently imposed by the trade-offs constraining super-resolution microscopy.

## Smart microscopy principle

The TA-GAN turns a deep generator into both a virtual super-resolution scout and an event trigger. A confocal frame is run through a GAN co-trained on an auxiliary task (segmentation, localisation, dendritic-spine classification) that ties its features to the same nanostructures a STED acquisition would resolve. The synthetic STED image is compared against the predicted output before and after candidate biological events; when the disagreement (effectively the model's reconstruction uncertainty about the true nanoscale content) crosses a threshold, the microscope fires a real STED acquisition only on that ROI. This swaps the usual smart-microscopy pattern — hand-engineered biological cue triggers high-resolution capture — for an unsupervised "the model isn't sure any more, so look harder" trigger. In their live F-actin / synapse demonstration the loop captured 1.6 STED images per 15-confocal sequence, an 89% light-dose reduction in the central ROI, while still catching the structural rearrangements that defined the experiment. The same TA-GAN doubles as a content-aware ROI selector during the confocal pre-scan: only fields whose synthetic SR prediction shows actionable nanostructure earn a STED follow-up.

## Implementation on pymmcore-plus

The control loop is two nested generators: an outer time-loop polls confocal (or any low-dose modality), and a per-frame trigger compares the current TA-GAN prediction against a recent baseline. When the prediction-divergence score exceeds a threshold, a `CustomAction` (or modality switch via the Channel state device) injects a STED burst on the changed ROI; otherwise the loop continues at low dose. The TA-GAN inference call sits inside the `on_frame` callback, so the trigger evaluates synchronously against the freshest frame.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

state = {"prev_pred": None, "rois": []}
DIVERGENCE_THRESHOLD = 0.15  # tuned per sample class on a calibration set

def ta_gan_trigger(img, event, meta=None):
    pred = ta_gan.predict_sr(img)            # synthetic STED + segmentation
    if state["prev_pred"] is not None:
        # Per-pixel disagreement, restricted to the auxiliary-task mask
        mask = ta_gan.task_mask(pred)
        diff = ((pred - state["prev_pred"]) ** 2 * mask).sum() / max(mask.sum(), 1)
        if diff > DIVERGENCE_THRESHOLD:
            cy, cx = ta_gan.peak_change(pred, state["prev_pred"])
            state["rois"].append((cx, cy))
    state["prev_pred"] = pred

def confocal_with_sted_bursts():
    for t in range(N_TIMEPOINTS):
        # Phase 1 -- low-dose confocal scout (pre-event + post-event pair)
        yield MDAEvent(channel={"config": "Confocal", "group": "Modality"},
                       exposure=20, min_start_time=t * dt)
        # Phase 2 -- STED only at TA-GAN-flagged ROIs
        for (x, y) in state["rois"]:
            yield MDAEvent(channel={"config": "STED", "group": "Modality"},
                           exposure=200, x_pos=x, y_pos=y)
        state["rois"].clear()

run_events(core, confocal_with_sted_bursts(), on_frame=ta_gan_trigger)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/multiscale.py` — `multiscale_acquire` already implements N-level hierarchical acquisition; an `interest_fn` that runs TA-GAN inference on the level-0 confocal frame and returns ROIs would slot directly into the existing scoring interface.
- `../../../src/Core/Concepts/Event-driven acquisition` — the trigger here is a model-divergence score rather than a hand-tuned biosensor; the existing event-driven state machine handles the transitions, only the `detect_fn` changes.
- A model-uncertainty trigger module (`src/core/workflows/model_triggered.py`) does not exist yet; this paper motivates one that wraps an arbitrary `predict_sr(img) -> array` callable, maintains a rolling baseline, and emits ROI lists when divergence exceeds a calibrated threshold. Calibration: run TA-GAN on a held-out confocal/STED pair set from the same sample class; pick the threshold from the change-vs-no-change distribution.

The trigger generalises beyond STED: any modality pair where a cheap scout can train a network to predict the expensive ground truth (widefield → SIM, brightfield → fluorescence, low-NA → high-NA) inherits the same paradigm.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — TA-GAN turns survey→rank→zoom into an SR-predictive variant: a confocal scout's synthetic STED prediction selects which ROIs deserve a real STED pass, with model disagreement as the ranking score.
- [[Core/Concepts/Event-driven acquisition]] — model-divergence trigger replaces hand-defined biological cues; the same poll/burst state machine transitions on "TA-GAN prediction changed" instead of "biosensor crossed threshold".
