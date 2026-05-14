---
title: Self-driving microscopy detects the onset of protein aggregation and enables intelligent Brillouin imaging
authors: Khalid A. Ibrahim, Camille Cathala, Carlo Bevilacqua, Lely Feletti, Robert Prevedel, Hilal A. Lashuel, Aleksandra Radenovic
year: 2025
venue: Nature Communications 16:6699
doi: 10.1038/s41467-025-60912-0
url: https://www.nature.com/articles/s41467-025-60912-0
researched: 2026-04-25
---

## Abstract

The process of protein aggregation, central to neurodegenerative diseases like Huntington's, is challenging to study due to its unpredictable nature and relatively rapid kinetics. Understanding its biomechanics is crucial for unraveling its role in disease progression and cellular toxicity. Brillouin microscopy offers unique advantages for studying biomechanical properties, yet is limited by slow imaging speed, complicating its use for rapid and dynamic processes like protein aggregation. To overcome these limitations, we developed a self-driving microscope that uses deep learning to predict the onset of aggregation from a single fluorescence image of soluble protein, achieving 91% accuracy. The system triggers optimized multimodal imaging when aggregation is imminent, enabling intelligent Brillouin microscopy of this dynamic biomechanical process. Furthermore, we demonstrate that by detecting mature aggregates in real time using brightfield images and a neural network, Brillouin microscopy can be used to study their biomechanical properties without the need for fluorescence labeling, minimizing phototoxicity and preserving sample health. This autonomous microscopy approach advances the study of aggregation kinetics and biomechanics in living cells, offering a powerful tool for investigating the role of protein misfolding and aggregation in neurodegeneration.

## Smart microscopy principle

Ibrahim et al. apply the modality-switching EDA pattern of [[Papers/Alvelid 2022]] and [[Papers/Stepp 2026]] to a fundamentally different second modality: **Brillouin microscopy** for biomechanics rather than STED for super-resolution. The constraint is the same — Brillouin acquisition is *intrinsically slow* (point-scan over a spectrometer, seconds to minutes per FOV) and would miss the rapid aggregation kinetics if run continuously, and aimless raster scanning wastes the entire experiment if no aggregate has yet formed. The constraint is opposite to STED's photodamage: Brillouin is gentle but *temporally expensive*, so the question becomes "when is it worth spending a Brillouin map?" rather than "when is it safe to spend STED dose?".

The decisive contribution is **predictive triggering**: a Vision Transformer ("AEGON") looks at one fluorescence frame of still-soluble Htt-mEGFP and predicts whether an aggregate is *about to form* in that cell, not whether one has already appeared. This pushes the trigger upstream of the event itself — by the time the slow Brillouin scan completes, the aggregate exists and its mechanical properties can be measured at the moment of onset. A second network on brightfield closes the alternative loop: detect mature aggregates label-free, then run Brillouin without ever exciting fluorescence. Together they generalise event-triggered modality switching from "fast scout, slow capture" to "predictive scout fires the slow capture early enough that the slow capture catches the event itself".

## Implementation on pymmcore-plus

The state machine is the modality-switching variant of [[Papers/Mahecic 2022]]: continuous fluorescence (or brightfield) polling, and on a positive prediction the controller hands off to a configured Brillouin sub-acquisition. Brillouin is not a standard pymmcore-plus channel — it is a separate scanning + spectrometer subsystem — so on a real rig the "burst" is a hardware handoff (often a separate process or device adapter), not a `setConfig` swap. The microscope-control framework still owns the polling loop, the trigger, and the FOV coordinates passed to the Brillouin scan.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

POLL_CHANNEL = {"config": "GFP-soluble", "group": "Channel", "exposure": 50}
PRED_THRESH  = 0.85   # ViT P(aggregation onset)
COOLDOWN     = 60     # poll frames after a Brillouin run before re-arming

def self_driving_brillouin(core, brillouin_run, predictor, max_frames=10000):
    """Predictive modality switching: fluorescence poll -> Brillouin map on trigger.

    predictor(img) -> P(aggregation onset within cell)  in [0, 1]
    brillouin_run(core, x_um, y_um) executes the Brillouin scan + readout.
    """
    state = {"cooldown": 0, "fired_at": None}

    def on_frame(img, event, meta=None):
        if state["cooldown"] > 0 or state["fired_at"] is not None:
            return
        p = float(predictor(img))
        if p > PRED_THRESH:
            x, y = core.getXPosition(), core.getYPosition()
            state["fired_at"] = (x, y)

    def gen():
        for i in range(max_frames):
            if state["fired_at"] is not None:
                x, y = state["fired_at"]
                brillouin_run(core, x, y)        # blocks; slow scan
                state["fired_at"] = None
                state["cooldown"] = COOLDOWN
                continue
            yield MDAEvent(channel=POLL_CHANNEL, index={"t": i})
            if state["cooldown"] > 0:
                state["cooldown"] -= 1

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `src/core/hardware/core.py` — `run_events` is the dispatcher; the predictor lives in the `on_frame` callback, the modality handoff is an out-of-band call that pauses the MDA generator.
- The same `event_driven` state machine planned for [[Papers/Stepp 2026]] covers this case if the burst is parametrised as a callable rather than a channel list. Two-state EDA is the unifying abstraction: rate switching (Mahecic), label-free → fluorescence (Stepp), widefield → STED (Alvelid), fluorescence → Brillouin (Ibrahim) all share one IDLE/POLL/EVENT/BURST/COOLDOWN skeleton with different burst payloads.
- The label-free brightfield path is a strict simplification — the predictor input changes, the rest is identical — so it shares one workflow function with the fluorescence path, just with `POLL_CHANNEL` and `predictor` swapped.

Calibration: the predictive trigger has a non-trivial false-positive cost (a wasted Brillouin scan, minutes long), so `PRED_THRESH` is tuned on a held-out set of pre-aggregation timelapses to maximise lead-time-weighted precision rather than raw classification accuracy. Cooldown matters more than in fluorescence-only EDA because the burst is itself slow — re-firing during a Brillouin scan is hardware-impossible, but re-firing immediately after one would catch the same aggregate twice.

## Cited by

- [[Core/Concepts/Event-driven acquisition]] — modality-switching EDA with a *predictive* trigger (single-frame ViT outputs P(onset) before the event happens) rather than a *detection* trigger. Sits alongside Mahecic 2022, Alvelid 2022, and Stepp 2026 as the four canonical EDA archetypes.
- [[Core/Strategies/Adaptive acquisition]] — example of survey→capture where the capture modality is *temporally* expensive (slow scan) rather than *photonically* expensive (STED bleach), generalising the cost axis the trigger is gating against.
