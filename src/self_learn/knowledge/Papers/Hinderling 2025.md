---
title: Real-time feedback control microscopy for automation of optogenetic targeting
authors: Lucien Hinderling, Alex E. Landolt, Benjamin Grädel, Laurent Dubied, Cédric Zahni, Moritz Kwasny, Dario Bassi, Agne Frismantiene, Talley Lambert, Maciej Dobrzyński, Olivier Pertz
year: 2025
venue: bioRxiv preprint (Pertz Lab, Univ. Bern; v2)
doi: 10.1101/2025.08.17.670729
url: https://www.biorxiv.org/content/10.1101/2025.08.17.670729v2
researched: 2026-04-25
---

## Abstract

FARO (Feedback Adaptive Real-time Optogenetics) is an experimental platform that combines automated image segmentation, tracking, feature extraction, and adaptive hardware control to dynamically adjust optogenetic stimulation based on live cell behavior. By continuously analyzing biosensor signals, FARO updates illumination patterns in real time across biological scales — from maintaining stimulation on specific subcellular regions, to selectively activating single cells in deforming tissue. The framework is fully automated, Python-based, and built on open standards for data management and microscope control. It enables reproducible, systematic, and high-throughput interrogation of spatiotemporal signaling, and is used to study how local signaling events shape cellular behavior, from subcellular dynamics and single-cell migration up to emergent tissue-level processes.

## Smart microscopy principle

FARO is the engineering complement to [[Papers/Passmore 2025]]: where Passmore et al. articulate the *paradigm* (a microscope as a controller of cell behaviour), FARO articulates the *substrate* — a Python-native, Micro-Manager-backed, multi-thread closed-loop pipeline that any optogenetics lab can drop in and re-target across scales. The contribution is **scale-agnosticism**: the same observe → segment → track → score → re-pattern loop runs at the subcellular level (keep a stim spot pinned to a moving organelle/region), the single-cell level (selectively activate one cell in a moving epithelium), and the tissue level (thousands of subcellular regions across hundreds of cells). The user changes the segmentation target and the per-object scoring function; the rest of the loop is unchanged.

Two architectural choices distinguish FARO from earlier closed-loop optogenetic systems. (1) **Decoupled threads:** scheduling (move stage, snap, stim) runs on the main thread; segmentation, tracking, and feature extraction run on background threads. This is what unlocks the "thousands of regions across hundreds of cells" scale — the controller never blocks the acquisition clock waiting for inference. (2) **Interactive pixel classifier in the loop:** rather than committing a pre-trained Cellpose / U-Net before the experiment, FARO uses Convpaint, an interactive classifier that can be (re-)trained on a single sample image *from the current experiment*. This collapses the per-prep training burden that normally gates closed-loop deployment — the segmentation model adapts to the day's labelling, illumination, and cell line in minutes.

The practical implication for anyone building closed-loop optogenetics: target the **right level of abstraction**. The decision loop is not "stim cell *N*" — it is "compute a feature on each segmented region, map features to a desired stimulation pattern, push the pattern to the SLM/DMD." Hold that interface stable and the same code retargets from focal-adhesion patterning to single-cell migration steering to tissue-flow choreography, just by swapping the segmenter and the feature→pattern mapping.

## Implementation on pymmcore-plus

FARO ships on Pycro-Manager + Micro-Manager, but the architecture is identical to a `run_events` generator with a heavy-weight `on_frame` callback that delegates to background workers. Each acquired frame is handed to a segmentation worker (Convpaint pixel classifier — or Cellpose CPSAM, or any drop-in segmenter); the resulting label map plus tracked-state history feeds a feature-scoring step; scores map to per-region stim parameters; an SLM image is rebuilt and queued for the next event. Any DMD exposed as a `genericSLM` device in Micro-Manager works (Andor Mosaic 3, Mightex Polygon 1000 are the reference targets); on pymmcore-plus the same hardware is reachable through `core.setSLMImage`.

```python
from useq import MDAEvent, SLMImage
from src.core.hardware.core import run_events

CHANNEL = "biosensor"
MAX_FRAMES = 500


def faro_loop(core, segment_fn, score_fn, build_pattern_fn,
              tracker, max_frames=MAX_FRAMES, channel=CHANNEL):
    """FARO-style closed loop: segment, track, score, re-pattern, stim.

    `segment_fn(img) -> labels` is interactive-classifier or Cellpose.
    `score_fn(labels, history) -> dict[label_id, stim_level]` is the
    feature-extraction + per-cell decision. `build_pattern_fn(scores,
    labels) -> np.uint8` rasterises an SLM mask. The tracker carries
    per-cell state across frames so `score_fn` sees history, not just
    the current frame.

    Generator-+-on_frame skeleton (cf. [[Core/Approach/OADA loop]]).
    """
    state = {"frame_idx": 0, "next_mask": None, "done": False}

    def on_frame(img, event):
        labels = segment_fn(img)                       # background-thread offload OK
        tracked = tracker.update(labels, state["frame_idx"])
        scores = score_fn(tracked, history=tracker.history)
        state["next_mask"] = build_pattern_fn(scores, labels) if scores else None
        state["frame_idx"] += 1

    def gen():
        for _ in range(max_frames):
            if state["done"]:
                return
            evt_kwargs = {"channel": {"config": channel}}
            if state["next_mask"] is not None:
                evt_kwargs["slm_image"] = SLMImage(data=state["next_mask"], device="SLM")
            yield MDAEvent(**evt_kwargs)

    return run_events(core, gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` is the dispatcher; `make_slm_circle` / `apply_slm` rasterise per-cell stim spots into an SLM frame. FARO's "any DMD that speaks `genericSLM`" maps onto pymmcore-proxy's transparent SLM device passthrough.
- `[[Core/Approach/OADA loop]]` — the generator + `on_frame` scaffold the snippet above instantiates. `on_frame` segments + scores the latest image and stages the next SLM mask in `state`; the generator emits an `MDAEvent` carrying that mask before the next snap. This collapses FARO's "scheduler thread + analysis thread" architecture onto a single-process generator for simulation work; the production architecture promotes the analysis branch onto a worker thread.
- `../../../src/core/workflows/optogenetics.py` — already exposes `target_cells` and stim-pattern helpers; the FARO contribution to absorb here is the **multi-region, history-aware scoring step** (a function that scores every tracked label and returns a dict, rather than picking one target). A clean FARO-style API would be `run_optogenetic_loop(core, segment_fn, score_fn, build_pattern_fn, tracker, max_frames)`, separating the reusable closed-loop scaffold from the four sample-specific pieces (segmentation, scoring, pattern synthesis, identity tracking).
- `../../../src/core/workflows/stage_tracking.py` — supplies the identity tracker; FARO's per-cell history requirement means the loop needs a Hungarian or centroid tracker that survives label flicker between frames.

Calibration on a real prep: (i) train (or re-train) the segmenter on **today's** labelling — Convpaint's interactive classifier is the lesson here, not the specific tool; an out-of-the-box Cellpose model may underperform a 30-second per-experiment retrain. (ii) Budget the inference latency: with background threads, segmentation and scoring can run concurrently with the next acquisition, but the *control interval* (how often the SLM mask is refreshed) must exceed worst-case end-to-end latency or stim drifts off-target. (iii) Validate the pattern → stim → response coupling with a non-feedback open-loop sweep first; closed-loop only adds value once the dose-response is in the linear regime. (iv) Identity tracking is the silent failure mode — when a cell flips identity, its stim history flips with it; cross-check tracker assignments against the SLM mask before trusting closed-loop measurements of per-cell trajectories.

The same scaffolding generalises across the three scales the paper demonstrates: subcellular pinning (segment on a sub-cell ROI, score = "stay locked on this region"), single-cell selective activation (segment whole cells, score = "activate only labels matching some criterion"), tissue-flow steering (segment all cells, score = a population-level objective like "drive a wavefront across the monolayer"). Identical loop, different segmenter + scorer.

## Cited by

- [[Core/Strategies/SLM optogenetics]] — FARO is the open-source production reference: any DMD/SLM behind a `genericSLM` Micro-Manager device, segmentation in the loop, per-cell scoring → mask synthesis → stim. Where [[Papers/Passmore 2025]] establishes the paradigm and [[Papers/Lugagne 2024]] establishes the model-based variant, FARO establishes the **deployable substrate** that scales the loop across subcellular / single-cell / tissue regimes.
- [[Core/Strategies/Feedback control]] — instantiates the observe → compute → modify → snap loop with explicit decoupled threading (scheduler vs analysis) so the control interval is bounded by the worst-of-the-two, not the sum.
- [[Core/Strategies/Simultaneous SLM targeting]] — multi-region, multi-cell stim with per-target parameters is the explicit FARO use case (thousands of subcellular regions across hundreds of cells); this paper is the canonical citation for "many SLM targets simultaneously, each with its own dose."
- [[Core/Strategies/Closed-loop state device]] — the SLM pattern register is the state device; FARO is the scale-out application showing that the same primitive supports subcellular, single-cell, and tissue-level control.
- [[Papers/Fox 2022]] — MicroMator factors reactive microscopy into composable event-condition-action rules; FARO inherits the Python-native ECA mindset but specialises and hardens it for the optogenetic-targeting use case (segmentation + tracking + per-cell pattern synthesis baked in).
- [[Papers/Passmore 2025]] — paradigmatic sibling: Passmore proves a microscope can be an outcome controller of cell behaviour; Hinderling et al. supply the open Python+Micro-Manager pipeline that any lab can adopt for that role across scales.
- [[Papers/Lugagne 2024]] — predictive sibling: Lugagne uses a learned forward model in a receding-horizon controller to *trajectory-track* per-cell gene expression; FARO targets *spatial* per-cell objectives (where, how much, for how long to stim) on the same closed-loop substrate. The two are composable — a FARO-style segmentation + tracking layer feeding a Lugagne-style MPC scorer is a natural next step.
