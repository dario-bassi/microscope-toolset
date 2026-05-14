---
title: "SmartEM: machine learning-guided electron microscopy"
authors: Yaron Meirovitch, Ishaan Singh Chandok, Core Francisco Park, Pavel Potocek, Lu Mi, Shashata Sawmya, Yicong Li, Thomas L. Athey, Vladislav Susoy, Neha Karlupia, Yuelong Wu, Daniel R. Berger, Richard Schalek, Caitlyn A. Bishop, Daniel Xenes, Hannah Martinez, Jordan Matelsky, Brock A. Wester, Hanspeter Pfister, Remco Schoenmakers, Maurice Peemen, Jeff W. Lichtman, Aravinthan D. T. Samuel, Nir Shavit
year: 2026
venue: Nature Methods 23(1):193-204
doi: 10.1038/s41592-025-02929-3
url: https://www.nature.com/articles/s41592-025-02929-3
researched: 2026-04-25
---

## Abstract

Connectomics provides nanometer-resolution, synapse-level maps of neural circuits to understand brain activity and behavior. However, few researchers have access to the high-throughput electron microscopes necessary to generate enough data for whole-brain or even whole-circuit reconstruction. To date, machine learning methods have been used after the collection of images by electron microscopy (EM) to accelerate and improve neuronal segmentation, synapse reconstruction and other data analysis. With the continual computational improvements in processing EM images, acquiring EM images will become the rate-limiting step in automated connectomics. Here, in order to speed up EM imaging, we integrate machine learning into real-time image acquisition in a single-beam scanning electron microscope. This SmartEM approach allows an electron microscope to perform data-aware imaging of specimens. SmartEM saves time by allocating the proper imaging time for each region of interest — first scanning all pixels rapidly and then rescanning more slowly only the small subareas where a higher quality signal is required. We demonstrate that SmartEM achieves up to an ~7-fold acceleration of image acquisition time for connectomic samples using a commercial single-beam SEM in samples from nematodes, mice and human brain. We apply this fast imaging method to reconstruct a portion of mouse cerebral cortex with an accuracy comparable to traditional electron microscopy.

## Smart microscopy principle

SmartEM is the survey-then-zoom paradigm transposed to single-beam SEM connectomics, with the survey and zoom living on the *same modality* and differing only in **per-pixel dwell time**. The microscope first rasters the entire tile at a fast, noisy dwell (the "scout"); a content-aware quality-prediction network scores each subregion for whether the fast scan will support downstream segmentation; flagged regions are then rescanned at a slow, low-noise dwell. Total beam-time is dominated by the fast pass plus the small fraction of pixels that earned a slow rescan, instead of slow-scanning everything — the paper measures up to ~7× acceleration on connectomic samples (nematode, mouse, human brain) and reconstructs mouse cortex at parity with a slow-scan baseline.

Three things make this distinct from the optical survey-then-zoom cluster ([[Papers/Conrad 2011]], [[Papers/Shi 2024]], [[Papers/Kandel 2023]]). First, **the cost being spent is electron dose × dwell time, not photons** — the binding constraint is sample charging / beam damage and total scan hours, not phototoxicity, but the optimisation shape is identical. Second, **the score is downstream-task-conditioned**: the quality predictor is trained against what segmentation networks need (membrane continuity, synaptic vesicle clarity), not against generic image fidelity. Regions that look visually ugly but reconstruct fine at fast dwell are *not* rescanned. Third, **scout and zoom are the same modality at different speeds** — closer to [[Papers/Ye 2025]] (rescan low-confidence pixels) than to Shi 2024 (modality switch) or Kandel 2023 (sparse sampling). SmartEM occupies the dwell-time axis of the smart-acquisition design space.

The generalisable lesson: when the cost of acquisition scales smoothly with a single knob (dwell time, exposure, laser power, line averages) rather than via discrete modality switches, the right structure is a fast pass plus a content-aware *rescan* — and the score should be conditioned on the *downstream task* that consumes the data, not on a task-agnostic fidelity metric. The further you push the fast-pass settings, the more critical it is that the predictor knows what a downstream segmenter or detector actually needs.

## Implementation on pymmcore-plus

pymmcore-plus does not directly drive SEMs, but the SmartEM control flow maps cleanly onto its `MDAEvent` generator + `on_frame` substrate for any microscope that exposes per-event exposure / dwell parameters. Concretely: a fast-exposure tile pass populates a quality map; a generator emits slow-exposure rescan events only for tiles below threshold; both passes route through the same dispatcher.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

FAST_EXPOSURE = 1.0    # ms — analogue of fast pixel dwell
SLOW_EXPOSURE = 10.0   # ms — analogue of slow pixel dwell
QUALITY_THRESH = 0.7   # rescan if predicted task-quality < threshold

def smart_rescan(core, grid_xy, quality_predictor, channel="BF"):
    """SmartEM-style two-pass acquisition: fast scout + content-aware slow rescan.

    quality_predictor(image) -> p in [0, 1]: predicted downstream-task quality
    of the fast image. Trained against e.g. segmentation-network agreement.
    """
    fast_imgs = {}     # (x, y) -> fast image
    rescan_list = []   # (x, y) tiles flagged for slow rescan

    def on_fast(img, event, meta=None):
        key = (event.x_pos, event.y_pos)
        fast_imgs[key] = img
        if float(quality_predictor(img)) < QUALITY_THRESH:
            rescan_list.append(key)

    def fast_gen():
        for (x, y) in grid_xy:
            yield MDAEvent(x_pos=x, y_pos=y,
                           channel={"config": channel},
                           exposure=FAST_EXPOSURE)

    run_events(core, fast_gen(), on_frame=on_fast)

    def slow_gen():
        for (x, y) in rescan_list:
            yield MDAEvent(x_pos=x, y_pos=y,
                           channel={"config": channel},
                           exposure=SLOW_EXPOSURE)

    return run_events(core, slow_gen())
```

Production hooks in `src/core/`:

- `src/core/hardware/core.py` — `run_events` is the dispatcher for both passes.
- `src/core/workflows/scanning.py` — `grid_positions` / `scan_events` produce the fast tile sweep; SmartEM adds a *second* generator over the flagged subset rather than visiting the full grid twice.
- `src/core/workflows/multiscale.py` — `multiscale_acquire` already implements an N-level "score → rank → revisit at higher quality" loop; SmartEM is the special case where "higher quality" is longer dwell at the same magnification rather than a magnification switch. A dedicated `dwell_rescan.py` workflow with a pluggable downstream-task quality predictor is a clean candidate for future work, sibling to the FAST sparse-scan and Ye 2025 uncertainty-rescan modules.
- The quality predictor is the substantive contribution. SmartEM trains it from paired fast/slow image stacks where the slow image is treated as ground truth and the fast image is annotated with where downstream segmentation diverges. On a real prep, hold back a fully slow-scanned reference tile, run the segmenter on both, and learn `fast_image -> task_quality` so the threshold is calibrated against actual reconstruction error, not generic SNR.

Calibration on a real microscope: the speedup depends entirely on the rescan fraction. Sweep `QUALITY_THRESH` against a held-out tile and pick the operating point where the segmentation-error curve plateaus — too aggressive a threshold rescans everything (no speedup); too lax loses connectomic continuity. Charge / drift between passes is a real concern on EM (less so on light); on light microscopy the analogous concern is photobleaching of fluorophores during the fast pass — for fluorescence, the fast pass should always be the cheaper modality.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — SmartEM extends the survey-then-zoom paradigm to single-modality dwell-time switching: scout = fast scan, zoom = slow rescan, score = downstream-task-conditioned quality predictor. Same control-loop shape as [[Papers/Conrad 2011]] / [[Papers/Shi 2024]] / [[Papers/Kandel 2023]] but with EM-specific cost (dwell × dose) and connectomic-task-specific score.
- [[Core/Strategies/Multi-scale morphometry]] — content-aware allocation of acquisition cost between a cheap pass and an expensive pass; SmartEM is the dwell-time analogue of magnification switching.
- [[Papers/Ye 2025]] — closest sibling: both papers re-acquire only low-confidence regions of an already-scanned image, differing in score (Ye uses calibrated reconstruction uncertainty from a denoising network; SmartEM uses task-conditioned quality prediction trained against downstream segmentation).
- [[Papers/Kandel 2023]] — adjacent: FAST decides *whether to acquire* a point at all (sparse sampling); SmartEM acquires every point at fast dwell and decides *whether to re-acquire at slow dwell*. Different axes of the same "where to spend the budget" question.
- [[Papers/Shi 2024]] — adjacent: SmartLLSM is modality-switching survey→zoom (widefield → LLSM), SmartEM is dwell-switching survey→rescan (fast SEM → slow SEM). Same paradigm, different cost knob.
