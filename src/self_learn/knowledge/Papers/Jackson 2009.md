---
title: "Intelligent acquisition and learning of fluorescence microscope data models"
authors: Charles Jackson, Robert F. Murphy, Jelena Kovačević
year: 2009
venue: IEEE Transactions on Image Processing 18(9):2071-2084
doi: 10.1109/TIP.2009.2024580
url: https://pubmed.ncbi.nlm.nih.gov/19502128/
researched: 2026-04-24
---

## Abstract

We propose a mathematical framework and algorithms both to build accurate models of fluorescence microscope time series, as well as to design intelligent acquisition systems based on these models. Model building allows the information contained in the 2-D and 3-D time series to be presented in a more useful and concise form than the raw image data. This is particularly relevant as the trend in biology tends more and more towards high-throughput applications, and the resulting increase in the amount of acquired image data makes visual inspection impractical. The intelligent acquisition system uses an active learning approach to choose the acquisition regions that let us build our model most efficiently, resulting in a shorter acquisition time, as well as a reduction of the amount of photobleaching and phototoxicity incurred during acquisition. We validate our methodology by modeling object motion within a cell. For intelligent acquisition, we propose a set of algorithms to evaluate the information contained in a given acquisition region, as well as the costs associated with acquiring this region in terms of the resulting photobleaching and phototoxicity and the amount of time taken for acquisition. We use these algorithms to determine an acquisition strategy: where and when to acquire, as well as when to stop acquiring.  Results, both on synthetic as well as real data, demonstrate accurate model building and large efficiency gains during acquisition.

## Smart microscopy principle

Jackson, Murphy and Kovačević formalise microscope acquisition as an active-learning loop **over a model being built online**: the microscope maintains a parametric model of the scene (here, object-motion dynamics inside a cell), and at each step chooses the next acquisition region to maximise an explicit utility — information-gain about the model minus an acquisition cost that puts photobleaching, phototoxicity and time on equal footing. The loop also answers **when to stop** acquiring, because diminishing model-information return per photon is directly measurable. This is the pre-deep-learning formulation of the now-standard "uncertainty-driven acquisition" pattern — the same shape as Ye 2025 (conformal uncertainty rescan) and Kandel 2023 (Expected Reduction in Distortion), but framed around a hand-specified generative motion model rather than a neural network's output.

## Implementation on pymmcore-plus

The loop is an adaptive MDA generator: each iteration fits/updates the model from everything acquired so far, scores every candidate region by predicted information-gain and acquisition cost, yields one `MDAEvent` for the argmax region, and terminates the generator when no candidate's net utility is positive.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

# Candidate regions: e.g. a coarse grid or a list of tracked objects
candidates = [{"x": wx, "y": wy, "id": i} for i, (wx, wy) in enumerate(grid)]

state = {
    "model": init_motion_model(),        # parametric model being fitted online
    "history": [],                       # (region_id, img, t) per acquired event
    "photon_budget_used": 0.0,
}

def info_gain(region, state):
    """Predicted reduction in model posterior variance if we image this region."""
    return posterior_variance(state["model"]) \
           - expected_posterior_variance(state["model"], region)

def cost(region, state, exposure_ms):
    """Photobleaching + phototoxicity + time cost for this region."""
    return (BLEACH_PER_MS * exposure_ms
            + PHOTOTOX_PER_MS * exposure_ms
            + TIME_PER_EVENT)

def on_frame(img, event, meta=None):
    region_id = event.metadata["region_id"]
    state["history"].append((region_id, img, meta["runner_t0"]))
    state["model"] = update_model(state["model"], img, region_id)

def intelligent_gen():
    while True:
        # Score every candidate by utility = info_gain - cost
        scored = [(r, info_gain(r, state) - cost(r, state, EXPOSURE_MS))
                  for r in candidates]
        best_region, best_utility = max(scored, key=lambda s: s[1])
        if best_utility <= 0:          # when to stop: no candidate pays for itself
            return
        yield MDAEvent(
            x_pos=best_region["x"], y_pos=best_region["y"],
            channel={"config": "GFP", "exposure": EXPOSURE_MS},
            metadata={"region_id": best_region["id"]},
        )

run_events(core, intelligent_gen(), on_frame=on_frame)
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/adaptive.py` — `rank_by_feature` and `zoom_and_measure` already rank-and-visit; they lack the model-update and cost-aware stopping criterion.
- `../../../src/core/workflows/reasoning.py` — `run_experiment_cycle` / `reasoning_loop` are the closest existing scaffold for "hypothesis → acquire → update → decide". Extending this module to carry an explicit posterior and an information-gain scorer would produce a Jackson-style active-learning workflow.
- No dedicated active-learning acquisition module exists yet; this paper motivates `src/core/workflows/active_learning.py` whose distinguishing feature over plain survey→rank→zoom is (a) an online-updated model and (b) a cost-aware stopping rule that halts acquisition when predicted information-gain no longer outweighs photobleaching/phototoxicity/time.

Calibration on a real prep: characterise per-channel bleach-rate and tolerable phototoxicity dose before the run (so the `cost` function is in real units), and pick initial model priors wide enough that early regions don't trivially dominate the utility ranking.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — Jackson 2009 is the pre-DL canonical formulation of active-learning-driven acquisition: utility = information-gain about an online-updated model minus a photobleaching/phototoxicity/time cost, with an explicit "when to stop" criterion. Slots in before Conrad 2011 on the adaptive-acquisition timeline.
