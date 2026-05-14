---
title: A machine learning approach for online automated optimization of super-resolution optical microscopy
authors: Audrey Durand, Theresa Wiesner, Marc-André Gardner, Louis-Émile Robitaille, Anthony Bilodeau, Christian Gagné, Paul De Koninck, Flavie Lavoie-Cardinal
year: 2018
venue: Nature Communications 9:5247
doi: 10.1038/s41467-018-07668-y
url: https://www.nature.com/articles/s41467-018-07668-y
researched: 2026-04-24
---

## Abstract

Traditional approaches for finding well-performing parameterizations of complex imaging systems, such as super-resolution microscopes rely on an extensive exploration phase over the illumination and acquisition settings, prior to the imaging task. This strategy suffers from several issues: it requires a large amount of parameter configurations to be evaluated, it leads to discrepancies between well-performing parameters in the exploration phase and imaging task, and it results in a waste of time and resources given that optimization and final imaging tasks are conducted separately. Here we show that a fully automated, machine learning-based system can conduct imaging parameter optimization toward a trade-off between several objectives, simultaneously to the imaging task. Its potential is highlighted on various imaging tasks, such as live-cell and multicolor imaging and multimodal optimization. This online optimization routine can be integrated to various imaging systems to increase accessibility, optimize performance and improve overall imaging quality.

## Smart microscopy principle

STED (and other point-scanning super-resolution modalities) exposes a high-dimensional illumination / depletion / dwell-time / pinhole parameter space where "best" depends on the sample, and every evaluation burns the specimen. Durand et al. replace the offline pre-imaging sweep with a **contextual-bandit / Gaussian-process optimiser that treats each real acquisition as one pull of a multi-armed bandit** — the acquisition *is* the exploration. A small number of early frames seed the model; subsequent frames are chosen to balance predicted image quality against predicted photodamage (a Pareto trade-off over multiple objectives), and the acquired frames also answer the biological question. No separate calibration run, no "throwaway" pre-sample.

The transferable idea is broader than STED: **when parameter optimisation and the experiment share the same photon budget, merge the two loops**. The microscope should keep optimising while it acquires, weighting objectives (resolution, SNR, bleaching rate, acquisition time) that the user declares once at the start. This is the active-learning / Bayesian-optimisation flavour of smart microscopy — complementary to content-aware triggering (Mahecic 2022) and closed-loop alignment (Royer 2016).

## Implementation on pymmcore-plus

The skeleton is a bandit / Bayesian-optimiser living outside the acquisition generator; on each `on_frame` callback the observed image yields a multi-objective score, the optimiser updates its posterior, and the next `MDAEvent` picks up the newly suggested parameter vector via `MDAEvent.properties` (for hardware settings) and `MDAEvent.exposure` (for dwell time / exposure proxy).

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
# any BO library works: scikit-optimize, BoTorch, Ax, or a simple GP wrapper
from skopt import Optimizer

PARAM_SPACE = [
    (10.0, 200.0),    # exposure (ms) — stand-in for dwell time on STED
    (1.0,  50.0),     # depletion / excitation laser power, arbitrary units
    (1.0,   8.0),     # gain
]

def score_image(img):
    """Multi-objective score: resolution proxy (Brenner), SNR, bleaching cost."""
    from src.core.workflows.autofocus import focus_metric
    from src.core.analysis.intensity import compute_snr
    resolution = focus_metric(img, method="brenner")    # higher = sharper
    snr        = compute_snr(img)["snr"]                # higher = cleaner
    # photodamage proxy: fraction of saturated pixels + exposure-weighted dose
    dose_cost  = float((img >= img.max() * 0.99).mean())
    # scalarise: user declares weights once
    return 1.0 * resolution + 0.3 * snr - 5.0 * dose_cost

def online_parameter_optimization(core, n_frames=60, channel="STED"):
    opt = Optimizer(PARAM_SPACE, base_estimator="GP", acq_func="EI")
    history = []

    def gen():
        for i in range(n_frames):
            exp_ms, laser_power, gain = opt.ask()
            yield MDAEvent(
                channel={"config": channel},
                exposure=exp_ms,
                properties=[
                    ("STEDLaser", "Power", f"{laser_power:.3f}"),
                    ("Camera",    "Gain",  f"{gain:.2f}"),
                ],
                metadata={"params": [exp_ms, laser_power, gain], "i": i},
            )

    def on_frame(img, event, meta=None):
        params = (event.metadata or {}).get("params")
        s = score_image(img)
        # skopt minimises -> negate our "higher is better" score
        opt.tell(params, -s)
        history.append({"i": event.metadata["i"], "params": params, "score": s})

    results = run_events(core, gen(), on_frame=on_frame)
    return results, history
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` is the dispatcher.
- `../../../src/core/workflows/exposure.py` — currently does **greedy** per-frame exposure updates (`recommend_exposure`, `optimize_exposure`). A Bayesian / bandit variant belongs alongside it as `online_bo(core, param_space, score_fn, n_frames)` — does not exist yet; this paper motivates the addition.
- Multi-objective scalarisation: a small helper `make_pareto_score(weights={"resolution": 1, "snr": 0.3, "dose": -5})` returning a scoring callable would keep user-facing weights in one place. Not yet implemented.

Calibration and safety: bandit / GP optimisers will happily drive a laser to its maximum if the reward function rewards brightness. Constrain the parameter space physically (soft maxima on laser power, minimum exposure) and **always include a dose-cost term in the scalar score** — this paper's core lesson is that multi-objective is not optional when photodamage is on the table.

## Cited by

- [[Core/Strategies/Imaging parameter optimization]] — the canonical reference for online (rather than pre-acquisition) parameter sweeps: merge optimisation with the experiment instead of running two separate loops.
- [[Core/Strategies/Gentle imaging]] — demonstrates that photodamage can be treated as one objective among several in a Bayesian / bandit optimiser rather than as a hard constraint tuned by hand.
