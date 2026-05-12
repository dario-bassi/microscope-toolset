---
title: "On-the-fly Raman image microscopy by reinforcement machine learning"
authors: Tamiki Komatsuzaki, Koji Tabata, Hiroyuki Kawagoe, J. Nicholas Taylor, Kentaro Mochizuki, Toshiki Kubo, Jean-Emmanuel Clement, Yasuaki Kumamoto, Yoshinori Harada, Atsuyoshi Nakamura, Katsumasa Fujita
year: 2022
venue: "Proc. SPIE PC12144 (Biomedical Spectroscopy, Microscopy, and Imaging II), paper PC121440B"
doi: 10.1117/12.2622529
url: https://doi.org/10.1117/12.2622529
researched: 2026-04-25
related: "Full peer-reviewed journal version: Tabata, Kawagoe, Taylor, Mochizuki, Kubo, Clement, Kumamoto, Harada, Nakamura, Fujita & Komatsuzaki, 'On-the-fly Raman microscopy guaranteeing the accuracy of discrimination', PNAS 121(12):e2304866121 (2024), doi:10.1073/pnas.2304866121."
---

## Abstract

"We present our recent study combined multi-armed Bandits algorithm in reinforcement learning with spontaneous Raman microscope for the acceleration of the measurements by designing and generating optimal illumination pattern 'on the fly' during the measurements while keeping the accuracy of diagnosis. We present our simulation and experimental studies using Raman images in the diagnosis of follicular thyroid carcinoma and non-alcoholic fatty liver disease, and show that this protocol can accelerate more than a few tens times in speedy and accurate diagnoses faster than line-scanning Raman microscope that requires the full detailed scanning over all pixels. The on-the-fly Raman image microscopy designs to accelerate measurements by combining one of reinforcement machine learning techniques, bandit algorithm utilized in the Monte Carlo tree search in alpha-GO, and a programmable illumination system. Given a descriptor based on Raman signals to quantify the likelihood of the predefined quantity to be evaluated, e.g., the degree of cancers, the on-the-fly Raman image microscopy evaluates the upper and lower confidence bounds in addition to the sample average of that quantity based on finite point/line illuminations, and then the bandit algorithm feedbacks the desired illumination pattern to accelerate the detection of the anomaly, during the measurement to the microscope."

The peer-reviewed journal follow-up (Tabata et al., PNAS 2024) demonstrates the method on follicular-thyroid-carcinoma vs normal-epithelial cells and on polystyrene / PMMA bead mixtures, reporting 3,333–31,683× fewer illuminations than full raster scanning and 104–4,350× fewer than fixed point illumination at matched discrimination accuracy.

## Smart microscopy principle

Spontaneous Raman microscopy is photon-starved — scattering cross-sections are tiny, and a full raster scan that visits every pixel can take hours per FOV. Komatsuzaki et al. argue that for **discrimination** tasks (is this cell cancerous? is this region steatotic?) the microscope does not need a full image; it needs just enough Raman counts at just the right spatial locations to push a class-likelihood descriptor across a confidence threshold. They cast the choice of where to illuminate next as a **multi-armed bandit problem** — each candidate illumination pattern (point, line, region) is an arm, the reward is information about the per-grid-cell anomaly score, and a UCB-style policy (the same family used inside AlphaGo's MCTS) feeds the next pattern back to a programmable illumination system *while the measurement is running*. Acquisition stops when the upper / lower confidence bounds on the discrimination descriptor cross the decision threshold — not when a fixed pixel count is reached.

The transferable idea sits on the active-learning / measurement-budgeting axis of smart microscopy: **let the discrimination question, not the pixel grid, decide what gets photons.** This is closely related to Durand 2018 (online bandit over STED *parameters*) but pushes the bandit one level outwards — the arms here index *spatial* sampling decisions, not illumination *intensity / dwell* knobs. It is also a sister of Ye 2025's uncertainty-driven rescan and Kandel 2023's FAST: all three couple a per-pixel "do I know enough yet?" signal back into the scanner. The novelty in Komatsuzaki 2022 is the explicit recognition that bandit theory must be adapted when the discriminator itself has finite accuracy below 100% — the conventional "infinite samples → certainty" assumption breaks down, and the algorithm must propagate finite-sample-size uncertainty in the descriptor through to the stopping rule.

## Implementation on pymmcore-plus

The control loop is a generator that, on each iteration, asks a bandit policy which sub-region to illuminate next, yields an `MDAEvent` parameterised on the chosen ROI / illumination-pattern config, and on `on_frame` updates per-grid-cell descriptor statistics. Acquisition terminates when the bandit's confidence bounds on the global discriminator cross the decision threshold or when a budget cap is hit.

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
import numpy as np

# Partition the FOV into a grid of K candidate illumination cells (the "arms").
# Each cell knows: visit count n_k, accumulated descriptor sum s_k, and a
# UCB1-style upper/lower confidence radius around its mean s_k / n_k.
class RamanBandit:
    def __init__(self, n_cells, c=2.0):
        self.n = np.zeros(n_cells, dtype=int)
        self.s = np.zeros(n_cells, dtype=float)
        self.c = c

    def pick(self, t):
        # UCB1 with a small-count exploration bias; never-visited arms first.
        unvisited = np.where(self.n == 0)[0]
        if len(unvisited):
            return int(unvisited[0])
        mean = self.s / self.n
        ucb = mean + self.c * np.sqrt(np.log(t + 1) / self.n)
        return int(np.argmax(ucb))

    def update(self, k, descriptor):
        self.n[k] += 1
        self.s[k] += descriptor

    def decided(self, threshold):
        # Stop once the global mean descriptor is statistically above / below
        # the discrimination threshold across the cells we've actually visited.
        seen = self.n > 0
        if seen.sum() < 5:
            return False
        means = (self.s[seen] / self.n[seen])
        ci = self.c * np.sqrt(np.log(self.n.sum()) / self.n[seen])
        return (means - ci > threshold).any() or (means + ci < threshold).all()


def on_the_fly_raman(core, cells_xy, descriptor_fn, threshold, budget=200):
    bandit = RamanBandit(n_cells=len(cells_xy))

    def gen():
        for t in range(budget):
            if bandit.decided(threshold):
                return
            k = bandit.pick(t)
            x, y = cells_xy[k]
            yield MDAEvent(
                channel={"config": "Raman"},
                x_pos=x, y_pos=y,
                metadata={"arm": k, "t": t},
            )

    log = []

    def on_frame(img, event, meta=None):
        d = descriptor_fn(img)              # e.g. CARS/Raman class-likelihood ratio
        bandit.update(event.metadata["arm"], d)
        log.append({"t": event.metadata["t"], "arm": event.metadata["arm"], "d": d})

    run_events(core, gen(), on_frame=on_frame)
    return bandit, log
```

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events` already accepts generator-yielded `MDAEvent`s with arbitrary `x_pos`/`y_pos`, so spatial-bandit arm selection requires no engine changes.
- `../../../src/core/workflows/scanning.py` — `scan_events` provides the deterministic-grid baseline this paper accelerates against; an `adaptive_scan_events(bandit, descriptor_fn, threshold)` companion belongs alongside it. Not yet implemented; this paper is its motivation.
- `../../../src/core/workflows/multiscale.py` — `multiscale_acquire`'s ROI-selection step currently uses heuristic feature scores; a bandit-driven ROI selector with explicit confidence-bound-based stopping would slot in there.
- A descriptor library (`descriptor_fn(img) -> float`) for spectral-discrimination tasks lives most naturally next to `analysis/intensity.py`. The discrimination threshold and the descriptor's finite accuracy must be calibrated on labelled training data before deployment — the paper's central caveat is that bandit guarantees collapse if the descriptor itself is biased.

Sample-side caveats that transfer beyond Raman: the bandit assumes the per-cell measurement is *informative on its own scale* — for fluorescence this works only if cells are bright enough that one short exposure gives a meaningful descriptor value. Always cap `budget` (max total illuminations / total dose) so a degenerate sample where no arm crosses threshold cannot bleach the specimen indefinitely.

## Cited by

- [[Core/Strategies/Imaging parameter optimization]] — extends the online-bandit lineage from illumination *parameters* (Durand 2018) to illumination *spatial patterns*: the bandit chooses where to sample, not how brightly, and the stopping rule is a confidence bound on a discrimination descriptor, not a Pareto trade-off.
- [[Core/Strategies/Adaptive acquisition]] — discrimination-driven sampling is a budget-aware variant of survey→rank→zoom: the "rank" step is a UCB on a class-likelihood descriptor, and "stop when decided" replaces "stop when the grid is filled".
- [[Core/Strategies/Gentle imaging]] — for photon-starved modalities (Raman, two-photon at depth, low-dose EM) replacing full raster scans with bandit-driven sparse sampling cuts total dose by orders of magnitude while preserving the *decision*, not the full image.
