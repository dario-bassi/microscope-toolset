---
title: "Data-driven microscopy allows for automated context-specific acquisition of high-fidelity image data"
authors: Oscar André, Johannes Kumra Ahnlide, Nils Norlin, Vinay Swaminathan, Pontus Nordenfelt
year: 2023
venue: Cell Reports Methods 3(3):100419
doi: 10.1016/j.crmeth.2023.100419
url: https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(23)00030-9
researched: 2026-04-24
---

## Abstract

Light microscopy is a powerful single-cell technique that allows for quantitative spatial information at subcellular resolution. However, unlike flow cytometry and single-cell sequencing techniques, microscopy has issues achieving high-quality population-wide sample characterization while maintaining high resolution. Here, we present a general framework, data-driven microscopy (DDM) that uses real-time population-wide object characterization to enable data-driven high-fidelity imaging of relevant phenotypes based on the population context. DDM combines data-independent and data-dependent steps to synergistically enhance data acquired using different imaging modalities. As a proof of concept, we develop and apply DDM with plugins for improved high-content screening and live adaptive microscopy for cell migration and infection studies that capture events of interest, rare or common, with high precision and resolution. We propose that DDM can reduce human bias, increase reproducibility, and place single-cell characteristics in the context of the sample population when interpreting microscopy data, leading to an increase in overall data fidelity.

## Smart microscopy principle

Data-driven microscopy (DDM) decomposes an acquisition into two phases with feedback between them. Phase 1 (data-independent) sweeps the sample at low, uniform dose and builds a population-wide characterisation of every object — position, size, phenotype score. Phase 2 (data-dependent) then re-visits only objects that the population model flags as relevant, at acquisition settings (magnification, channels, exposure, frame rate) matched to the question being asked. The novelty over single-event triggers is that the acquisition decision is conditioned on the **whole population's distribution**, not just the current FOV: a cell is "interesting" relative to its peers, not relative to a fixed threshold.

For long live-cell experiments, this rebalances the photon budget in exactly the way "AI-guided exposure and frame-rate adjustment" needs. Quiescent or uninformative objects are polled cheaply; objects near a phenotypic transition are imaged at higher rate, higher mag, and whatever exposure is needed to resolve the event. The population-context step is what lets the controller know *which* setting to pick for *which* object, rather than applying one adaptive rule uniformly.

Worth noting: DDM is a framework, not a single algorithm. The population characterisation can be anything from a classical feature-cluster to a CNN — the interface is "object → phenotype score", and the acquisition policy takes the score distribution as input.

## Implementation on pymmcore-plus

The two-phase structure maps to two `MDASequence`s joined by an on-frame callback that populates a shared object table. Phase 1 is a plain multi-position survey. Phase 2 is an adaptive generator that reads the phenotype-ranked table and yields per-object events with per-object acquisition parameters.

```python
from useq import MDAEvent, MDASequence
from src.core.hardware.core import run_events

# Phase 1: survey the whole sample at low dose, uniform settings.
survey_seq = MDASequence(
    stage_positions=[{"x": x, "y": y} for (x, y) in grid],
    channels=[{"config": "BF"}, {"config": "nucleus", "exposure": 20}],
)

population = []  # list of {x, y, area, nuc_mean, ...}
def survey_on_frame(img, event, meta=None):
    for cell in detect_cells(img):
        population.append({**cell, "event_pos": (event.x_pos, event.y_pos)})

run_events(core, list(survey_seq), on_frame=survey_on_frame)

# Score each object relative to the population distribution.
scores = score_against_population(population)  # e.g. z-score of nuc_mean
targets = [obj for obj, s in zip(population, scores) if s > 2.0]

# Phase 2: re-visit each target with per-object acquisition parameters.
def adaptive_gen():
    for i, obj in enumerate(targets):
        # High-information objects: fast frame rate, long exposure.
        # Lower-priority objects: slow polling, short exposure.
        rate_s, exp_ms = pick_rate_and_exposure(obj)
        for t in range(20):
            yield MDAEvent(
                x_pos=obj["x"], y_pos=obj["y"],
                channel={"config": "GFP", "exposure": exp_ms},
                min_start_time=i * 40.0 + t * rate_s,
                index={"p": i, "t": t},
            )

run_events(core, adaptive_gen())
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/scanning.py` — `scan_and_detect` already produces the Phase-1 object table; DDM extends it with a population-scoring step.
- `../../../src/core/workflows/adaptive.py` — `rank_by_feature` and `survey_cells` do single-feature ranking; DDM generalises this to multi-feature phenotype scores.
- A DDM workflow module (`src/core/workflows/data_driven.py`) does not yet exist; this paper motivates one. The missing piece is a policy function mapping (per-object score, remaining photon budget) to (exposure, frame_rate, channel list) so that exposure and frame rate are tuned **per object** rather than globally.

Calibration on a real prep: the population-scoring step needs a reference distribution. Spend the first survey sweep doing nothing but collecting features; only from the second sweep onward does the policy start adjusting exposure and frame rate. Without that baseline, the "interesting relative to peers" contract is empty.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — survey→rank→zoom is the canonical DDM pattern; André et al. generalise the ranking step to population-wide phenotype scoring.
