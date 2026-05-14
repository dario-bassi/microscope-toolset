---
title: "The rise of data-driven microscopy powered by machine learning"
authors: Leonor Morgado, Estibaliz Gómez-de-Mariscal, Hannah S. Heil, Ricardo Henriques
year: 2024
venue: Journal of Microscopy 295(2):85-92
doi: 10.1111/jmi.13282
url: https://onlinelibrary.wiley.com/doi/10.1111/jmi.13282
researched: 2026-04-24
---

## Abstract

Optical microscopy is an indispensable tool in life sciences research, but conventional techniques require compromises between imaging parameters like speed, resolution, field of view and phototoxicity. To overcome these limitations, data-driven microscopes incorporate feedback loops between data acquisition and analysis. This review overviews how machine learning enables automated image analysis to optimise microscopy in real time. We first introduce key data-driven microscopy concepts and machine learning methods relevant to microscopy image analysis. Subsequently, we highlight pioneering works and recent advances in integrating machine learning into microscopy acquisition workflows, including optimising illumination, switching modalities and acquisition rates, and triggering targeted experiments. We then discuss the remaining challenges and future outlook. Overall, intelligent microscopes that can sense, analyse and adapt promise to transform optical imaging by opening new experimental possibilities.

## Smart microscopy principle

Morgado et al. is the field's review-of-record on data-driven microscopy: it lays out the conceptual scaffolding under which event-driven, content-aware, modality-switching, and rate-switching acquisition all sit as the same paradigm — a feedback loop where a machine-learning analyser sits in line with the acquisition controller. The review's contribution beyond André 2023 (which proposed DDM as a framework) is taxonomic: it organises the field by what the loop *adapts* — illumination, acquisition rate, modality, or downstream-experiment triggering — and ties each axis to the ML method best suited to it (CNN segmentation for triggering, regression for parameter selection, RL for online optimisation).

The candidate hint that motivated this entry — "data-driven vs task-driven microscopy distinction: low-mag phenotype score vs research-question-conditioned score" — is one framing the review draws out. Data-driven microscopy scores objects against the population (André's definition); task-driven microscopy scores objects against the experimenter's research question (Yu 2024 Comms Biol's pathology-foundation-model framing). Morgado et al. position both as instances of the same closed-loop architecture, differing only in what supplies the score. For an agent deciding "what to image", that distinction names the choice: rank by population context (find outliers), or rank by question relevance (find what matches the brief).

The review is the right anchor when a knowledge note needs a citation for "the field exists and has a name" rather than a single algorithm.

## Implementation on pymmcore-plus

A review doesn't have its own algorithm to implement. Its value to the agent is as a citation anchor and a vocabulary fix: the same `run_events(core, gen, on_frame=...)` pattern underlies every adaptive paradigm the review surveys, and the choice of what goes inside `gen` and `on_frame` is what differentiates illumination-tuning from modality-switching from rate-adaptation.

```python
# Same skeleton, four adaptive paradigms — Morgado et al.'s taxonomy.
from useq import MDAEvent
from src.core.hardware.core import run_events

state = {"score_distribution": []}

def on_frame(img, event, meta=None):
    # (i)   Illumination tuning  -> update next event's exposure / laser power
    # (ii)  Acquisition-rate     -> shorten or lengthen min_start_time of next event
    # (iii) Modality switching   -> set state["mode"] = "STED" / "LLSM" / etc.
    # (iv)  Targeted experiment  -> append a new MDAEvent for FRAP/zoom/photoactivation
    score = ml_analyser(img)             # CNN, regressor, RL policy, ...
    state["score_distribution"].append(score)
    decide_next_action(state, score)

def adaptive_gen():
    while not_done(state):
        yield build_next_event(state)    # parameters chosen from state

run_events(core, adaptive_gen(), on_frame=on_frame)
```

What's missing in `src/core/` to match the review's full taxonomy:

- `workflows/data_driven.py` — population-context scoring (covered by André 2023 paper note; not yet implemented).
- A "task score" hook that scores frames against a research-question prompt (LLM-conditioned, foundation-model-conditioned). Currently every workflow scores against fixed feature thresholds. The review's data-driven-vs-task-driven distinction is a candidate for a future workflow module that takes a `score_fn(image, question) -> float` and supplies it to the same survey→rank→zoom skeleton already in `workflows/scanning.py` + `workflows/adaptive.py`.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — review-of-record for the data-driven-microscopy paradigm under which survey→rank→zoom and its variants are organised; names the data-driven vs task-driven scoring distinction that the strategy note's "Writing a Good Event Scorer" section implicitly chooses between.
