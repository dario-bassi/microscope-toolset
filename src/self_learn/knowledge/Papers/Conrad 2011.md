---
title: "Micropilot: automation of fluorescence microscopy-based imaging for systems biology"
authors: Christian Conrad, Annelie Wünsche, Tze Heng Tan, Jutta Bulkescher, Frank Sieckmann, Fatima Verissimo, Arthur Edelstein, Thomas Walter, Urban Liebel, Rainer Pepperkok, Jan Ellenberg
year: 2011
venue: Nature Methods 8(3):246-249
doi: 10.1038/nmeth.1558
url: https://www.nature.com/articles/nmeth.1558
researched: 2026-04-24
---

## Abstract

Quantitative microscopy relies on imaging of large cell numbers but is often hampered by time-consuming manual selection of specific cells. The 'Micropilot' software automatically detects cells of interest and launches complex imaging experiments including three-dimensional multicolor time-lapse or fluorescence recovery after photobleaching in live cells. In three independent experimental setups this allowed us to statistically analyze biological processes in detail and is thus a powerful tool for systems biology.

## Smart microscopy principle

Micropilot is the canonical survey→zoom paradigm in closed-loop form. A fast low-magnification sweep scans many fields; an online classifier identifies cells matching a trained phenotype (e.g. prometaphase, specific organelle morphology); the instrument then autonomously switches to a high-magnification, multi-channel, time-resolved acquisition protocol on each hit — including FRAP or 3D time-lapse — without user intervention. The novel contribution is **unattended coupling** between machine-learning-based cell-of-interest detection and launching of a complex secondary protocol: the microscope spends its photon and time budget on cells the experiment actually cares about, not on uniform sampling. This makes rare-phenotype experiments (e.g. mitotic sub-stages, transient events at low frequency) statistically tractable where manual selection would be prohibitive.

## Implementation on pymmcore-plus

The two-tier acquisition maps cleanly to two linked `MDASequence`s: a survey sequence that feeds detections into a shared queue, and an adaptive generator that yields high-magnification follow-up events per hit. An `on_frame` callback runs the phenotype classifier on each survey tile and enqueues world-coordinate targets.

```python
from useq import MDAEvent, MDASequence
from src.core.hardware.core import run_events, set_objective

# --- Phase 1: low-mag survey, classifier watches every frame ---
survey_seq = MDASequence(
    stage_positions=[{"x": x, "y": y} for (x, y) in grid],
    channels=[{"config": "nucleus", "exposure": 20}],
)

hits = []  # list of {"x": wx, "y": wy, "score": p}

def classify_on_frame(img, event, meta=None):
    for cell in detect_cells(img):
        p = phenotype_classifier(img, cell)   # e.g. small CNN or shape rule
        if p > TRIGGER_THRESHOLD:
            wx, wy = pixel_to_world(cell["centroid_px"], event)
            hits.append({"x": wx, "y": wy, "score": p})

set_objective(core, 10)
run_events(core, list(survey_seq), on_frame=classify_on_frame)

# --- Phase 2: high-mag follow-up protocol on every hit ---
def followup_gen():
    for i, h in enumerate(hits):
        # FRAP-like protocol: pre-bleach z-stack, bleach, recovery time-lapse.
        for z in range(-2, 3):
            yield MDAEvent(x_pos=h["x"], y_pos=h["y"], z_pos=z * 0.5,
                           channel={"config": "GFP", "exposure": 100},
                           index={"p": i, "z": z + 2, "t": 0})
        yield MDAEvent(x_pos=h["x"], y_pos=h["y"],
                       action={"type": "bleach", "power": 100},
                       index={"p": i, "t": 1})
        for t in range(20):
            yield MDAEvent(x_pos=h["x"], y_pos=h["y"],
                           channel={"config": "GFP", "exposure": 50},
                           min_start_time=t * 1.0,
                           index={"p": i, "t": 2 + t})

set_objective(core, 40)
run_events(core, followup_gen())
```

Production hooks in `src/core/`:

- `../../../src/core/workflows/scanning.py` — `scan_and_detect` produces the Phase-1 hit table.
- `../../../src/core/workflows/adaptive.py` — `survey_cells`, `rank_by_feature`, `zoom_and_measure` already encode the survey→rank→zoom skeleton.
- A dedicated Micropilot-style workflow module (`src/core/workflows/phenotype_triggered.py`) does not yet exist; this paper motivates one whose distinguishing feature over a plain survey→zoom is that the follow-up is a **configurable protocol** (FRAP, 3D time-lapse, multi-colour) chosen per phenotype class, not just a higher-mag snap.

Calibration on a real prep: train the classifier on a few dozen hand-labelled survey frames before the live run; hold back a validation set to set `TRIGGER_THRESHOLD` at the precision/recall point the experiment needs (typically precision-biased — false positives waste the per-hit protocol budget, false negatives are recoverable by re-sweeping).

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — Micropilot is the canonical reference for the survey→rank→zoom pattern with an unattended high-mag follow-up protocol.
