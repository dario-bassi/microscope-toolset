---
title: "navigate: an open-source platform for smart light-sheet microscopy"
authors: Zach Marin, Xiaoding Wang, Dax W. Collison, Conor McFadden, Jinlong Lin, Hazel M. Borges, Bingying Chen, Dushyant Mehra, Qionghua Shen, Seweryn Gałecki, Stephan Daetwyler, Steven J. Sheppard, Phu Thien, Baylee A. Porter, Suzanne D. Conzen, Douglas P. Shepherd, Reto Fiolka, Kevin M. Dean
year: 2024
venue: Nature Methods 21(11):1967-1969
doi: 10.1038/s41592-024-02413-4
url: https://www.nature.com/articles/s41592-024-02413-4
researched: 2026-04-25
---

## Abstract

navigate is a turnkey, open-source software solution designed to enhance light-sheet fluorescence microscopy (LSFM) by integrating smart microscopy techniques into a user-friendly framework. It provides a Python-based control system that supports GUI-reconfigurable acquisition routines and the integration of diverse hardware sets, and is the only software that enables decision-based acquisition routines to be generated in a code-free format.

(Nature Methods *Brief Communication* — no formal journal abstract; the text above is paraphrased from the published article and the matching bioRxiv preprint, doi:10.1101/2024.02.09.579083, PMID 39261640, PMC11540721.)

## Smart microscopy principle

navigate's contribution is a **substrate that pushes smart-microscopy authoring out of Python scripts and into a GUI**, specialised for the light-sheet hardware set. Where [[Papers/Pinkard 2021]] turned the μManager Core into a Python streaming pipeline (events + hooks + image processors as code), navigate factors the same closed-loop into two GUI-editable abstractions:

1. **Features** — reusable acquisition / analysis routines (e.g. autofocus, tile, segment-tissue, change-channel, run-z-stack) that expose their parameters to the GUI.
2. **Feature containers** — directed graphs of features, including **decision nodes** that branch the acquisition based on the output of the previous feature. *"Acquire a Z-stack only if tissue is present at this stage position"* is a two-feature container with one decision node, drawn in a workflow editor and saved as configuration — no Python.

Above the feature container, navigate exposes a **plugin architecture** for adding device adapters (so far covering mesoSPIM, axially-swept light-sheet (ASLM), digitally-scanned light-sheet, oblique-plane microscopy, and field-synthesis lattice-class instruments) plus a **REST API** so external analysis services (ilastik-class segmenters, Cellpose, custom CNNs) can return decisions to the feature container without being implemented inside navigate. Demonstrated smart routines in the paper include sensorless adaptive optics with an entropy-metric reward, automated tissue segmentation followed by targeted high-magnification capture, and event-triggered acquisition of dynamic biological processes.

This is the **light-sheet sibling of the substrate cluster** ([[Papers/Edelstein 2010]] device-abstraction layer; [[Papers/Pinkard 2021]] Python programming surface; [[Papers/Tosi 2021]] ImageJ-macro front-end for the survey→detect→zoom loop). Where Pycro-Manager makes the loop *programmable* and AutoScanJ makes the detection layer *macro-authorable*, navigate makes the **whole feature graph (including decision branches) GUI-editable** — the same lowering of authoring barrier that AutoScanJ achieved for ImageJ users, applied to light-sheet operators.

## Implementation on pymmcore-plus

navigate is **not** a pymmcore-plus reimplementation — it is a parallel-stack substrate (its own device-abstraction layer, its own MVC control core) targeting light-sheet hardware that μManager covers patchily. The translation is conceptual: navigate's *feature container* is a generator of `MDAEvent`s with branching, and a *decision node* is the `if`/`elif` in that generator. A two-feature container "scout for tissue, then z-stack each tile that has tissue" maps to:

```python
from useq import MDAEvent
from src.core.hardware.core import run_events
from src.core.detection.tissue import segment_tissue

decisions = []  # filled by the scout pass

def scout_on_frame(img, event, meta=None):
    """Feature 1: tissue-presence classifier on each scout tile."""
    mask, _ = segment_tissue(img)
    decisions.append({"event": event, "has_tissue": mask.sum() > 5_000})

def routine():
    # Feature 1 — low-mag scout
    for (x, y) in scout_grid:
        yield MDAEvent(x_pos=x, y_pos=y,
                       channel={"config": "BF"}, exposure=10)
    # Decision node + Feature 2 — z-stack only the tissue-positive tiles
    for d in decisions:
        if not d["has_tissue"]:
            continue
        for z in z_slices:
            yield MDAEvent(x_pos=d["event"].x_pos,
                           y_pos=d["event"].y_pos,
                           z_pos=z,
                           channel={"config": "GFP"}, exposure=50)

run_events(core, routine(), on_frame=scout_on_frame)
```

What the navigate user assembles in a GUI workflow editor, the pymmcore-plus user writes as a generator that consults shared state — the smart-microscopy logic is identical, the authoring surface is the difference. Production hooks in `src/core/`:

- `../../../src/core/workflows/scanning.py` — `scan_and_detect` / `scan_and_detect_mda` are the navigate scout-feature analogue.
- `../../../src/core/workflows/adaptive.py` — `survey_cells`, `rank_by_feature`, `zoom_and_measure` encode the survey→decide→zoom skeleton.
- `../../../src/core/workflows/engine.py` — `MicroscopyEngine` (custom MDA actions) plays the role of navigate's plugin-loaded device features for actions that don't fit `MDAEvent` (objective swaps, software autofocus, log writes).

The paper is also the **substrate citation** for [[Papers/Daetwyler 2025]] from the same Dean / Fiolka lab — Daetwyler 2025's continuous mSPIM-overview / ASLM-capture loop is the kind of dual-modality smart-acquisition routine navigate's feature-container model is designed to express, and the two papers share co-authors (Daetwyler, Fiolka, Dean). When this codebase needs to point at *"the open-source substrate for light-sheet smart microscopy"* it points here; the analogous citations are Edelstein 2010 for any motorised microscope and Pinkard 2021 for Python-native programmable acquisition.

## Cited by

- [[Core/Strategies/Adaptive acquisition]] — navigate's GUI-authorable decision-tree feature graph is the no-code substrate analogue of the survey→detect→zoom pattern; sibling to AutoScanJ (ImageJ-macro front-end), Pycro-Manager (Python-native), and Micropilot (vendor-specific Java) as alternate authoring surfaces for the same online-classifier-driven-acquisition paradigm.
- [[Core/Concepts/MDA standard]] — navigate's *feature* + *feature container* + *decision node* abstractions are an alternate front-end to the same smart-acquisition state machine that `useq.MDAEvent` / `MDASequence` + a generator with branching express in code.
