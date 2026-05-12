---
title: "Pycro-Manager: open-source software for customized and reproducible microscope control"
authors: Henry Pinkard, Nico Stuurman, Ivan E. Ivanov, Nicholas M. Anthony, Wei Ouyang, Bin Li, … et al.
year: 2021
venue: Nature Methods 18(3):226-228
doi: 10.1038/s41592-021-01087-6
url: https://www.nature.com/articles/s41592-021-01087-6
researched: 2026-04-24
---

## Abstract

μManager, an open-source microscopy acquisition software, has been an essential tool for many microscopy experiments over the past 15 years, but is not easy to use for experiments in which image acquisition and analysis are closely coupled. This is because μManager libraries are written in C++ and Java, whereas image processing is increasingly carried out with data science and machine learning tools most easily accessible through the Python programming language. We present Pycro-Manager, a tool that enables rapid development of such experiments, while also providing access to the wealth of existing tools within μManager through Python.

(Note: the Nature Methods Brief Communication does not carry an official journal abstract — PubMed lists "no abstract available". The text above is the matching abstract from the authors' arXiv preprint of this work, arXiv:2006.11330, which corresponds 1:1 to the published version's framing.)

## Smart microscopy principle

Pycro-Manager's contribution is to make the **acquisition loop itself programmable from Python** without sacrificing access to the broad μManager device-abstraction layer. The high-level interface exposes three composable primitives:

1. **Acquisition events** — declarative records describing what to acquire (channel, exposure, position, slice). The user generates these in Python, optionally on-the-fly.
2. **Acquisition hooks** — user-supplied Python callables that the engine invokes at well-defined points in the acquisition pipeline (before hardware setup, after hardware setup, before image saved, …) to inject custom hardware actions or modify the queue.
3. **Image processors** — Python callables that receive each image as a numpy array as it streams off the camera, and can analyse, transform, or drop it before it reaches storage.

Together these turn the microscope into a Python-native streaming pipeline: events flow in, images flow out, and hooks/processors close the loop. The paper does not introduce a specific biological detector or modality; the contribution is the **programming surface** that subsequent smart-microscopy tools (MicroMator, CyberSco.Py, this codebase via pymmcore-plus / useq-schema) all build on. Real-time smart microscopy without this layer requires writing Java plug-ins; with this layer, every closed-loop paper in the rest of this library is expressible as ≤ 100 lines of Python.

## Implementation on pymmcore-plus

The lineage runs Pycro-Manager → pymmcore-plus / useq-schema. The three Pycro-Manager abstractions map almost 1:1 onto the model used throughout `src/core/`:

| Pycro-Manager primitive | pymmcore-plus / useq equivalent |
|---|---|
| Acquisition event | `useq.MDAEvent` (channel, exposure, x/y/z, slm_image, metadata) |
| Acquisition event sequence | `useq.MDASequence` or any Python iterable / generator of `MDAEvent` |
| Pre-/post-hardware acquisition hook | `MDAEngine.setup_event` / `exec_event` overrides; `core.mda.events.eventStarted` / `frameReady` / `sequenceFinished` signals |
| Image processor | `on_frame(image, event)` callback connected to `core.mda.events.frameReady` (and used everywhere in this codebase via `run_events(core, gen, on_frame=…)`) |

A Pycro-Manager-style closed-loop sketch translates directly:

```python
from useq import MDAEvent
from src.core.hardware.core import run_events

shared = {"latest": None, "n_cells": 0}

def image_processor(img, event, meta=None):
    """Equivalent to a Pycro-Manager image_processor: runs on every frame."""
    shared["latest"] = img
    shared["n_cells"] = detect_and_count(img)

def event_generator():
    """Equivalent to a Pycro-Manager acquisition-event source: yields events
    on-the-fly so the next event can depend on what we just observed."""
    for i in range(200):
        if shared["n_cells"] >= 100:          # closed-loop early stop
            return
        # When n_cells crosses a threshold, switch from BF polling to
        # short-exposure GFP burst — the same poll/burst pattern Mahecic
        # 2022 implements.
        if shared["n_cells"] >= 30:
            yield MDAEvent(channel={"config": "GFP"}, exposure=10,
                           index={"t": i})
        else:
            yield MDAEvent(channel={"config": "BF"}, exposure=50,
                           index={"t": i})

run_events(core, event_generator(), on_frame=image_processor)
```

What was a Pycro-Manager *acquisition hook* (e.g. "before each frame, query the latest segmentation and decide the channel") is, in this codebase, a generator that yields the next `MDAEvent` after consulting state mutated by `on_frame`. Custom hardware actions that don't fit the `MDAEvent` schema (objective swaps, autofocus calls, log writes) are handled by subclassing `MDAEngine` — see [[Core/Concepts/MDA engine]].

Production hooks in `src/core/`:

- `../../../src/core/hardware/core.py` — `run_events(core, events, on_frame=…)` is the dispatcher that fuses Pycro-Manager's "events + hooks + image processors" into one entry point.
- `../../../src/core/workflows/engine.py` — `MicroscopyEngine` subclasses `MDAEngine` for non-`MDAEvent` actions (objective switching, software autofocus, metric logging) — exactly the role Pycro-Manager's pre-/post-hardware acquisition hooks play.
- The MicroMator ([[Papers/Fox 2022]]) and CyberSco.Py ([[Papers/Chiron 2022]]) ECA frameworks, the event-driven trigger of [[Papers/Mahecic 2022]], and every closed-loop paper in this library all run on top of the Pycro-Manager-class programming model — the "events + hooks + image processors" surface is the lingua franca.

## Cited by

- [[Core/Concepts/MDA standard]] — Pycro-Manager's three abstractions (acquisition events, acquisition hooks, image processors) are the direct ancestors of `useq.MDAEvent` + `MDAEngine` overrides + `on_frame` callback that this note documents.
- [[Core/Concepts/MDA engine]] — Pycro-Manager's acquisition hooks (pre-/post-hardware) are the analogue of subclassing `MDAEngine.setup_event` / `exec_event` for custom hardware actions that don't fit the `MDAEvent` schema.
