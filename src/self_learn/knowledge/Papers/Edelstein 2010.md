---
title: Computer control of microscopes using µManager
authors: Arthur Edelstein, Nenad Amodaj, Karl Hoover, Ron Vale, Nico Stuurman
year: 2010
venue: Current Protocols in Molecular Biology 92(1):14.20.1-14.20.17
doi: 10.1002/0471142727.mb1420s92
url: https://currentprotocols.onlinelibrary.wiley.com/doi/10.1002/0471142727.mb1420s92
researched: 2026-04-24
---

## Abstract

With the advent of digital cameras and motorization of mechanical components, computer control of microscopes has become increasingly important. Software for microscope image acquisition should not only be easy to use, but also enable and encourage novel approaches. The open-source software package µManager aims to fulfill those goals. This unit provides step-by-step protocols describing how to get started working with µManager, as well as some starting points for advanced use of the software.

## Smart microscopy principle

µManager is the **vendor-agnostic device-abstraction layer** that the entire modern smart-microscopy stack is built on. Its contribution is not a closed-loop algorithm but the substrate that makes closed-loop algorithms portable: a single C++ Core (`MMCore`) that exposes every motorised microscope, camera, stage, filter wheel, shutter, and SLM through a uniform device-property API, with a thin set of typed device adapters bridging vendor SDKs (Nikon, Zeiss, Leica, Olympus, Andor, Hamamatsu, …) underneath. Above the Core sits a Java GUI and a scripting surface (Beanshell, then Python via `pycromanager` / `pymmcore` / `pymmcore-plus`) so the same acquisition logic runs unchanged across instruments.

This is the precondition for everything else in this library. Event-driven acquisition (Mahecic 2022), reactive ECA frameworks (MicroMator, CyberSco.Py), AI-guided ROI selection (Bouchard 2023, Shi 2024), and LLM-driven instrument control (Mandal 2025) are all written against the µManager device abstraction (directly or via Pycro-Manager / pymmcore-plus). Without a uniform device API, every closed-loop paper would have to rewrite its instrument layer per microscope; with µManager, the smart-acquisition logic is the only thing that has to change. The Edelstein 2010 *Current Protocols* unit is the canonical citation for the platform — the paper that the smart-microscopy field cites when it needs to point at "the open-source microscope-control substrate."

## Implementation on pymmcore-plus

`pymmcore-plus` *is* µManager — accessed from Python instead of from the µManager Java GUI. The lineage is:

```
µManager device adapters (C++)  ←  Edelstein 2010 substrate
        ↓
MMCore (C++)                    ←  vendor-agnostic device API
        ↓
pymmcore (CPython binding)      ←  raw Core in Python
        ↓
pymmcore-plus  (CMMCorePlus)    ←  Pythonic wrapper + MDA engine + signals
        ↓
useq-schema  (MDAEvent / MDASequence)  ←  declarative acquisition spec
        ↓
this codebase (`src/core/hardware/core.py::run_events`, `src/core/workflows/engine.py::MicroscopyEngine`)
```

Every call we make in a solve script eventually crosses the µManager device-API boundary:

```python
from pymmcore_plus import CMMCorePlus
from useq import MDAEvent
from src.core.hardware.core import run_events

core = CMMCorePlus.instance()
core.loadSystemConfiguration("MMConfig_demo.cfg")  # µManager config file format

# Discover devices via the µManager Core API — same call on every microscope:
groups = core.getAvailableConfigGroups()        # ('Channel', 'Objective', ...)
channels = core.getAvailableConfigs('Channel')  # ('DAPI', 'GFP', 'BF', ...)
labels = core.getStateLabels('Objective')       # ('4x', '10x', '20x', '40x', ...)

# Same MDA logic runs against any µManager-supported instrument:
def on_frame(img, event):
    ...

run_events(core, [MDAEvent(channel={"config": "GFP"}, exposure=50)],
           on_frame=on_frame)
```

What this means in practice: **every device-control idiom documented in [[Core/Concepts/Core basics]] — `setConfig(group, preset)`, `setState`, `setPosition`, `getPixelSizeUm`, `setXYPosition`, the property-browser model — is a direct surfacing of the µManager Core API**. The configuration-file format (`.cfg` / property presets), the device-property naming convention, the synchronous `snapImage()` / `getImage()` pattern, and the device-adapter plug-in model are all µManager. `pymmcore-plus` adds a Pythonic wrapper, signals, and an MDA engine on top; `useq-schema` adds a declarative acquisition spec; this codebase sits one layer further up. The closed-loop generators and `on_frame` callbacks documented in [[Core/Concepts/MDA engine]] are *only* possible because µManager already solved the "talk to arbitrary microscope hardware" problem.

There is no production module in `src/core/` that re-implements µManager — the whole point is that we never have to. The `src/core/hardware/` layer is the thinnest viable shim over `CMMCorePlus`, which is itself a thin shim over µManager's MMCore.

## Cited by

- [[Core/Concepts/Core basics]] — every Core-API call documented there (`setConfig`, `setState`, `getStateLabels`, `getAvailableConfigGroups`, `setXYPosition`, `getPixelSizeUm`, `snapImage`/`getImage`) is a direct surfacing of the µManager device-abstraction layer this paper introduces; pymmcore-plus accesses the same C++ Core from Python.
- [[Core/Concepts/MDA standard]] — the MDA programming model of [[Papers/Pinkard 2021]] and `useq.MDASequence` runs on top of the µManager Core; without the device-abstraction layer Edelstein 2010 documents, the declarative acquisition spec would have nothing portable to drive.
