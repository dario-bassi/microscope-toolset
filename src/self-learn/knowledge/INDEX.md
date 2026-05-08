# Knowledge Index

This is a teaching notebook for programmatic smart microscopy on pymmcore-plus at PhD-student level. Written so someone new can read through `Core/`, understand the field, and know where the production code lives (`src/core/`).

See `README.md` for the framing, contract, and sim-agnostic-prose test.

## Primary entry points

- **[[Core index]]** — the sim-agnostic teaching notebook, with its own sub-indexes for `Approach/`, `Concepts/`, `Strategies/`, `Pitfalls/`.
- **[[Recipes index]]** — sim-specific playbooks, grouped by sample class.

The lists below are an inventory; the indexes above are the curated entry points.

## Core/ — sim-agnostic

### Approach/ — meta: how to think about a microscopy problem

Process-level notes: how to open a new challenge, design experiments backward from the answer, close the observe-decide-act loop, verify visually.

- `How to approach a problem.md`
- `Backward design.md`
- `OADA loop.md`
- `Information driven.md`
- `Visual verification.md`
- `Error recovery.md`
- `Pre-submission checklist.md`
- `Pre-submit guard architecture.md`
- `Acquisition strategy.md`
- `Cell classification.md`
- `Confidence assessment.md`
- `Coordinate systems.md`
- `Detection strategy.md`
- `Image quality.md`
- `MDA solve pattern.md`
- `When to use what.md`

### Concepts/ — reference: microscopy physics + pymmcore-plus API

Short reference entries. Two kinds intermingled: (1) **physical microscopy** — exposure, saturation, photodamage, SNR, Nyquist, depth of field, chromatic aberration; (2) **pymmcore-plus / useq API** — core objects, MDA, event-driven acquisition. Each entry is mostly self-contained and links to `src/core/` for working code.

- `Exposure and photodamage.md` *(physics)*
- `Nyquist sampling.md` *(physics)*
- `SNR and dynamic range.md` *(physics)*
- `Depth of field.md` *(physics)*
- `Fluorophore basics.md` *(physics)*
- `Chromatic aberration.md` *(physics)*
- `Core basics.md` *(API)*
- `Event-driven acquisition.md` *(API)*
- `MDA engine.md`, `MDA generators.md`, `MDA standard.md` *(API)*
- `Migration notes.md`

*All physics concept gaps closed.* Add more as questions arise that benefit from having physics explained once and cited many times.

### Strategies/ — workflow applications

Literature-backed smart-microscopy applications, each one implemented (or sketched) as a pymmcore-plus workflow. The pattern: name the application, cite the paper or class of experiments it comes from, sketch the control loop, link to the production implementation in `src/core/workflows/` or `src/core/acquisition/`.

- `Gentle imaging.md`
- `Measurement methodology.md`
- `Adaptive acquisition.md`
- `Closed-loop autofocus.md`
- `Closed-loop state device.md`
- `Connectivity mapping.md`
- `Feedback control.md`
- `Imaging parameter optimization.md`
- `Multichannel scan.md`
- `Multi-position survey.md`
- `Multi-scale morphometry.md`
- `Multi-well comparison.md`
- `Physical-unit thresholds.md`
- `Segmentation backend.md`
- `Simultaneous SLM targeting.md`
- `SLM optogenetics.md`
- `Timelapse design.md`
- `Wave propagation.md`
- `Auto recipe selection.md`
- `Sample-class auto-detection.md`
- `Smart microscopy substrates.md`
- `Reinforcement learning in acquisition.md`
- `Session retrospective.md`
- `Operating the solve loop.md`
- `Rate-limited drive.md`
- `Pulsed schedule trajectory.md`
- `Fit-then-control.md`

### Pitfalls/ — generalizable failure modes

Methodology traps with structural fixes. Sim-agnostic framing: describe the measurement problem, not the specific challenge where it bit you.

- `FOV vs well coverage.md`
- `MDA silent truncation on proxy.md`
- `Reaction-diffusion classification.md`
- `Run-and-tumble tracking.md`
- `Sim-state vs rendered count asymmetry.md`
- `Z-drift autofocus.md`

## Recipes/ — sim-specific playbooks

Sample/scale-tuned observations. Each pairs (often) with a `src/recipes/*.py` module of the same name.

- `Adaptive STED.md`, `Bacteria growth.md`, `Bacterial light trap.md`, `Blood smear WBC.md`, `Brightfield cells.md`
- `C elegans.md`, `Calcium imaging.md`, `Colocalization.md`, `Cybergenetics.md`, `Dictyostelium.md`, `Disk diffusion.md`
- `Dose response.md`, `Drug response timelapse.md`, `Event-driven modality switch.md`, `Fibroblasts.md`, `Flow cytometry.md`, `Fluorescence.md`, `FRAP.md`, `FUCCI.md`
- `Galvanotaxis.md`, `Hemocytometer.md`, `Histology.md`, `Lipid droplet.md`, `Malaria.md`, `MDA workflows.md`, `Microfluidic.md`
- `Mitochondrial dynamics.md`, `Motile organism tracking.md`, `Multi-position expression scan.md`
- `Neurons.md`, `Optogenetics.md`, `Organoid.md`, `Perturbation kinetics.md`, `Photoconversion.md`
- `Spatial distribution.md`, `Spheroid.md`, `SPT diffusion.md`, `Stress granule formation.md`
- `Temperature kinetics.md`, `Timelapse.md`, `Tissue triage.md`, `Wound healing.md`, `Yeast bud scars.md`, `Yeast division.md`
- `Zebrafish.md`, `Z-stack 3D.md`

## Reading order for a new challenge

1. `Core/Approach/How to approach a problem.md` — always.
2. `Core/Approach/Backward design.md` — design from the answer backward.
3. If the sample is familiar: `Recipes/<sample>.md`.
4. Relevant `Core/Strategies/*.md` for the workflow (multi-scale, feedback control, etc.).
5. `Core/Concepts/` when physics or API questions come up.

## When you write a new note

- **Ask: does this belong in core or recipes?** Copy-into-real-lab-onboarding test. If it cites ch### or a simulator state integer as *load-bearing*, it's a recipe. If a recipe is cited as *an example* of a core principle, that's fine.
- **Link to src/.** A knowledge note that doesn't point at `src/core/X` or `src/recipes/X` is probably incomplete — either the implementation should be extracted to code, or the note should cite the existing module.
- **Link to sibling notes.** Microscopy concepts build on each other; every note should cite 2–4 sibling notes under a "See also" footer. A note on SNR links to exposure, Nyquist, and the strategies that use SNR as a decision variable. A strategy links to the concepts it assumes and the pitfalls it avoids. Treat the notebook as a graph — a reader following any single note should see natural next hops.
- **Literature citations welcome in strategies/.** Smart microscopy has a real literature (event-driven, adaptive, closed-loop). Cite papers, describe the class of experiment, sketch the control loop. The production details live in `src/`.

