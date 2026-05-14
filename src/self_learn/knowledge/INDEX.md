# Knowledge Index

A teaching notebook for programmatic smart microscopy on pymmcore-plus, written at PhD-student level. Someone new to the field should be able to read through `Core/`, understand what smart microscopy is, see worked examples of real applications (with code), and know where to find the production implementation.

**What lives here vs `src/`:**
- `knowledge/` — prose, principles, worked examples, literature-backed application sketches, pitfalls.
- `src/self_learn/` — the production library the prose refers to. Knowledge notes *point at* `src/self_learn/` modules; they don't duplicate them. A snippet in a knowledge note is illustrative; the real implementation (error handling, parameterization, tests) is in `src/`.

## Structure

| Folder | Holds | Example |
|---|---|---|
| `Core/Approach/` | **Meta: how to think about a problem** — open an experiment, backward design, OADA loops, visual verification | "snap all channels first, look with vision, confirm markers before committing" |
| `Core/Concepts/` | **Reference: physics + API** — exposure/saturation, photodamage, SNR, Nyquist; pymmcore-plus core, useq MDA | "exposure × intensity is the dose budget; saturation destroys dynamic range" |
| `Core/Strategies/` | **Workflow applications** — literature-backed smart microscopy implemented as pymmcore-plus workflows | "Event-driven microscopy (Mahecic 2022): trigger → switch mode → capture → return" |
| `Core/Pitfalls/` | **Failure-mode catalogue** — generic methodology mistakes with their structural fixes | "angle threshold must exceed the 95th-percentile noise angle" |
| `Papers/` | **Verified citations** — fetched DOI + copy-faithful abstracts. Cite with `[[../Papers/<slug>]]`. |  |

## The test for a core note

*Copy the prose into a real lab's onboarding doc — would a PhD student at a real microscope read it naturally?*

- **Yes:** it's core.
- **No — it names a specific experiment or backend as the source of insight:** scope it to the experiment instead.
- **Partially:** split — pull out the generalizable principle as a core note, leave the specific bits in a companion note.

## Reading order for a new experiment

1. `Core/Approach/How to approach a problem.md` — always.
2. `Core/Approach/Backward design.md` — design from the answer backward.
3. Relevant `Core/Strategies/*.md` for the workflow (multi-scale, feedback control, etc.).
4. `Core/Concepts/` when physics or API questions come up.
5. `Core/Pitfalls/` as you hit familiar-sounding trouble.

### By experiment type

**Timelapse**
→ `Core/Strategies/Timelapse design.md` → `Core/Strategies/Gentle imaging.md` → `Core/Strategies/Closed-loop autofocus.md` → `Core/Concepts/MDA engine.md` → `Core/Pitfalls/Photobleaching during timelapse.md` → `Core/Pitfalls/Stage drift exceeding autofocus range.md`

**Multi-position scan**
→ `Core/Strategies/Multi-position survey.md` → `Core/Strategies/Multichannel scan.md` → `Core/Strategies/Multi-scale morphometry.md` → `Core/Pitfalls/FOV vs well coverage.md` → `Core/Strategies/Physical-unit thresholds.md`

**SLM / optogenetics**
→ `Core/Strategies/SLM optogenetics.md` → `Core/Strategies/Feedback control.md` → `Core/Strategies/Connectivity mapping.md` → `Core/Strategies/Simultaneous SLM targeting.md` → `Core/Strategies/Dose threshold.md`

**Motility / tracking**
→ `Core/Strategies/Per-cell measurement from frames.md` → `Core/Strategies/Timelapse design.md` → `Core/Pitfalls/Run-and-tumble tracking.md` → `Core/Strategies/Wave propagation.md`

## Core/Approach/ — meta: how to think about a microscopy problem

Process-level notes: how to open a new experiment, design experiments backward from the answer, close the observe-decide-act loop, verify visually.

- `How to approach a problem.md`
- `Backward design.md`
- `OADA loop.md`
- `Information driven.md`
- `Visual verification.md`
- `Error recovery.md`
- `Experiment Verification Checklist.md`
- `Measurement Verification Architecture.md`
- `Real-Microscope Contract.md`
- `Acquisition strategy.md`
- `Cell classification.md`
- `Confidence assessment.md`
- `Coordinate systems.md`
- `Detection strategy.md`
- `Image quality.md`
- `MDA solve pattern.md`
- `When to use what.md`

## Core/Concepts/ — reference: microscopy physics + pymmcore-plus API

Short reference entries. Two kinds intermingled: (1) **physical microscopy** — exposure, saturation, photodamage, SNR, Nyquist, depth of field, chromatic aberration; (2) **pymmcore-plus / useq API** — core objects, MDA, event-driven acquisition.

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

## Core/Strategies/ — workflow applications

Literature-backed smart-microscopy applications, each implemented (or sketched) as a pymmcore-plus workflow. The pattern: name the application, cite the paper or class of experiments it comes from, sketch the control loop, link to the production implementation in `src/self_learn/workflows/`.

- `Adaptive acquisition.md`
- `Auto recipe selection.md`
- `Axis sweep alignment.md`
- `Closed-loop autofocus.md`
- `Closed-loop state device.md`
- `Connectivity mapping.md`
- `Dose threshold.md`
- `Feedback control.md`
- `Fit-then-control.md`
- `Gentle imaging.md`
- `Imaging parameter optimization.md`
- `Measurement methodology.md`
- `Multichannel scan.md`
- `Multi-position survey.md`
- `Multi-scale morphometry.md`
- `Multi-well comparison.md`
- `Per-cell measurement from frames.md`
- `Physical-unit thresholds.md`
- `Pulsed schedule trajectory.md`
- `Rate-limited drive.md`
- `Reinforcement learning in acquisition.md`
- `Sample-class auto-detection.md`
- `Segmentation backend.md`
- `Simultaneous SLM targeting.md`
- `SLM optogenetics.md`
- `Smart microscopy substrates.md`
- `Strategies index.md`
- `Timelapse design.md`
- `Wave propagation.md`

## Core/Pitfalls/ — generalizable failure modes

Methodology traps with structural fixes. Sim-agnostic framing: describe the measurement problem, not the specific experiment where it bit you.

- `FOV vs well coverage.md`
- `MDA silent truncation on proxy.md`
- `Reaction-diffusion classification.md`
- `Run-and-tumble tracking.md`
- `Sim-state vs rendered count asymmetry.md`
- `Z-drift autofocus.md`
- `Photobleaching during timelapse.md`
- `Stage drift exceeding autofocus range.md`
- `Channel bleedthrough.md`

## When you write a new note

- **Ask: is this generalizable?** Real-lab-onboarding test. If it only makes sense for one specific sample type or setup, scope it accordingly.
- **Link to src/.** A knowledge note that doesn't point at implementation code is probably incomplete.
- **Link to sibling notes.** Microscopy concepts build on each other; every note should cite 2–4 sibling notes under a "See also" footer. Treat the notebook as a graph — a reader following any single note should see natural next hops.
- **Literature citations welcome in Strategies/.** Smart microscopy has a real literature (event-driven, adaptive, closed-loop). Cite papers, describe the class of experiment, sketch the control loop.
