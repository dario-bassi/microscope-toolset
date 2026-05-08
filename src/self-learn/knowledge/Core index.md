# `Core/` — sim-agnostic teaching notebook

The transferable half of the knowledge base. Everything here should read naturally if you paste it into a real lab's onboarding doc. If a note only makes sense in the context of one simulator or one challenge, it belongs in `Recipes/` instead.

Four tiers, each with its own index. Read in the order given for a new challenge; consult out of order when you need reference material.

## [[Core/Approach index]] — how to think about the problem

Process-level notes. How you open a challenge, design experiments backward from the answer, close the observe-decide-act loop, verify visually, recover from errors. This is the meta-layer — a sharp method beats a sharp tool.

**Start here** for any new problem: [[Core/Approach/How to approach a problem]] → [[Core/Approach/Backward design]] → [[Core/Approach/OADA loop]].

## [[Core/Concepts index]] — physics + API reference

Reference entries. Two kinds intermingled: **physical microscopy** (exposure, SNR, Nyquist, depth of field, chromatic aberration, fluorophore photophysics) and **pymmcore-plus / useq API** (core objects, MDA engine, event-driven acquisition). Read as needed — strategy notes cite these when a reader hits "wait, what does Nyquist have to do with this?".

**Most-cited**: [[Core/Concepts/Nyquist sampling]], [[Core/Concepts/Exposure and photodamage]], [[Core/Concepts/MDA standard]], [[Core/Concepts/MDA generators]].

## [[Core/Strategies index]] — workflow applications

Literature-backed smart-microscopy applications, implemented as pymmcore-plus workflows. Multi-position survey, adaptive acquisition, feedback control, closed-loop autofocus, multi-scale morphometry, physical-unit thresholds, event-driven stimulation. Each note sketches the control loop and links to `src/core/workflows/` for the full implementation.

## [[Core/Pitfalls index]] — failure modes

Methodology traps with structural fixes. Each describes a *measurement problem* (FOV vs well coverage, run/tumble tracking, Z-drift autofocus collapse) — not a one-off incident. A reader at a real microscope should recognise these in their own data.

## The reading order for a new challenge

1. [[Core/Approach/How to approach a problem|Approach/how_to_approach_a_problem]] — always.
2. [[Core/Approach/Backward design|Approach/backward_design]] — what do you need to measure? Work back from there.
3. If the sample is familiar: [[Recipes index]]`<sample>.md`.
4. Relevant [[Core/Strategies index|strategies]] for the workflow (multi-scale, feedback control, etc.).
5. [[Core/Concepts index|concepts]] when physics or API questions come up.
6. [[Core/Pitfalls index|pitfalls]] as you hit familiar-sounding trouble.

## See also

- [[Recipes index]] — the sim-specific playbook tier.
- [[INDEX]] — top-level entry point.
- `../src/core/` — the production library these notes describe.
