# `concepts/` — microscopy physics + pymmcore-plus API reference

Reference notes. Two kinds intermingled: **physical microscopy** (exposure, SNR, Nyquist, depth of field, chromatic aberration, fluorophore basics) and **pymmcore-plus / useq API** (core objects, MDA engine, event-driven acquisition). Each entry is mostly self-contained and links to `src/core/` for the working code it describes.

Read concepts as you need them — they're cited from `strategies/` and `approach/` notes when a reader hits "wait, what's Nyquist doing here?".

## Physics

- [[Core/Concepts/Exposure and photodamage]] — the single knob entangled with dynamic range AND biology; set once per channel.
- [[Core/Concepts/SNR and dynamic range]] — two axes of image quality, often confused.
- [[Core/Concepts/Nyquist sampling]] — why magnification choice is decided by what you need to measure, not what looks nice.
- [[Core/Concepts/Depth of field]] — the Z-budget of a single frame; sets how thick a stack needs to be.
- [[Core/Concepts/Chromatic aberration]] — why channels don't agree on where a point is; matters for colocalization.
- [[Core/Concepts/Fluorophore basics]] — four decisions per fluorescence acquisition: fluorophore, excitation, emission, exposure.

## pymmcore-plus / useq API

- [[Core/Concepts/Core basics]] — snap, move, set-config: the minimum viable microscope-control vocabulary.
- [[Core/Concepts/MDA standard]] — `MDASequence` for pre-planned, non-adaptive acquisitions.
- [[Core/Concepts/MDA generators]] — generator-based MDA for conditional logic and adaptive decisions.
- [[Core/Concepts/MDA engine]] — subclassing `PMDAEngine` for non-standard hardware actions.
- [[Core/Concepts/Event-driven acquisition]] — poll/burst state machine for transient-event capture.
- [[Core/Concepts/Migration notes]] — porting old `scratch/` scripts to `pymmcore-proxy` + `run_events`.

## See also

- `Approach index.md` — how these concepts plug into the decision-making loop.
- `Strategies index.md` — worked workflows that cite these concepts.
- `../../src/core/hardware/core.py`, `src/core/workflows/mda.py` — the production code these notes document.

## References

Paper citations live in [[Papers index]]. Topics relevant to this tier of notes (confocal-physics handbook, event-driven acquisition, automated adaptive acquisition) are tracked in [[Papers candidates]] until verified.
