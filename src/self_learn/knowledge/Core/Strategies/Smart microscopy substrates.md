# Smart-microscopy substrates

> **When to use:** When selecting the computational framework or hardware substrate for implementing a closed-loop microscopy experiment.

Most strategy notes here describe **what to do in the loop**: the score function, the trigger, the survey-then-zoom decision. This note describes **what holds the loop together** — the open-source frameworks that supply the device adapters, the streaming dataflow, the GUI on top of it, and the plugin slots where decisions hang. Picking the right substrate up front is the difference between "a closed-loop demo on one rig" and "a routine pipeline running thousands of chambers in parallel".

The four substrate papers below cluster on a single axis: **how is closed-loop authoring exposed to the experimenter?** Pure-Python (Pycro-Manager), Qt-widget GUI on a single process (ImSwitch), streaming multi-process orchestration (Arkitekt), GUI-editable feature-graphs (Navigate), event-driven middleware for production microfluidics (EAP4EMSIG), or a still-paper multi-agent vision (Kesavan).

## The substrate axis

Every substrate makes a different cut on the same control loop:

```
  hardware ↔ acquisition ↔ analysis ↔ event detector ↔ planner ↔ next event
            └───────────── what the substrate exposes ─────────────┘
```

| Substrate          | Authoring surface                  | Cut |
|--------------------|------------------------------------|-----|
| Pycro-Manager      | Python (events + image processors) | Code-as-experiment |
| ImSwitch           | Single-process Qt GUI + plugins    | Acquisition + reconstruction in one place |
| Arkitekt           | Streaming dataflow across services | Plug ML services as nodes |
| Navigate           | GUI-editable feature-graph         | No-code decision routing |
| EAP4EMSIG          | Event-detector middleware          | Production-rate microfluidics |
| Multi-agent (vision) | LLMs as orchestrators of agents  | Aspirational; no impl yet |

Pycro-Manager is the historical baseline — the rest of the substrates explicitly position themselves against the "everything is a Python script" cut by raising the authoring surface up the stack.

## Pycro-Manager — code-as-experiment

[[Papers/Pinkard 2021]] — exposes the µManager Core via a Python streaming pipeline (events + hooks + image processors). This is the "drop the GUI, write everything in code" cut. The `useq.MDASequence` + `run_events(core, gen, on_frame=cb)` pattern in this codebase is the same shape one level removed. Pycro-Manager is the substrate against which the others define themselves.

## ImSwitch + Arkitekt — single-process and streaming

[[Papers/Casas Moreno 2021, Roos 2024]] — two complementary cuts from the Testa/Sibarita labs:

- **ImSwitch** keeps acquisition and reconstruction in one Python process. JSON setup file describes the hardware; `imcontrol` is the Qt GUI on top. The single-process cut means feedback rules can fire on **reconstructed-image features**, not just raw frames — useful for super-resolution platforms (RESOLFT, MoNaLISA) where the loop wants to react to the reconstructed image, not the diffraction-limited camera frames.
- **Arkitekt** (later, Nat Methods 2024) takes the opposite cut: orchestrate streams across multiple processes / services. ML inference, segmentation, and decision logic each become independent services that the substrate wires together. The orchestration cost is higher but the decoupling lets you swap a Cellpose service for a custom CNN service without touching the acquisition code.

Authoring choice: **single-process if reconstruction is part of the trigger**, streaming if you have heterogeneous services (multiple labs' models, GPU-bound inference, web dashboards) that need to compose freely.

## Navigate — GUI-editable feature graphs

[[Papers/Marin 2024]] — pushes smart-microscopy authoring out of Python and into a workflow editor specialised for light-sheet hardware (mesoSPIM, ASLM, oblique-plane, lattice). Two abstractions:

1. **Features** — reusable acquisition / analysis routines (autofocus, tile, segment-tissue, run-z-stack) with GUI-exposed parameters.
2. **Feature containers** — directed graphs of features with **decision nodes** that branch on the previous feature's output. *"Acquire a Z-stack only if tissue is present at this stage position"* is a 2-feature graph saved as configuration — no Python.

Plugin architecture for hardware adapters; REST API for external analysis services to return decisions. The contribution is "code-free decision routing" for users who can already operate the rig but don't want to maintain Python.

## EAP4EMSIG — event-driven middleware for production scale

[[Papers/Friederich 2025]] — takes the substrate question one rung higher: how do you run an event-driven microscope **at production scale** (thousands of microfluidic chambers, 24/7, since 01/2025)? An 8-module dataflow middleware where every component runs at frame-rate. Two flagship technical contributions sit inside the loop:

- An MLP-based autofocus: 87 ms inference, 0.105 µm MAE on a single calibration z-stack — distinct from [[Papers/Pinkard 2019]] in that it works without the off-axis-LED hardware.
- An 11-method segmentation benchmark establishing Cellpose 3 (93.6 % PQ, 1.1 s) and a custom distance-based method (93.0 % PQ, 121 ms) as the speed/accuracy frontier.

Trigger logic is explicit and tunable: technical events (focus loss when |Δz| > 0.5 µm for 3 consecutive timepoints) and biological events (rapid growth = ≥ 10 % cell-count rise between frames) auto-actuate the planner or notify the operator via Slack / dashboard. The lesson is that **routine production smart-microscopy needs explicit operator overrides** built into the substrate, not as an afterthought.

## Multi-agent — the aspirational substrate

[[Papers/Kesavan 2025]] is a position / framework paper, not an implementation. Argues that putting *one* LLM above the instrument's tool surface (the [[Papers/Mandal 2025]] / [[Papers/Boiko 2023]] cut) conflates five qualitatively different cognitive jobs: empirical control, measurement, epistemic linking to a hypothesis, narrative synthesis, orchestration. The proposal is to factor those into five named agent roles, each an LLM with its own context, tools, and objective. No implementation yet — but worth knowing as the framing critique against which next-generation smart-microscopy LLM stacks will be designed.

## How to pick

Decision tree if you're authoring a new closed-loop experiment:

- **One-off Python prototype** → Pycro-Manager. The cut where everything is code.
- **Single-process where reconstruction is the trigger** → ImSwitch.
- **Heterogeneous ML services that need to compose** → Arkitekt.
- **No-code decision routing on light-sheet hardware** → Navigate.
- **Production-rate microfluidics with operator overrides** → EAP4EMSIG.
- **Future-proofing against the multi-agent shape** → leave room in your tool layer for orchestrator factoring.

Don't carry a substrate from a demo into a production setup without re-evaluating the cut. The right substrate at one scale is wrong at the next.

## See also

- [[Core/Strategies/Adaptive acquisition]] — what runs on top of these substrates.
- [[Core/Strategies/Feedback control]] — the primitive these substrates orchestrate.
- [[Core/Approach/MDA solve pattern]] — Pycro-Manager's `MDASequence + run_events` is the shape this codebase uses.
- [[Papers/Edelstein 2010]] — the µManager core that ImSwitch and Pycro-Manager both wrap.
- [[Papers/Kesavan 2025]] — the multi-agent framing critique.
