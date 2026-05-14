# Real-Microscope Contract

> **TL;DR** — A protocol is transferable to real hardware only if all its inputs come from
> camera frames + standard device properties. No simulation-internal state allowed.
> Allowed: `snapImage()`, `getProperty()`, `setXYPosition()`, `mda.run()`, any standard MM API.
> Forbidden: sim ground truth, per-frame physics accounting, custom bridge RPC layers.
> Rule: *"How would I measure this on a real microscope?"* — that answer is the protocol.

A measurement protocol is *transferable* to real hardware iff every input
it consumes and every action it issues would exist on a real
pymmcore-plus microscope. If a protocol reads a quantity that only
exists in a simulation or custom bridge layer, it transfers to
**zero** real microscopes.

## Allowed inputs

- **Camera frames.** `core.snapImage()` + `core.getImage()` returning a
  raw sensor numpy array. MDA `frameReady` callbacks. These work on
  any real hardware.
- **Standard device-property reads.** `core.getProperty(device, prop)`
  for properties a real Micro-Manager device adapter publishes
  (Exposure, Binning, Gain, current state of a state device, etc.).
- **Experiment description / brief text.** Any context provided
  to the agent before the run.

## Allowed actions

- `core.setProperty(device, prop, value)`
- `core.setConfig(group, preset)`
- `core.setExposure(ms)`
- `core.setXYPosition(x, y)` / `core.setPosition(z)`
- `core.setSLMImage(slm, mask)` / `core.displaySLMImage(slm)`
- `core.mda.run(events)` (and the local `run_events` wrapper).

Anything else is a sim escape hatch — re-derive the quantity from
the camera, or do not use it.

## Forbidden patterns

- **Simulation-internal state reads.** Direct access to simulator
  ground truth, internal cell states, per-frame physics accounting.
  A real camera only publishes sensor config (binning, gain, ROI,
  temperature), not optical-path metadata or per-frame physics.
- **Anything that returns ground truth without image analysis.**
  True cell positions, true per-cell internal states, true
  noise-free signal. If a bench scientist would need a microscope
  + analysis to know it, so do you.
- **Custom bridge RPC layers.** Any RPC interface that exposes
  simulation-internal state has no real-microscope equivalent.

## Why it matters

Any protocol that reads simulation-internal quantities has zero
generality the moment those quantities aren't available — which is
always the case on real hardware. When you find yourself wishing
for a sim-internal quantity, the question to ask is:

> *How would I measure this on a real microscope?*

That answer is the protocol.

## Re-derive from frames

Per-cell quantities (concentration, state, dose received) must be
re-derived from camera frames + standard segmentation:

1. Segment: `detect_cells(img)` or `segment_tissue(img)`
2. Measure per-ROI: intensity, morphology, texture
3. Filter on a predicate over the measurements

This is the canonical replacement for any "read from sim state"
pattern. It transfers to real hardware because it only uses camera
frames and standard image analysis.

## See also

- [[Core/Approach/How to approach a problem]] — the 7-step process
  that puts real-microscope constraints into practice.
- [[Core/Strategies/Measurement methodology]] — measurement patterns
  that satisfy this contract.
- [[Core/Pitfalls/Sim-state vs rendered count asymmetry]] — what
  happens when the protocol diverges from what the image shows.
