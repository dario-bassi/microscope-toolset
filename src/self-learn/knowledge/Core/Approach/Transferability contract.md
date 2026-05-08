# Transferability contract

A solve is *transferable* iff every input it consumes and every action
it issues would also exist on a real pymmcore-plus microscope. The
sim is a stand-in; if a recipe scores 10/10 by reading a quantity
that only the simulator exposes, that recipe transfers to **zero**
real microscopes — the score is bookkeeping, not knowledge.

## The contract

Allowed inputs:
- **Camera frames.** `core.snapImage()` + `core.getImage()` returning a
  raw sensor numpy array. MDA `frameReady` callbacks. Same on real
  hardware.
- **Standard device-property reads.** `core.getProperty(device, prop)`
  for properties a real Micro-Manager device adapter would publish
  (Exposure, Binning, Gain, current state of a state device, etc.).
- **Brief text.** Whatever scenario context the orchestrator hands
  the agent before the run.

Allowed actions:
- `core.setProperty(device, prop, value)`
- `core.setConfig(group, preset)`
- `core.setExposure(ms)`
- `core.setXYPosition(x, y)` / `core.setPosition(z)`
- `core.setSLMImage(slm, mask)` / `core.displaySLMImage(slm)`
- `core.mda.run(events)` (and the local `run_events` wrapper).

Anything else is a sim escape hatch — re-derive the quantity from
the camera or do not use it.

## Forbidden patterns

- **Bridge RPC.** `core._rpc("bridge.get_cell_state")`,
  `bridge.step`, `bridge.get_dose_report`, `bridge.set_dose_budget`,
  `bridge.set_pixel_mask`. The full `bridge.*` namespace was closed
  2026-04-27 and there is no real-microscope equivalent.
- **Camera virtual properties exposing sim state.**
  `Camera.SimTime`, `Camera.LastFrameBleachDose`,
  `Camera.LastFrameExposureMs`, `Camera.LastFramePixelCoverage`,
  `Camera.EmissionWavelengthNm`. A real camera publishes sensor
  config (binning, gain, ROI, temperature), not optical-path
  metadata or per-frame physics accounting.
- **Anything that returns ground truth without going through image
  analysis.** True cell positions, true per-cell internal states,
  true noise-free signal, true budget remaining. If the bench
  scientist would need a microscope + analysis to know it, so do you.

## Why it matters

Sim-internal accounting (bleach budgets, per-cell concentrations,
pacemaker identities) has to be *re-derived from frames* on a real
rig. A recipe that reaches for the bridge collapses to zero
generality the moment the bridge isn't there. The 30-of-33 win
streak that ended 2026-04-27 was inflated by ~9 solves that took
that shortcut. The streak number was real; the transfer value
wasn't.

When you find yourself wishing for a sim-internal quantity, the
question to ask is: *how would I measure this on a real microscope?*
That answer is the recipe.

## See also

- [[Core/Strategies/Per-cell measurement from frames]] — the
  canonical replacement for `bridge.get_cell_state` →
  `Population.filter`. Segment + per-ROI extract + predicate filter,
  on real camera frames.
- `../../../NON_NEGOTIABLES.md` env rule 1 — camera = raw sensor.
- `CLAUDE.md` agent rule 2 — never read simulator source.
- `../../../../virtual-env/skills/transferability-audit.md` —
  the audit procedure that catches sim-escape recipes before they
  ship.
