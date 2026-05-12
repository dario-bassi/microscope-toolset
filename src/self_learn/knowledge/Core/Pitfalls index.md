# `Pitfalls/` — generalisable failure modes

Methodology traps and their structural fixes. Each note names a **measurement problem** (not an event), describes the symptom, explains why the obvious approach fails, and points at the fix. Sim-agnostic framing: a PhD student at a real microscope should read these and recognise the trap in their own data.

A good pitfall note has three parts: *symptom*, *root cause*, *fix*. If a note is really a retrospective of one specific challenge, it belongs in a recipe (or `_archive/`) rather than here.

## Files

- [[Core/Pitfalls/FOV vs well coverage]] — one FOV is never the whole sample; when to tile.
- [[Core/Pitfalls/Run-and-tumble tracking]] — tracking vs event-counting: conflating tumble events with noise angles.
- [[Core/Pitfalls/Reaction-diffusion classification]] — classifying spatial patterns (spots vs stripes vs spirals) from fluorescence.
- [[Core/Pitfalls/Z-drift autofocus]] — when autofocus makes the drift worse; mode collapse on flat samples.
- [[Core/Pitfalls/Sim-state vs rendered count asymmetry]] — when grader counts per-object sim state but the rendered image fuses/fragments object boundaries (shared membranes, gap junctions). No segmentation tuning closes the gap.
- [[Core/Pitfalls/MDA silent truncation on proxy]] — when the proxy WebSocket blips mid-MDA: runner finishes its event list server-side but a subset of frames never fire the local `on_frame` callback; downstream metric-on-N-frames is silently wrong. Fix: `run_events_checked(..., expected_frames=N)`.

## See also

- [[Core/Approach index]] — the diagnostic approaches that catch these.
- [[Core/Strategies index]] — every strategy note should cite the pitfalls it avoids.
- `../Recipes/` — sample-specific recipes often reference pitfalls with concrete parameters.
