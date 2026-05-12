# Sample time vs wall-clock time

A microscope acquires *frames* on its own wall-clock schedule (camera exposure + readout + per-event delay). The *sample* evolves on its own clock, which can be:

- **Wall-clock-locked**: cells respire, contractile rings ingress, tubes grow, calcium oscillates whether you're imaging or not. This is the realistic case for almost all live biology.
- **Snap-locked**: a controlled stimulus delivered via the acquisition script — drug pulses through a microfluidic, SLM patterns, optogenetic illumination — only advances when you fire the corresponding event.
- **Hybrid**: e.g., spontaneous calcium oscillations run continuously, but a drug front advances only when you trigger the perfusion valve.

A scientist's mental model for a real-microscope experiment must match the actual coupling, otherwise the pipeline measures the wrong thing.

## Why this matters

If you assume **wall-clock-locked** when the system is **snap-locked**:

- Reading frame N at t = N seconds of wall-clock implies the sample at t = N. But if every snap costs 50 ms exposure and the sample evolves at 1 sim-step / snap, the sample is at *N steps*, not *N seconds*. Rates expressed in "per second" will be wrong by the snap-time / sim-step ratio.

If you assume **snap-locked** when the system is **wall-clock-locked**:

- Pre-experiment probing (diagnostic snaps, parameter tuning, dry-runs) does not pause the sample. By the time the actual acquisition starts, the sample has moved on. For dynamics with a finite window — beat-frequency-during-drug-pulse, FRAP recovery, calcium-spike-immediately-after-stim — the answer is missed.

## Diagnosis

Run two snaps separated by a known wall-clock gap and compare the frames.

- If the rendered content is identical → snap-locked OR static OR paused. Differentiate by waiting longer (does identical persist over many seconds?) or by performing a known stimulus and checking whether the system responds.
- If the content evolves → wall-clock-locked. Measure the rate empirically: known cell-position drift / known wall-clock interval.
- Measure the snap-to-sim-step coupling directly: fire one MDA event, look at sample. Fire another. Look. If the sample advanced *exactly one perceptible step per snap*, it's snap-locked at 1:1.

```python
core.snapImage(); im0 = core.getImage()
time.sleep(2.0)
core.snapImage(); im = core.getImage()
print('after 2s: changed?', not (im == im0).all())
```

## Coupling strategies

- **Wall-clock-locked dynamics with a finite window**: minimise pre-acquisition probes (one preview is fine; multiple probes consume the window). Connect → orient → acquire in one uninterrupted burst. Don't dry-run on the same proxy / hardware — dry-run on a local synthetic sim if you need to debug the analysis pipeline.
- **Snap-locked dynamics**: probe freely. Each diagnostic snap costs one sim-step but it's accounted for. Budget a known number of snaps.
- **Hybrid**: split your budget. Diagnostic snaps for the snap-locked component (they don't advance the wall-clock biology), but be quick about it (the wall-clock biology runs through the diagnostic).

## On a real microscope

The same coupling shows up:

- **Imaging-driven dynamics**: photobleaching, photoconversion, DNA damage from UV. These advance per snap.
- **Background dynamics**: cell-cycle progression, growth, calcium signaling. These advance with wall-clock.
- **Stimulus-driven**: drug exposure, optogenetic activation, SLM patterning. These advance when you decide to fire them.

A scientist who knows their sample's clock can pre-allocate the experiment correctly. The smart microscope should do the same.

## Anti-patterns

- **Pre-MDA orientation snaps + wall-clock biology**: the most common trap. The orientation snap looks free but the biology has been running between connect and burst-start. Symptoms: sample is past the expected starting state (tubes already at world-edge, rings already closed, cells already in late stage). One preview is fine; three is a session-killer.
- **Dry-run-then-live on the same hardware with finite-budget dynamic state**: the dry-run consumes state. Be conservative with pre-acquisition probes for finite-window dynamics.
- **Reporting rates "per second"**: if the sample is snap-locked, report rates "per snap" (== per sim-step). Per-second rates depend on the wall-clock spacing of your snaps, which is hardware-specific.

## Related

- `[[Core/Concepts/MDA engine]]` — how useq events drive the camera schedule.
- `[[Core/Concepts/Event-driven acquisition]]` — when each event waits on a callback decision.
- `[[Core/Strategies/Closed-loop state device]]` — controllers acting on snap-locked stimulus axes.
