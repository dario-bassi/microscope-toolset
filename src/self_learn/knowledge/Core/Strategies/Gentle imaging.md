# Gentle Imaging

> **When to use:** When phototoxicity or photobleaching is a concern and the photon budget must be minimized while preserving necessary signal.

Minimize photon dose while maximizing information. Every unnecessary photon
is phototoxicity with no benefit. The gentlest experiment that answers the
question is the best experiment.

> **Note**: Code examples below use conceptual pseudocode. All multi-frame
> loops MUST use `run_events(core, generator(), on_frame=callback)` in practice.
> Single-frame acquisitions use `snap(core, channel=ch)` from `hardware.core`.

---

## Why Gentleness Matters

In live-cell imaging:
- **Photobleaching** degrades fluorophores, reducing signal over time
- **Phototoxicity** damages cells, altering the biology you are measuring
- **Reactive oxygen species** from fluorescence excitation kill cells
- **Heat** from illumination can change cell behavior

Gentle imaging maps directly to real microscope practice and reduces
acquisition time while preserving sample health.

---

## Principle 1: Brightfield First

Brightfield illumination is essentially free. No bleaching, no phototoxicity,
no fluorophore consumption.

**Always start with BF:**
```python
# Good: BF first, fluorescence only when needed
core.setConfig(group, "BF")
core.snapImage()
bf_img = core.getImage().copy()

# Analyze BF -- can we answer the question without fluorescence?
cells = detect_cells_bf(bf_img)
if len(cells) > 0 and question_is_about_count:
    return len(cells)  # Done! No fluorescence needed.

# Only now switch to fluorescence
core.setConfig(group, "nucleus")
core.snapImage()
nuc_img = core.getImage().copy()
```

**When BF suffices:**
- Cell counting (sparse samples)
- Cell morphology (fibroblasts, worms)
- Wound width measurement
- Sample identification
- Colony counting on agar

**When fluorescence is needed:**
- Specific protein localization
- Nuclear counting in dense tissue
- Live/dead classification
- Calcium dynamics
- Bud scar detection

---

## Principle 1b: Camera Gain

Camera gain amplifies signal electronically after readout. Available via:
```python
core.setProperty("Camera", "Gain", 4.0)  # range 1.0-32.0
```

**Key tradeoff:**
- Gain amplifies signal AND noise equally — does NOT improve SNR
- **Short exposure + high gain** = bright but noisy
- **Long exposure + low gain** = bright with better SNR (more photons)

**When to use gain:**
- Fast timelapse where long exposure causes motion blur
- Photobleaching-sensitive samples where light dose must be minimal
- Dim fluorescence where you need to see the signal quickly

**Default approach:** Prefer longer exposure over higher gain. Only use gain
when exposure is limited by motion blur or bleaching constraints.

---

## Principle 2: Adaptive Exposure

Start with minimal exposure and increase only if signal is insufficient.

```python
def adaptive_snap(core, group, channel, target_snr=10):
    """Snap with adaptive exposure to minimize photon dose."""
    exposures = [10, 25, 50, 100, 200]  # ms, ascending

    for exp in exposures:
        core.setExposure(exp)
        core.setConfig(group, channel)
        core.snapImage()
        img = core.getImage().copy()

        # Estimate signal-to-noise
        signal = np.percentile(img, 95) - np.percentile(img, 5)
        noise = np.std(img[img < np.percentile(img, 25)])
        snr = signal / max(noise, 1)

        if snr >= target_snr:
            return img

    return img  # Return best effort
```

The principle extends to choosing WHICH channels to acquire — skip fluorescence
entirely if BF answers the question.

---

## Principle 3: Temporal Frugality

Acquire frames only as fast as the biology demands.

### Match Frame Rate to Dynamics

| Process Speed | Frame Interval | Justification |
|--------------|----------------|---------------|
| Calcium wave (fast) | Every step | Miss it and it's gone |
| Cell migration | Every 2-5 steps | Slow, smooth motion |
| Wound healing | Every 3-5 steps | Very slow closure |
| Steady state | One frame | Nothing changes |

### Skip Frames During Quiescence

```python
def smart_timelapse(core, group, max_steps=50):
    """Acquire faster during events, slower when stable."""
    prev_img = None
    skip_interval = 1

    for step in range(max_steps):
        core.setConfig(group, "BF")
        core.snapImage()
        img = core.getImage().copy()

        if prev_img is not None:
            # Measure change between frames
            change = np.mean(np.abs(img.astype(float) - prev_img.astype(float)))

            if change < 2.0:
                skip_interval = min(skip_interval + 1, 5)  # slow down
            else:
                skip_interval = 1  # speed up

        prev_img = img

        # Skip frames during quiescence — sleep waits real time on a real
        # microscope; on a simulator where snaps advance the clock, the
        # equivalent is an MDAEvent with min_start_time set.
        import time
        time.sleep(skip_interval * dt)
```

---

## Principle 4: Channel Multiplexing

Not every channel is needed every frame.

### Alternating Strategy

```python
for step in range(max_steps):
    # BF every frame (free, useful for tracking)
    core.setConfig(group, "BF")
    core.snapImage()
    bf = core.getImage().copy()

    # Fluorescence only every 5th frame (expensive, for quantification)
    if step % 5 == 0:
        core.setConfig(group, "GFP")
        core.snapImage()
        gfp = core.getImage().copy()
        quantify_fluorescence(gfp)

    track_cells(bf)
```

### On-Demand Channel Acquisition

```python
# Snap BF to check if fluorescence is warranted
core.setConfig(group, "BF")
core.snapImage()
bf = core.getImage().copy()

# Only snap fluorescence if cells are present
if detect_cells(bf) > 0:
    core.setConfig(group, "nucleus")
    core.snapImage()
    nuc = core.getImage().copy()
```

---

## Principle 5: Z-Stack Optimization

Acquiring full Z-stacks is expensive. Minimize Z slices.

### Equatorial Plane Only

For most measurements, a single in-focus plane suffices:
```python
# Find focus by scanning Z
best_z, best_sharpness = find_focus(core, group, channel="BF")
core.setPosition(core.getFocusDevice(), best_z)
# Now acquire at best focus -- no need for full stack
```

### Sparse Z-Sampling

If 3D information is needed, acquire only slices with signal:
```python
# Quick survey: coarse Z-scan
z_positions = np.linspace(z_min, z_max, 10)
signal_per_z = []
for z in z_positions:
    core.setPosition("ZStage", z)
    core.snapImage()
    signal_per_z.append(np.max(core.getImage()))

# Fine scan only where signal exists
signal_range = [z for z, s in zip(z_positions, signal_per_z) if s > threshold]
# Acquire dense Z only in signal_range
```

---

## Principle 6: Photon-Free Measurements

Some measurements require zero photons:

- **Stage position readout**: `core.getXPosition()`, `core.getYPosition()`
- **Z position**: `core.getPosition("ZStage")`
- **Time elapsed**: frame counter or system clock
- **Hardware state**: `core.getProperty(device, prop)`

Use these for tracking when possible:
```python
# If the stage is moving (e.g., microfluidics flow),
# stage position may provide useful data without a snap
x, y = core.getXPosition(), core.getYPosition()
```

---

## Principle 7: Active Dose Budgeting

A real microscope has a finite photobleaching budget for any given
sample. *On every snap*, ask whether the next exposure adds new
information vs simply consumes budget. The Mandal 2025 sleepwalking-
mode failure ([[Papers/Mandal 2025]]) is
the agent that keeps snapping reflexively until the sample is
bleached past recovery — a strict generalisation of the "skip frames
during quiescence" principle into "skip snaps when no new
information is encoded".

**Decision rules at the snap boundary:**

- **Static stain** (Hoechst-stained nuclei, cell-painted morphology,
  any fixed-cell measurement) → **one snap is enough for a count**.
  Repeated exposures of a static target add no information beyond
  the SNR floor of a well-exposed first frame. ch602 (10/10 r1)
  validated this — 1 snap × 50 ms = 12.5 % of a 400 ms budget gave
  an exact-match cell count.
- **Kinetic / time-resolved** → budget snaps to span the dynamics,
  not to oversample any single time-point. A FRAP recovery doesn't
  benefit from 50 baseline frames; 4 baseline + 25 recovery is
  plenty.
- **Closed-loop scout** → use the wave-period auto-tuner
  (`utils.wave_period.estimate_wave_period`) to scale the
  scout window to ~4 wave cycles, not "as many as possible". A
  GCaMP scout that needs 32 frames does not benefit from 64.
- **Multi-channel** → channels with non-redundant information each
  cost their own budget. Don't snap channels that won't be
  analysed.

**Self-reported budget tracking.** When the simulator (or real
microscope) exposes a hard budget gate, the agent must self-track:
keep an integer counter of snaps × exposure_ms, submit
`final_cum_exposure_ms` honestly. Triggering `BudgetExceededError`
fails the gate.

**Rule of thumb**: leave 50 % budget headroom. ch602's 12.5 %
utilisation isn't always achievable, but 50 % headroom guards
against an unexpected stage-move or focus-recovery snap pushing
into the limit.

## Quick Reference: Gentleness Checklist

Before each snap, ask:
- [ ] Can I answer this with BF instead of fluorescence?
- [ ] Do I need this channel right now, or can I wait?
- [ ] Is the frame rate faster than the biology requires?
- [ ] Am I acquiring Z slices that contain no signal?
- [ ] Could I get this information without a snap at all?
- [ ] Does this snap add **new** information vs the previous one?
- [ ] Am I tracking my cumulative dose against any known budget?

**The ideal experiment acquires the minimum data needed to answer the question
with confidence, and not one photon more.**

## Literature

- [[Papers/Weigert 2018]] — CARE: acquire at ~60× lower dose, 10× axial under-sampling, or 20× faster frame rate and restore the image offline with a trained deep-learning prior. The deep-learning flavour of gentleness — shifts budget from acquisition to inference, at the cost of paired training data per sample class.
- [[Papers/Jin 2020]] — DL-SIM: 5× fewer raw SIM frames and 100× lower photon counts per frame, with a trained network recovering the missing illumination phases. Same deep-learning-prior bargain as CARE, applied to the SIM frame-budget specifically — enables live super-resolution with reduced photobleaching.
- [[Papers/Durand 2018]] — puts photodamage on equal footing with resolution and SNR inside a Pareto / bandit optimiser, so the microscope actively trades illumination dose against image quality instead of treating "gentle" as a hand-tuned hard constraint.
- [[Papers/Bilodeau 2024]] — sim-trained reinforcement learning bakes the resolution-vs-photodamage trade-off into the policy at training time inside a physically faithful STED simulator, so the agent arrives at the live microscope already gentle without paying any exploration dose on the real sample.
- [[Papers/Stepp 2026]] — hybrid-EDA: surveillance runs in label-free phase-contrast (zero excitation budget) and a dynamics-informed neural-network detector triggers fluorescence acquisitions only on rare events. The canonical reference for "Brightfield first" elevated to a closed-loop modality-switching paradigm — fluorescence dose collapses to the events themselves, enabling ~10× longer experiments / ~10× more rare events captured.
