# Gentle Imaging

Minimize photon dose while maximizing information. Every unnecessary photon
is phototoxicity with no benefit. The gentlest experiment that answers the
question is the best experiment.

> **Note**: Code examples below use conceptual pseudocode. All multi-frame
> loops MUST use `run_events(core, generator(), on_frame=callback)` in practice.
> Single-frame acquisitions use `snap(core, channel=ch)` from `src.hardware.core`.

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

        # Skip frames during quiescence
        for _ in range(skip_interval - 1):
            core.snapImage()  # advance time without recording
            _ = core.getImage()
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

## Quick Reference: Gentleness Checklist

Before each snap, ask:
- [ ] Can I answer this with BF instead of fluorescence?
- [ ] Do I need this channel right now, or can I wait?
- [ ] Is the frame rate faster than the biology requires?
- [ ] Am I acquiring Z slices that contain no signal?
- [ ] Could I get this information without a snap at all?

**The ideal experiment acquires the minimum data needed to answer the question
with confidence, and not one photon more.**
