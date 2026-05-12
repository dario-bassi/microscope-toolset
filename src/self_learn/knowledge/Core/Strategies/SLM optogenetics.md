# SLM Optogenetics

## SLM Basics

`np.uint8` mask matching SLM resolution, in viewport/camera space. 255 = light on, 0 = light off.
Query SLM dimensions from the device or match camera resolution.

### MDA-native approach (preferred)

Use `SLMImage` on each `MDAEvent` — the engine applies the mask automatically:

```python
from useq import MDAEvent, SLMImage

mask = make_slm_circle(target_x, target_y, radius=40)
event = MDAEvent(
    channel={"config": "GFP"},
    slm_image=SLMImage(data=mask, device="SLM"),
)
```

`run_events()` handles `slm_image` natively — the engine applies the mask before each snap.

### Imperative approach (for quick scripts)

```python
from self_learn.hardware.core import apply_slm, snap
apply_slm(core, mask)      # calls setSLMImage + displaySLMImage
img = snap(core, "GFP")
```

**Most common bug**: forgetting SLM mask before `snap()`. The MDA-native approach avoids this.

## Mask Creation

```python
from self_learn.hardware.core import make_slm_circle

mask = make_slm_circle((cx, cy), radius, core=core)  # auto-detects SLM size

# Manual barrier mask
def make_slm_barrier(y_center, width=60, size=512):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[max(0, y_center - width//2):min(size, y_center + width//2), :] = 255
    return mask
```

## SLM Modes

```python
# Check available mode labels first (varies by backend)
allowed = core.getAllowedPropertyValues("SLM-Mode", "Label")
# Typical labels: ("excite", "inhibit") or ("0", "1")
core.setProperty("SLM-Mode", "Label", "excite")   # Activates cells
core.setProperty("SLM-Mode", "Label", "inhibit")   # Suppresses activity
```

Set mode ONCE before the loop unless switching mid-experiment.
Always check allowed labels — backends vary.

## Dose: separate geometry from intensity

**The mask defines geometry; a scalar property defines dose.** On real DMD/SLM
setups the laser power, dwell time, or driver gain is a separate scalar
controllable through a device property — it is *not* encoded in the mask
itself. Treat the binary/uint8 mask as "where", and the scalar property
as "how strong".

```python
# Discover the scalar
props = core.getDevicePropertyNames("SLM")        # backend-specific
# Common names: "intensity", "Power", "Gain", "Exposure"
core.setProperty("SLM", "intensity", 1.0)         # set dose
core.setSLMImage("SLM", binary_mask)              # set geometry
core.displaySLMImage("SLM")
img = snap(core, channel="GCaMP")
```

For a **dose-response curve**, sweep the scalar property log-spaced over
≥ 1.5 decades (e.g. {0.1, 0.3, 1.0, 3.0, 10.0}) and keep the mask fixed.
Sweeping mask values (0.1 → 1.0) instead is a classic anti-pattern: many
backends downsample / threshold the mask to bool internally, collapsing
all non-zero values to a single supra-threshold dose. The data then looks
like noise with one wave at the highest "intensity" — easy to misread as
near-EC50 luck (ch595 r3 lesson). Verify scaling by snapping a small
calibration sweep before fitting Hill.

**Independent FOV per dose** when the prep has a long refractory period
(FHN excitable media: ~10–30 snaps recovery). Move the stage 100+ px
between intensities so each dose hits virgin tissue; otherwise the
response amplitude reflects refractory state, not dose.

**…unless the SLM coords are world-centred and stage moves are no-ops**
(the SLM disc lands at world centre regardless of stage). In that case
prefer a `Reset` state device on the simulator — `setState('Reset', 1);
setState('Reset', 0)` wipes the field cleanly between doses with no
spatial gymnastics.

### Realtime-engine pacing (ch595 r14 lesson)

When a calcium / FHN-style scenario is driven by a wall-clock RT engine
that ticks the PDE between snaps, **the MDA engine's tight event queue
can starve the RT engine of ticks**, leaving the field unevolved under
stim and producing flat ΔF/F across all doses. Snap-loops accidentally
work because HTTP RPC latency provides ~10–50 ms of pacing per snap;
MDA's bulk submission does not.

Fix: add `min_start_time` to each `MDAEvent`, monotonically increasing,
matched to the RT-engine tick rate (10 Hz = 0.1 s/event was the working
value for ch595). The MDA engine respects `min_start_time` for engine-
driven sequencing.

```python
def event_gen():
    t = 0.0
    for intensity in INTENSITIES:
        core.setState("Reset", 1); core.setState("Reset", 0)
        for _ in range(K_BASELINE):
            yield MDAEvent(channel={"config": "GCaMP"},
                           slm_image=SLMImage(data=zero_mask, device="SLM"),
                           min_start_time=t)
            t += 0.10
        for _ in range(K_FIRE):
            yield MDAEvent(channel={"config": "GCaMP"},
                           slm_image=SLMImage(data=disc(intensity), device="SLM"),
                           min_start_time=t)
            t += 0.10
```

ch595 r13 (no `min_start_time`) → flat ΔF/F ~ 0.1 across all doses.
ch595 r14 (with 0.1 s pacing) → clean dose-response, EC50 0.0070 vs
GT 0.008 (log10 diff 0.06 ≪ tolerance 0.30) → 9/10.

## Phototaxis (Volvox) — OADA approach

Colony swims TOWARD light. Place mask at TARGET position, not on colony.

```python
from useq import MDAEvent, SLMImage
from self_learn.workflows.mda import adaptive_phase_events

mask = make_slm_circle((target_x, target_y), radius=40)
slm = SLMImage(data=mask, device="SLM")

base = [MDAEvent(channel={"config": "GFP"}, slm_image=slm) for _ in range(30)]

def on_decide(image, event, state):
    pos = detect_colony(image)
    if distance(pos, (target_x, target_y)) < 10:
        state.stop = True

gen, on_frame, state = adaptive_generator(base, on_decide=on_decide)
run_events(core, gen(), on_frame=on_frame)
```

Phototactic turning rate depends on light intensity and colony size — calibrate per experiment.
Apply the SLM mask before the first acquisition frame.

## Calcium Wave Barrier

```python
# Use the human label rather than a state index — the "1"→"Inhibit" mapping
# is sim-specific; on a real setup you might switch modes with a label like
# "inhibition-mode" or toggle a shutter.
core.setProperty("SLM-Mode", "Label", "inhibit")
mask = make_slm_barrier(barrier_y, width=60)
slm = SLMImage(data=mask, device="SLM")

events = [MDAEvent(channel={"config": "GFP"}, slm_image=slm) for _ in range(n_steps)]
run_events(core, events)
```

Apply barrier BEFORE wavefront arrives. Lead time = (distance to barrier) / (wave speed); measure wave speed on your prep (radius vs frame on a test pulse) rather than assuming a number. Applying after the wave passes does nothing.

## Photoconversion

Irreversible. Apply mask once, cells stay converted forever.

```python
slm = SLMImage(data=make_slm_circle((cx, cy), radius), device="SLM")
events = [
    MDAEvent(channel={"config": "GFP"}, slm_image=slm),   # Conversion frame
    MDAEvent(channel={"config": "GFP"}),                    # Verify next frame
]
run_events(core, events)
```

Conversion may not be visible same frame -- check NEXT frame.

## Closed-Loop SLM with OADA

For dynamic masks that change each frame (e.g., tracking a moving target):

```python
def on_decide(image, event, state):
    target = detect_target(image)
    new_mask = make_slm_circle(target, radius=30)
    # Inject event with updated SLM mask
    state.extra_events.append(
        MDAEvent(channel={"config": "GFP"},
                 slm_image=SLMImage(data=new_mask, device="SLM"))
    )

gen, on_frame, state = adaptive_generator(
    [MDAEvent(channel={"config": "GFP"}, slm_image=initial_slm)],
    on_decide=on_decide, max_frames=50,
)
```

## ChR2 + GCaMP functional imaging

Neurons co-expressing ChR2 + GCaMP. SLM activates ChR2 → depolarization → Ca²⁺ influx → GCaMP brightens.

```python
# Key check: GCaMP shares a GFP-like filter with most green structural markers.
# Before interpreting a bright frame as "calcium firing", verify the pulse is
# actually driving the response — apply SLM on a test neuron, snap, compare to baseline.

# Protocol: baseline (no SLM) → stimulation (with SLM) → recovery (no SLM)
# Use min_start_time for proper temporal spacing.
# ΔF/F = (stim - baseline) / baseline.
# Saturated ChR2 drive typically gives several-hundred-percent ΔF/F at the
# targeted cell; off-target cells should stay at ~0%. Calibrate the exact number
# on your prep — ChR2 density, GCaMP variant, and laser power all matter.
```

See `../../Recipes/Neurons.md` for sample-tuned parameters.

## Activation threshold — "no response" is diagnostic

ChR2 (and all opsins) is a **threshold process**: the membrane either reaches firing threshold and depolarises fully, or it stays sub-threshold and nothing happens. You don't get a weak wave from weak stimulation — you get no wave at all.

Before reporting "no propagation", rule out sub-threshold stimulation:

1. **Integrated dose**: photons delivered = `intensity × exposure × n_pulses`. Doubling exposure may be the difference between 0 and full response.
2. **Mask size**: a too-small mask may excite too few cells to seed a propagating wavefront. Try ~30–60 µm radius (hundreds of cells worth) before blaming the opsin.
3. **Wavelength**: ChR2 peaks at ~470 nm (blue); red-shifted opsins (ChRmine, Chrimson) peak at ~590 nm. Wrong colour → order-of-magnitude lower activation.
4. **Calibration snap**: image the opsin-tag channel (mCherry-tagged ChR2 usually has its own channel) to confirm expression before interpreting functional readouts.

In excitable-media models (FitzHugh-Nagumo, Hodgkin-Huxley-like), the activation variable `u` must cross a fixed threshold (e.g. u > 0) for wavefront propagation. If stim_strength × duration integrates to less than (threshold − resting), the medium relaxes back silently. **Calibrate threshold-crossing empirically before running the experiment protocol.**

## Common Pitfalls

- **Mask dtype**: Must be `np.uint8`. Other dtypes fail silently.
- **Coordinate space**: Camera pixels, NOT world coordinates.
- **Stage movement**: Recompute mask positions after moving stage.
- **Reactive vs proactive**: For fast processes, intervene BEFORE the event arrives.
- **Wrong calcium channel**: GCaMP uses GFP filter → may appear in structural marker channel, not a dedicated calcium channel. Always verify by test stimulation.
- **Calcium decay**: Wait ≥10s between experiment phases for full decay to baseline.
- **Sub-threshold stimulation** (see above): no response ≠ sim/hardware bug; always check dose × threshold first.

## Literature

- [[Papers/Passmore 2025]] — paradigm paper for closed-loop optogenetics: the SLM pattern is recomputed each frame from image analysis so the microscope steers cell behaviour (migration, nucleocytoplasmic transport) toward a predefined outcome.
- [[Papers/Lugagne 2024]] — deep-MPC optogenetic control: a learned forward model rolls candidate light-input sequences over a short horizon, the SLM applies the predicted-best input, and the loop tracks arbitrary per-cell gene-expression reference trajectories across thousands of bacteria in parallel.
