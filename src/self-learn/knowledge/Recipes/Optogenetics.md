# Playbook: Optogenetics & SLM Pattern Control

## When to Use
- SLM (Spatial Light Modulator) targeting of cells/regions
- Light-activated protein stimulation (channelrhodopsins, LOV domains)
- Patterned illumination for selective activation/inhibition
- Phototaxis steering, bacterial light traps, reaction-diffusion control

## SLM Setup (CRITICAL)

### Standard SLM Pattern
```python
import numpy as np

# Always verify SLM device
slm = core.getSLMDevice()  # typically "SLM"
core.setSLMDevice("SLM")  # ensure it's set

# Check SLM dimensions
w = core.getSLMWidth("SLM")
h = core.getSLMHeight("SLM")
# Usually 512x512

# Create mask: numpy uint8 array (0=off, 255=on)
slm_mask = np.zeros((512, 512), dtype=np.uint8)

# Circular target at (col, row) with radius r
yy, xx = np.ogrid[:512, :512]
target_col, target_row = 300, 256  # x=300, y=256
slm_mask[((yy - target_row)**2 + (xx - target_col)**2) <= r**2] = 255

# Apply and display
core.setSLMImage("SLM", slm_mask)  # numpy array, NOT .tobytes()
core.displaySLMImage("SLM")
```

### Key SLM Rules
- **Coordinate convention**: "center at (300,256)" means x=300=col, y=256=row
- **Pass numpy array** to setSLMImage, NOT `.tobytes()` — proxy handles serialization
- **Always call displaySLMImage** after setSLMImage
- **Re-apply mask each frame** if needed (some scenarios clear between snaps)
- SLM is in **camera/viewport space** (pixel coordinates)

### SLM Modes (if available)
```python
# Check available modes
labels = core.getStateLabels("SLM-Mode")  # e.g. ('excite', 'inhibit')

# Switch mode
core.setState("SLM-Mode", 0)  # excite
core.setState("SLM-Mode", 1)  # inhibit
```

## Common Optogenetics Workflows

### 1. Bacterial Light Trap (ch509-514, ch539=10/10)
- Use **intensity-based enrichment** as primary metric
- `measure_bacteria_intensity()` from `src.recipes.bacteria_trap`
- Formula: `enrichment = (intensity_in / intensity_total) / area_frac`
- SLM circle centered on trap, bacteria fluoresce when illuminated
- See `scratch/solve_509_v3_bacterial_trap.py` for template

### 2. Phototaxis Steering
- Light stimulus drives organism movement
- SLM creates directional gradient
- Track organism position relative to light
- Use `track_multiframe()` for centroid tracking

### 3. Optogenetic Protein Activation
- SLM targets specific cells/regions
- Before/after comparison of marker intensity
- Use fixed threshold for temporal comparison
- Multiple channels: activation channel + readout channel

### 4. Reaction-Diffusion Pattern Control
- PDE systems (Gray-Scott, Turing) can be modulated
- SLM excite = increase feed rate → nucleate patterns
- SLM inhibit = increase kill rate → suppress patterns
- Measure: coverage = fraction above threshold in region
- Compare stimulated vs control regions

## Measurement Patterns

### Coverage (fraction of active pixels)
```python
from skimage.filters import threshold_otsu
threshold = threshold_otsu(img.astype(np.uint8))
coverage = (img > threshold).sum() / total_pixels
```

### Region-Specific Coverage
```python
yy, xx = np.ogrid[:512, :512]
roi = ((xx - cx)**2 + (yy - cy)**2) <= r**2
cov_in = (img[roi] > threshold).mean()
cov_out = (img[~roi] > threshold).mean()
```

### Enrichment
```python
area_frac = roi.sum() / (512 * 512)
intensity_in = img[roi].sum()
intensity_total = img.sum()
enrichment = (intensity_in / intensity_total) / area_frac
```

## Protocol Template
```python
# 1. Baseline (no SLM)
clear_mask = np.zeros((512, 512), dtype=np.uint8)
core.setSLMImage("SLM", clear_mask)
core.displaySLMImage("SLM")
baseline_img = snap(core, channel)

# 2. Stimulation
core.setState("SLM-Mode", mode)  # 0=excite, 1=inhibit
core.setSLMImage("SLM", slm_mask)
core.displaySLMImage("SLM")
# Acquire N frames
for i in range(n_stim_frames):
    time.sleep(interval)
    core.snapImage()
    stim_img = core.getImage()

# 3. Recovery (clear SLM)
core.setSLMImage("SLM", clear_mask)
core.displaySLMImage("SLM")
for i in range(n_recovery_frames):
    time.sleep(interval)
    core.snapImage()
    recovery_img = core.getImage()
```

## Key Lessons

1. **MDA generator + SLM control between yields** — preferred pattern over snap loops
2. **SLMImage in MDA events fails over pymmcore-proxy** — use manual `setSLMImage`/`displaySLMImage`
3. **Keep SLM ON during cascade observation** — sustained stimulation lets downstream neurons respond
5. **Re-apply SLM mask** each frame if needed (set before yielding each event)
6. **Use fluorescence channel** for quantification, not brightfield
7. **Fixed threshold** across all timepoints for temporal comparison
8. **Separate ROI and control** — always measure both stimulated and unstimulated regions
9. **Check SLM effect** before full experiment — does coverage actually change?
10. **Detect a frozen sim from the frames themselves** — historically `Camera.SimTime` exposed the simulator clock, but that virtual property was removed 2026-04-27. To detect a non-advancing backend now, compare two consecutive snaps: if pixel-wise mean and a small spot-difference are bit-identical across N frames, treat the sim as frozen and submit target positions from pre-fire image analysis (e.g. lowest local-std regions = quiescent) rather than trying to verify each nucleation. (ch591 RD 2026-04-24 lesson.)
11. **Simultaneous multi-spot SLM >> sequential firing for N-target nucleation.** Sequential firing (fire target 1 → observe → fire target 2 → observe → …) wastes simtime between firings: by the time you fire target N, targets 1..N-1 have been propagating for a while and disturb the baseline. Instead: compose a SINGLE SLM mask with all N target discs illuminated simultaneously, reapply for 50+ snaps, then read the post-fire state once. On ch591 this moved nucleation success from 1/6 → 6/6 in one iteration. The multi-spot pattern is also what a real adaptive microscope would do — SLMs natively display arbitrary shapes, so firing multiple targets per frame costs nothing extra. (ch591 RD 2026-04-25 lesson; 10/10 converged.)

## Connectivity Mapping Pattern

### Protocol (confirmed working)
Per neuron stimulation cycle:
1. **Baseline** (SLM off) — all neurons at resting intensity
2. **Sustained SLM stimulation + rapid imaging** — target fires, keep SLM on
   while imaging at fast intervals to catch cascade propagation (direct
   connections respond after ~1 synaptic delay, indirect after ~2)
3. **Recovery** (SLM off) — wait for calcium to decay to baseline

### Detection
- Direct (1-hop) connections: ~6x baseline increase (ΔF/F > 2.0)
- `infer_connectivity_dff(state, n, dff_threshold=2.0, direct_only=True)`
- Filter indirect: they respond later than direct connections

### Code
```python
# MDA generator pattern with SLM control between yields
def connectivity_events():
    for stim_idx in range(n_neurons):
        cx, cy = soma_positions[stim_idx]
        # Baseline (SLM off)
        core.setSLMImage("SLM", clear_mask)
        core.displaySLMImage("SLM")
        yield MDAEvent(channel=ch_kw, exposure=50,
                      metadata={'neuron': stim_idx, 'phase': 'baseline'})
        # SLM on for 3 observation snaps
        slm_mask = make_slm_circle((cx, cy), radius=20, size=512)
        core.setSLMImage("SLM", slm_mask)
        core.displaySLMImage("SLM")
        for si in range(3):
            yield MDAEvent(channel=ch_kw, exposure=50,
                          metadata={'neuron': stim_idx, 'phase': 'slm_on', 'snap_idx': si})
        # Decay (SLM off, 8 snaps)
        core.setSLMImage("SLM", clear_mask)
        core.displaySLMImage("SLM")
        for d in range(8):
            yield MDAEvent(channel=ch_kw, exposure=50,
                          metadata={'neuron': stim_idx, 'phase': 'decay'})

run_events(core, connectivity_events(), on_frame=on_frame)
```
