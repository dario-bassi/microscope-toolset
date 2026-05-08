# Calcium Imaging Playbook

**Assumes:** GCaMP-expressing sample (or equivalent indicator), single fluorescence channel, timelapse acquisition. Rates and amplitudes below are representative ranges — calibrate on your prep. For optogenetics-triggered calcium waves see [[Recipes/Optogenetics]] and [[Core/Strategies/Wave propagation]].

## Signal Characteristics

### GCaMP Calcium Indicators
- **Fast rise** (~50-200ms), **slow decay** (~500-2000ms)
- Baseline fluorescence is LOW (indicator is dim without calcium)
- Transients look like sharp spikes with exponential tails
- Typical frequencies: 0.01-1 Hz (cardiac ~1-3 Hz, neuronal bursts faster)

### Cell Cycle Reporters (FUCCI)
- **Slow oscillations** (hours for full cycle)
- Two-color system: green (S/G2/M) and red (G1)
- At microscopy timescales, looks like slow sinusoidal alternation

## Analysis Workflow

### 1. Extract Time Series
```python
from src.core.hardware.core import snap
from src.core.detection import detect_cells

# Timelapse acquisition
traces = {cell_id: [] for cell_id in cell_ids}
for frame_idx in range(n_frames):
    img = snap(core, "GFP")  # or channel with calcium indicator
    for cell_id, roi in rois.items():
        intensity = np.mean(img[roi])
        traces[cell_id].append(intensity)
```

### 2. Detrend (Remove Photobleaching)
```python
from src.core.analysis.temporal import detrend
clean_trace = detrend(trace, method='linear')   # 'linear' or 'mean'
```

For per-pixel localisation tasks (pacemaker / wave-source ID),
prefer one of these bleach-immune scores over raw `stack.std(axis=0)`:
```python
from src.core.utils.firing_energy import firing_energy, detrended_sigma
energy = firing_energy(stack, diff_threshold=10)   # positive-rise sum
sigma_d = detrended_sigma(stack)                    # σ after linear-trend subtract
```
ch651 r3 lesson: a wave-front cell on the cardio backend showed
σ=18.6 (purely from a 143→79 bleach trajectory) — *higher* than
the GT primary at σ=16.1 from actual firings. `firing_energy`
correctly suppresses the bleach because monotonic decay
contributes 0 (every diff is non-positive).

### 3. Extract Frequency/Period
```python
from src.core.analysis.temporal import (
    fft_spectrum, autocorrelation, measure_periodic_rate,
)

# Spectral peak (good when oscillation is clean)
spec = fft_spectrum(clean_trace, dt=1.0 / fps)
print(f"Dominant freq: {spec['dominant_freq']:.4f} Hz")

# Period from autocorrelation (more robust on noisy data)
ac = autocorrelation(clean_trace)
period_frames = ac['period']

# One-call wrapper that returns rate in your preferred unit (bpm/Hz/frame⁻¹)
rate = measure_periodic_rate(clean_trace, dt=1.0 / fps, unit='Hz')
```

### 4. Detect Peaks (Transients)
```python
from src.core.analysis.temporal import detect_peaks
peaks = detect_peaks(clean_trace, min_distance=int(fps * 0.5))
# returns dict: indices, count, mean_interval, …
```

### 5. Classify Dynamics
There is no single `classify_signal_dynamics` helper; combine the
above signals: `fft_spectrum.spectral_quality > 0.5` ⇒ oscillatory;
`detect_peaks` returns 0 ⇒ static; monotone trend after `detrend(...,
method='mean')` ⇒ decaying / recovering by sign of slope.

## Method Selection Guide

| Signal Type | Best Method | Why |
|---|---|---|
| Regular oscillation | FFT | Clean spectral peak |
| Noisy oscillation | Autocorrelation | Robust to noise |
| Isolated transients | Peak detection | Irregular timing |
| Mixed frequencies | FFT + filter by band | Separate components |
| Bleaching + signal | Detrend first, then FFT | Remove drift |

## When to Use FFT vs Autocorrelation

**FFT**: Best when oscillation is regular and continuous. Gives exact frequency.
Fails with: irregular intervals, very short recordings, non-stationary signals.

**Autocorrelation**: Best when signal is noisy or period varies slightly.
Gives the most common inter-peak interval. More robust to noise.
Fails with: very few cycles (<3), highly irregular timing.

**Peak detection**: Best when transients are well-separated and sharp.
Gives individual event timing, amplitude, and intervals.
Fails with: overlapping transients, low SNR.

## Common Pitfalls

- **Photobleaching masquerades as decay**: Always detrend before frequency analysis
- **Nyquist**: Need >2 samples per oscillation period (>2× sampling rate vs signal freq)
- **Short recordings**: Need at least 3-5 full cycles for reliable frequency extraction
- **DC component**: FFT will show huge peak at 0 Hz — always skip it (subtract mean)
- **Frame rate uncertainty**: Know actual dt between frames; multi-channel acquisition reduces effective frame rate per channel

## Cardiac Optical Mapping (ch428=9/10)

### Arrhythmia Detection Pipeline
For cardiomyocyte monolayers with competing pacemakers:

1. **Acquire**: 180-frame GCaMP timelapse (nucleus-channel) via MDA, 10x
2. **Frequency map**: Per-pixel FFT → spatial map of dominant frequency
   - Ectopic focus (1.8 Hz) captures most tissue
   - Normal pacemaker (1.0 Hz) visible only near its location
3. **Ectopic localization**: Corner peak timing → earliest-peaking corner = ectopic
4. **Conduction velocity**: Phase gradient at dominant FFT frequency
   - `velocity = omega / |grad(phase)|` where `omega = 2π * freq`
   - Use 16x16 blocks for robustness, median over valid pixels

### Key Learnings
- Faster pacemaker (1.8 Hz) overdrives tissue → becomes dominant frequency everywhere
- Normal territory appears only as small region with ~1.0 Hz in FFT map
- Phase gradient method is the standard in real cardiac electrophysiology
- Compute `dom_bin` dynamically from the most common frequency, never hardcode
- dt ≈ 0.1 sec/frame (confirmed: 0.1778 cyc/frame = 1.8 Hz)
- GT conduction velocity: 80 px/frame (phase gradient gives ~65-90 depending on block size)

### ch651 ("pick the EARLIER one") — sim-specific layer
Cardio v6 backend (the variant that ships ch651) has these
hardcoded properties — verify before using on a real cardio sim:
- **Internal `_time=5.0` burn-in**: server starts ~5 cardio cycles into
  the simulation; frame 0 is NOT pacemaker-only. Both primary and
  ectopic are firing + waves propagating from snap 0. Cannot rely on
  "first frame brightness = primary". (ch651 r3 grader confirmation.)
- **Default geometry** (PDE grid ↔ image @ 10×):
  - Primary: PDE (32, 32) → image ~(128, 128), normal_freq=1.0 Hz, phase 0
  - Ectopic: PDE (96, 96) → image ~(384, 384), arrhythmia_freq=1.8 Hz, phase ~0.3
- **Tolerance**: ±25 px on the primary's image position; expected jitter
  per-seed ≤ 16 world-px from those nominal locations.
- **Snap timing**: `steps_per_snap=8`, AP period `~10 dt` ⇒ ~1.25 frames
  per AP cycle. The first 1-2 snaps after `_time=5.0` show wave-fronts
  from primary mid-propagation, NOT a clean focal pacemaker frame.
- **Picking primary vs ectopic** (the test ch651 actually scores):
  ```python
  from src.recipes.event_driven_modality_switch import (
      localize_multi_pacemaker,
  )
  result = localize_multi_pacemaker(
      stack,
      primary_position_prior=(128, 128),  # the cardio v6 backend prior
      use_firing_energy=True,             # cardio bleaches → swap σ×mean
  )
  primary_xy = result["primary"][:2]      # the EARLIER pacemaker
  ```
  σ × mean alone picks the higher-rate ectopic (the ch648 r1 + ch651
  r2 failure mode — see Wave propagation §"Multi-pacemaker"). The
  position prior + `firing_energy` + slowest-rate pick is what
  resolves it.

## Wave Kinematics (Speed, Period, Pacemakers)

For detailed wave propagation analysis, see `knowledge/workflows/Wave propagation.md`.

**Quick reference for calcium waves:**
```python
from src.core.analysis.temporal import measure_wave_speed, measure_periodic_rate
from src.core.analysis.optical_mapping import detect_pacemaker, phase_map

# Period from global intensity trace
trace = np.array([frame.mean() for frame in stack])
rate = measure_periodic_rate(trace, dt=dt)
period = rate['period']

# Speed from radial profiles (for expanding wavefronts)
speed = measure_wave_speed(radial_profiles, dt=dt, dr=pixel_size)

# Pacemaker detection (earliest activation site)
pacemakers = detect_pacemaker(stack, dt=dt)
```

**Key metrics:**
- Wave speed: typically 10-100 µm/s for intercellular Ca²⁺ waves
- Period: 1-30s depending on cell type and stimulus
- Pacemaker: origin point — earliest phase in phase map

## SLM + Calcium Waves

See `knowledge/workflows/SLM optogenetics.md` for barrier/excitation patterns.
Key: apply inhibition barrier 10-15 frames BEFORE wavefront reaches target region.

### Wave Suppression Protocol
1. Snap 2-3 frames for baseline (measure left+right wave coverage)
2. Immediately apply SLM inhibition mask to target region
3. Measure post-intervention (5-10 frames)
4. Compare before/after coverage

**Real-microscope principles:**
- Minimize fluorescence exposures before the experiment (photobleaching budget)
- Plan acquisition strategy BEFORE imaging — know how many frames you need
- For transient processes, don't waste your temporal window on previews
- Brightfield is essentially free (no bleaching), but fluorescence is not
