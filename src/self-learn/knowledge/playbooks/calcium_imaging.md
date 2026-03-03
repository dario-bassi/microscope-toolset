# Calcium Imaging Playbook

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
from src.hardware.core import snap
from src.detection import detect_cells

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
from src.analysis.temporal import detrend_signal
result = detrend_signal(trace, method='polynomial')  # quadratic bleach correction
clean_trace = result['detrended']
```

### 3. Extract Frequency/Period
```python
from src.analysis.temporal import extract_period, find_dominant_frequency

# For periodic signals
period = extract_period(clean_trace, sampling_rate=fps)
print(f"Period: {period['period']:.2f}s, Frequency: {period['frequency']:.4f} Hz")

# For detailed spectral analysis
fft = find_dominant_frequency(clean_trace, sampling_rate=fps)
print(f"Dominant freq: {fft['dominant_freq']:.4f} Hz, Amplitude: {fft['amplitude']:.2f}")
```

### 4. Detect Peaks (Transients)
```python
from src.analysis.temporal import detect_peaks_in_trace

peaks = detect_peaks_in_trace(clean_trace, min_distance=int(fps * 0.5))
print(f"Found {peaks['count']} transients, mean interval: {peaks['mean_interval']/fps:.2f}s")
```

### 5. Classify Dynamics
```python
from src.analysis.temporal import classify_signal_dynamics

dynamics = classify_signal_dynamics(trace, sampling_rate=fps)
# Returns: 'oscillatory', 'decaying', 'recovering', 'transient', or 'static'
```

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

## SLM + Calcium Waves

See `knowledge/workflows/slm_optogenetics.md` for barrier/excitation patterns.
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
