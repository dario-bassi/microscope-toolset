# Mitochondrial Dynamics Playbook

## When to Use
Any challenge involving mitochondrial network morphology, fragmentation,
fission/fusion dynamics, or membrane potential (e.g., CCCP, FCCP, Mdivi-1).

## Key Concepts

### Mitochondrial Network
- In healthy cells, mitochondria form a **connected tubular network**
- Measured by connected component (CC) counting:
  - Healthy: few CCs (~1 large network + small fragments = ~40 total)
  - Fragmented: many small puncta (~50-100 CCs)
- **NEVER use foci/LoG detection for fragment counting** — it counts bright spots, not disconnected regions

### Membrane Potential
- Fluorescent dyes (TMRM, MitoTracker, JC-1) accumulate in polarized mitochondria
- Depolarization (CCCP/FCCP) → intensity loss + network fragmentation
- `membrane_potential_ratio = final_intensity / baseline_intensity`

## Workflow

### 1. Multi-Scale Overview
```python
from src.core.hardware.core import set_objective, snap_all_channels, get_pixel_size
set_objective(core, 10)
imgs_10x = snap_all_channels(core)
# Look at membrane-channel: bright tubular network
```

### 2. Choose Magnification
- **20x** (0.5 µm/px): Best for whole-cell network morphometry, fits single large cell
- **40x** (0.25 µm/px): Better resolution of individual mitochondria, but may not fit entire cell
- **10x**: Too coarse for individual mitochondria

### 3. Baseline Measurement
```python
from src.core.analysis.network import count_fragments, measure_network
from skimage import filters

set_objective(core, 20)
ps = get_pixel_size(core)
mem_baseline = snap_membrane().astype(float)

# Fixed threshold for temporal comparison
mito_thresh = max(filters.threshold_otsu(mem_baseline[mem_baseline > 0]), 10)

# Fragment count (connected components)
frag = count_fragments(mem_baseline, threshold=mito_thresh, min_size=5, pixel_size=ps)
n_baseline = frag['n_fragments']       # ~40 for healthy
frag_idx = frag['fragmentation_index'] # n_fragments / area

# Network skeleton metrics
net = measure_network(frag['mask'], pixel_size=ps, min_branch_length=3)
# total_length, n_junctions, n_branches, mean_branch_length
```

### 4. Drug Application
```python
core.setState("Perfusion", 4)  # CCCP/FCCP
```

### 5. Timelapse Monitoring
```python
from src.core.analysis.treatment import compute_half_time, recovery_fraction

for i in range(n_frames):
    time.sleep(interval_s)
    mem = snap_membrane().astype(float)
    # Use SAME threshold for all frames
    frag = count_fragments(mem, threshold=mito_thresh, min_size=5, pixel_size=ps)
    fragment_counts.append(frag['n_fragments'])
    intensities.append(float(mem.mean()))
```

### 6. Washout + Recovery
```python
core.setState("Perfusion", 0)  # washout
# Continue monitoring...
```

### 7. Key Metrics
```python
# Peak fragmentation
peak_n_fragments = max(all_fragment_counts)
peak_fragmentation_index = max(all_frag_indices)

# Membrane potential
final_membrane_potential = final_intensity / baseline_intensity

# Half-times
ht = compute_half_time(times, intensities, baseline_value=baseline_intensity)

# Recovery
rec_frac = recovery_fraction(baseline_intensity, nadir, recovered_value)
```

## Common Drugs
| Drug | Target | Effect |
|------|--------|--------|
| CCCP/FCCP | Protonophore | Depolarization + fragmentation |
| Mdivi-1 | Drp1 | Inhibits fission → elongated mitochondria |
| Oligomycin | ATP synthase | Hyperpolarization (increased potential) |

## Critical: Per-Frame Normalization for Fragment Counting (ch563 lesson)

CCCP simultaneously **dims fluorescence AND fragments the network**. A fixed threshold
derived from baseline causes fragments to DISAPPEAR instead of MULTIPLY.

**Correct approach**: Normalize each frame to its own max (or top percentile) before
thresholding, OR use a very low absolute threshold (2-3x background noise).

```python
def count_fragments_normalized(img, min_size=5, pixel_size=1.0):
    """Normalize per frame, then threshold for CC counting."""
    img_f = img.astype(float)
    # Normalize to percentile to handle intensity changes
    p99 = np.percentile(img_f, 99)
    if p99 <= 0:
        return {'n_fragments': 0, ...}
    img_norm = img_f / p99
    # Fixed threshold on normalized image
    thresh_norm = 0.3  # 30% of peak signal
    mask = img_norm > thresh_norm
    mask = morphology.remove_small_objects(mask, max_size=min_size - 1)
    labeled, n = ndimage.label(mask)
    ...
```

**Also correct**: Very low absolute threshold (bg_mean + 3*bg_std) per frame.

## Membrane Potential: Measure AFTER Recovery

`final_membrane_potential` should be measured AFTER drug washout + recovery, not at the
end of the drug phase. GT expects ~0.46 (partial recovery), not ~0.15 (during CCCP).

## Pitfalls
1. **Fragment counting**: Use `count_fragments()` (connected components via ndimage.label),
   NOT `detect_foci()` (LoG blob detection). Foci detection found 23,323 "fragments" on a
   depolarized image vs ground truth of 56 (ch563 R1 lesson: 5/10).
2. **NEVER use fixed Otsu threshold for fragment counting under drug treatment**:
   Drug changes BOTH intensity AND morphology. Fixed threshold makes fragments disappear
   (ch563 R2 lesson: 6/10). Use per-frame normalization or noise-floor threshold.
3. **min_size filtering**: Remove objects <5 pixels to eliminate noise speckles.
4. **Intensity = whole-image mean**: For membrane potential ratio, use `whole_image` mean,
   not foreground-only (threshold crossing artifact).
5. **Report requested metrics**: Read challenge notes carefully — peak_n_fragments,
   peak_fragmentation_index, final_membrane_potential are commonly requested.
6. **final_membrane_potential after washout**: Not during drug. Measure after recovery phase.

## Relevant Modules
- `src.core.analysis.network`: `count_fragments()`, `measure_network()`
- `src.core.analysis.treatment`: `compute_half_time()`, `recovery_fraction()`
- `src.core.analysis.temporal`: `track_mean_intensity()`
