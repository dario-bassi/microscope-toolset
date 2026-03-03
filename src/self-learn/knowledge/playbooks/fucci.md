# FUCCI Cell Cycle Phase Mapping

## Biology
FUCCI (Fluorescent Ubiquitination-based Cell Cycle Indicator) uses two reporters:
- **Cdt1-mCherry** (nucleus-channel): Bright in G1 (Cdt1 accumulates), dim in S/G2/M
- **Geminin-GFP** (geminin-channel): Bright in G2/M (Geminin stable), degrades in G1

| Phase | nucleus-channel | geminin-channel | Detection |
|-------|----------------|-----------------|-----------|
| G1    | BRIGHT (~0.85)  | DIM (~0.05)     | nuc >> gem |
| S     | moderate (~0.45)| moderate (~0.40) | nuc ≈ gem |
| G2    | DIM (~0.08)     | BRIGHT (~0.80)  | gem >> nuc |
| M     | very dim (~0.05)| VERY BRIGHT (~0.90) | gem >> nuc |
| Division | drops | signal drops → G1 level | geminin disappears |

## Standard Multi-Scale Workflow

### Step 1: 10x Survey — Count G2/M cells
```python
set_objective(core, 10)  # Full FOV = 512x512 world px
core.setXYPosition(256.0, 256.0)

nuc = snap('nucleus-channel').astype(float)  
gem = snap('geminin-channel').astype(float)

# Segment cells: use max(nuc, gem) with Otsu threshold
combined = np.maximum(gaussian(nuc, sigma=2), gaussian(gem, sigma=2))
thresh = threshold_otsu(combined)
mask = combined > thresh  # nuclei mask

# Classify G2/M: gem/nuc ratio > 2.5-3.0 AND gem_max > ~15
for each region:
    ratio = gem_max / (nuc_max + 1e-6)
    is_g2m = ratio > 2.5 and gem_max > 15
```

At 10x: image pixel = world pixel (ps=1.0, stage at 256,256).

### Step 2: Navigate to G2/M Cell at 40x
```python
# Select brightest G2/M cell (highest gem_max = most likely M-phase)
best = max(g2m_cells, key=lambda c: c['gem_max'])

# Convert 10x pixel -> world
target_wx = 256.0 + (best['x'] - 256.0) * ps10  # = best['x'] when ps10=1.0
target_wy = 256.0 + (best['y'] - 256.0) * ps10

set_objective(core, 40)  # FOV = 128x128 world px at 40x
core.setXYPosition(target_wx, target_wy)
```

### Step 3: 12-Frame Geminin Timelapse at 40x
```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    time_plan={"loops": 12, "interval": interval_s},
    channels=[{"config": geminin_channel, "group": group}],
)
frames = []
run_events(core, list(seq), on_frame=lambda img, ev: frames.append(img.astype(float)))
```

### Step 4: Division Detection
```python
# Find cell in first frame (near center at 40x)
smooth_first = gaussian_filter(frames[0], sigma=3)
yr, xr = np.unravel_index(smooth_first.argmax(), smooth_first.shape)

# ROI around cell
r = 50  # radius in pixels
cell_mask = np.zeros(frames[0].shape, bool)
cell_mask[max(0,yr-r):yr+r, max(0,xr-r):xr+r] = True

# Compute per-frame intensity
intensities = [f[cell_mask].mean() for f in frames]

# Division = drop to < 40% of baseline
threshold = intensities[0] * 0.40
division_frame = next(
    (i+2 for i, v in enumerate(intensities[1:]) if v < threshold),
    None
)
division_observed = division_frame is not None
```

## Key Parameters
- `G2M_RATIO_THRESHOLD = 2.5` (gem/nuc max ratio)
- `G2M_GEM_MIN = 15` (minimum gem_max to avoid false positives)
- `DIVISION_DROP_THRESHOLD = 0.40` (< 40% of baseline = division)
- `ROI_RADIUS = 50` pixels around brightest spot at 40x

## What to Report
```python
answer = {
    'n_G2M_cells': int,           # from 10x survey
    'target_cell_pos': {'x': wx, 'y': wy},  # world coords
    'division_observed': bool,
    'division_frame': int or None,
}
```

## Pitfalls
- **Don't confuse G1 and G2/M**: G2/M = gem bright. G1 = nuc bright.
- **At 10x ps=1.0**: world_x = pixel_x (stage at center 256,256)
- **At 40x ps=0.25**: FOV is only 128x128 world px — cell should be near center
- **Division signal drop**: Geminin degrades sharply at M→G1. Expect a clear drop
  in 12 frames for M-phase cells. G2 cells may not show division.
- **S-phase cells**: Both moderate — threshold ratio > 2.5 excludes them correctly
- **Edge cells**: May be partially out of FOV at 10x — include in count if centroid visible

## ch515 Lessons (5/10 → fix for R2)

### G2/M Classification Threshold
- **WRONG**: `ratio > 2.5 AND gem_max > 15` → counted 20 cells (GT=9, 2.2x overcount)
- **ROOT CAUSE**: S-phase cells have gem≈0.40, nuc≈0.45 → ratio≈0.89, but scaled to max_gem=92:
  gem_max≈0.40*92=37, nuc_max≈0.45*121=54, ratio=37/54≈0.69 (EXCLUDED by ratio>2.5 ✓)
  BUT some S cells have slightly higher gem that gets caught in the ratio>2.5 net
- **FIX**: Use `ratio > 6` (not > 2.5) to exclude S-phase cells
  OR: require `gem_max > 60` (absolute pixel value, not fraction)
  G2/M cells: gem_max = 0.80*92 ≈ 74-92 → clearly > 60
  S cells: gem_max ≈ 0.40*92 ≈ 37 → clearly < 60
- **ALSO CHECK**: For each gem-bright candidate, verify nucleus-channel pixel < 15

### Target Cell Selection
- The brightest gem_max cell was exactly correct (world 470.7, 263.3 vs GT 470.9, 263.9)
- Sort G2/M cells by gem_max descending and use first one

### Division Detection
- 12 frames may not be enough for G2 cells (they need to complete M-phase)
- M-phase cells (very bright, gem≈0.90) will show clear drop in 12 frames
- G2 cells (gem≈0.80) may not divide in 12 frames → division_observed=False is correct

## ch526/528/529 Critical Discovery (R1→R2 improvement)

### WRONG: Blob detection on nucleus-channel misses G2/M cells
G2/M cells have DIM mCherry nuclei → blob_log on nucleus-channel may MISS them entirely.
Using nucleus-channel blobs as anchors = finding mostly G1+S cells (bright nuclei).
- nucleus-channel `blob_log` detected ratio>5: 0 G2/M cells (none found!)
- Actual G2/M had nuc=7 (too dim for blob detection in background of 120 max)

### CORRECT: Blob detection on geminin-channel, then filter by ratio
```python
# Step 1: Find all geminin-positive spots
blobs_gem = blob_log(gem / gem.max(), min_sigma=2, max_sigma=10, threshold=0.08)

# Step 2: For each blob, sample BOTH channels
for b in blobs_gem:
    r, c = int(b[0]), int(b[1])
    gem_val = gem[r-4:r+5, c-4:c+5].max()  # max in 4px window
    nuc_val = nuc[r-4:r+5, c-4:c+5].max()  # mCherry at same position
    ratio = gem_val / max(nuc_val, 1.0)
    
    if ratio > 5.0 and gem_val > 25:  # G2/M criterion
        g2m_cells.append(cell)
```

### Ratio threshold tuning
From actual pixel values in simulation:
- G2/M: gem≈90-93, nuc≈5-11, ratio≈10-15  ← TRUE G2/M
- S: gem≈37, nuc≈54, ratio≈0.7  ← correctly excluded by ratio>5
- G1: gem≈5, nuc≈109, ratio≈0.05  ← correctly excluded

Threshold of ratio > 5.0 gives ~30-36 cells vs GT of 8-26.
The count is still systematically high → need higher ratio threshold.

### Still-open question: what ratio gives the correct count?
- ch526 GT=13, ch528 GT=8, ch529 GT=26 (each seed different)
- Our measured: ~32-35 with ratio>5
- With ratio>8: estimated ~20-25 cells
- RECOMMENDATION: Use ratio > 8 as default threshold

### Division detection (12 frames)
- Geminin signal monotonically decreases during active division
- ch529: 38% drop over 12 frames (4.18→2.50) → division happening!
- 35-38% drop threshold for division_observed
- ALWAYS check: if signal is still high at fr12, division_frame=None (cell still in G2)

## FUCCI v2 Critical Update (ch533=9/10, ch534=submitted)

### Backend Fix: Step-Function Intensities
The virtual environment was updated to use step-function intensities:
- G1:  gem_norm ≈ 0.05 (very dim)
- S:   gem_norm ≈ 0.25 (moderate)  ← clear gap at 0.65!
- G2:  gem_norm ≈ 0.85 (bright)
- M:   gem_norm ≈ 0.95 (very bright)

### CORRECT Detection Approach (v2)
```python
from scipy.ndimage import label, center_of_mass

# After snapping geminin-channel:
gem_norm = gem / (gem.max() + 1e-6)
binary_G2M = gem_norm > 0.65  # Clean separation: G2M=0.85-0.95, S=0.25

labeled_G2M, n_G2M = label(binary_G2M)

G2M_positions_px = center_of_mass(binary_G2M, labeled_G2M, range(1, n_G2M + 1))
# (row, col) → world (x=col, y=row) at 10x with ps=1.0
```

### Division Detection (v2)
```python
from src.analysis.cell_cycle import detect_fucci_division

result = detect_fucci_division(timelapse, baseline_frames=3, drop_threshold=0.4)
division_observed = result['division_observed']
division_frame = result['division_frame']
```

### OR use the module directly:
```python
from src.analysis.cell_cycle import detect_fucci_g2m

gem_norm = gem / (gem.max() + 1e-6)
result = detect_fucci_g2m(gem, threshold=0.65)
n_G2M = result['n_G2M']
positions = result['positions']  # list of (x, y) world coords
```

### Results Validation
- ch533 (v2): n_G2M=17 vs GT=18 (1 missed), target position <1px error, division at frame 9 → 9/10
- ONE point deducted: didn't use nucleus-channel for ratio confirmation (more robust)
- RECOMMENDATION: Still snap nucleus-channel for context, but primary detection via gem > 0.65

### Robustness tip
- After gem > 0.65 detection, optionally verify with nucleus-channel:
  `gem_nuc_ratio = gem_val / max(nuc_val, 1.0)` should be > 3 for true G2/M
- This handles edge cases where bright debris might pass gem > 0.65

## ch534 v2 Lesson (8/10)

### Division Detection
- ch534: division_observed=False when GT=True (16/18 G2M cells predicted to divide in 12 frames)
- ROOT CAUSE: 12 frames was not enough; geminin drop was subtle
- FIX: Use 20 frames at 40x for reliable division detection
- Also increase sensitivity: drop_threshold=0.5 (50% of baseline, not 40%)
- From GT feedback: "when geminin drops from ~200 to ~50 ADU, that is M→G1 transition"

```python
# v3 division detection
div_result = detect_fucci_division(timelapse, baseline_frames=3, drop_threshold=0.5)

# And use 20 frames:
seq = MDASequence(
    time_plan={"loops": 20, "interval": 0.0},
    channels=[{"config": "geminin-channel", "group": group}]
)
```

### Counting Accuracy
- ch534: n_G2M=15 vs GT=18 (83% accuracy) — within ±5 tolerance
- The ratio filter (gem/nuc > 3.0) helps remove debris but may also remove borderline G2M cells
- If count is too low, consider lowering ratio_min from 3.0 to 2.0

### Summary: v3 Parameters
- threshold=0.65 (gem_norm)
- ratio_min=3.0 (gem/nuc, optional)
- timelapse_frames=20 (not 12!)
- drop_threshold=0.5 (50% of baseline)

## ⚠️ Real Microscope Considerations

### Spectral Bleedthrough
**Simulation assumption:** Perfect color separation; no cross-talk between channels.
**Real hardware:** mCherry (nucleus-channel) and GFP (geminin-channel) have spectral overlap:
- mCherry emits some light in green range (leaks into GFP channel)
- GFP emits negligible red light (less bleedthrough this direction)
- High mCherry expression (especially in G1) contaminates gem measurements

**Impact on G2/M detection:**
- Measured gem values in S/G1 cells may be artificially high
- False positives in gem-based detection
- Ratio thresholds may need adjustment (higher threshold required)

**Adjustments for real hardware:**
- Use **spectral unmixing** if available (most microscopes can apply correction matrices)
- Measure bleedthrough on control samples (high mCherry only, high GFP only)
- Increase ratio threshold: consider `gem/nuc > 5.0–8.0` (not 3.0) for robust G2/M
- Optionally: snap both channels and apply linear unmixing:
  ```python
  gem_corrected = gem - 0.15 * nuc  # subtract cross-talk fraction
  nuc_corrected = nuc - 0.05 * gem  # minimal GFP→mCherry bleedthrough
  ```
- Validate ratio thresholds on known G2/M control cells

### Photobleaching
**Simulation:** Fluorescence is constant throughout experiment.
**Real hardware:** Fluorescence decreases with repeated illumination:
- nucleus-channel (mCherry): moderate bleaching (~5–10% per 10 frames)
- geminin-channel (GFP): faster bleaching (~10–20% per 10 frames)
- Most significant at 40x high-laser-power imaging

**Impact on division detection:**
- Geminin signal decays over timelapse (even without division)
- May confuse photobleaching for cell division
- Drop_threshold becomes sensitive to frame count and laser power

**Adjustments for real hardware:**
- Minimize laser power: use lowest intensity that maintains SNR
- Reduce frame rate or add dark intervals between acquisitions
- Fit exponential decay to first few frames (pre-division) and subtract from later frames
- Increase drop_threshold to account for baseline bleach (e.g., 0.6–0.7 instead of 0.5)
- Use anti-fade mounting medium if possible
- For longer timecourses (>20 frames): consider reference ROI method:
  - Define reference region (non-dividing cell)
  - Normalize target by reference bleaching to remove drift

### Focus Drift
**Real hardware:** Focal plane drifts during extended timelapse (thermal expansion, stage settling).
- At 40x, 5 μm drift = severe signal loss
- Typical drift: 1–5 μm per 10 minutes

**Impact on measurement:**
- Signal intensity drops even without photobleaching
- Division detection threshold becomes unreliable

**Adjustments for real hardware:**
- Use **autofocus** if available (Definite Focus, hardware AF)
- Apply **post-hoc focus correction** (phase correlation between frames)
- For manual correction: skip first 2–3 frames (allow focus stabilization)
- Consider shorter timelapse (12–15 frames at high interval) to minimize drift
