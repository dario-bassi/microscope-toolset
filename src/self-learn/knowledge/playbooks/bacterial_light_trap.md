# Bacterial Light Trap (SLM Phototaxis) Playbook

## Biology / Physics
Bacteria slow down in illuminated areas (Frangipane 2018 speed reduction model).
They don't steer toward light — they slow down, spending more time in bright areas.
- Speed factor: 0.15 (85% slower inside SLM circle)
- Result: bacteria accumulate passively → density inside > density outside
- Expected enrichment: ~3× after 40 steps at 10x

## Standard Workflow

### Step 1: Read Challenge Description for Target
**CRITICAL**: The target position is GIVEN in the challenge description. DO NOT search for it.
Example: "Target region: circle at (300, 256), radius 80 px"
- TARGET_X = 300 (image column)
- TARGET_Y = 256 (image row)
- TRAP_RADIUS = 80 (pixels)

### Step 2: Initial Survey at 10x
```python
set_objective(core, 10)
core.setXYPosition(256.0, 256.0)  # Center = see all bacteria in 512x512 world

gfp_img = snap('nucleus-channel')  # GFP fluorescence channel (NOT brightfield)
```

At 10x: entire 512x512 world fits in ONE FOV. All ~80 bacteria visible.

### Step 3: Count Bacteria (GFP, NOT Brightfield)
```python
from src.workflows.bacteria_trap import detect_bacteria_gfp, compute_enrichment

pos_init, n_init, thresh = detect_bacteria_gfp(
    gfp_img,
    threshold=None,  # Auto-Otsu
    min_area=3, max_area=50
)
```

**Use GFP (nucleus-channel), NOT brightfield (BF)**
- BF phase contrast: dark rods with halos → overcounting (2.3× false positive rate)
- GFP fluorescence: bright rods on dark background → clean threshold → accurate count
- ch470 example: BF detected 181, GFP detected 79, actual ~80

### Step 4: Apply SLM Circle
```python
from src.hardware.core import make_slm_circle

yy, xx = np.ogrid[:512, :512]
slm = ((xx - TARGET_X)**2 + (yy - TARGET_Y)**2 <= TRAP_RADIUS**2).astype(np.uint8) * 255
core.setSLMImage("SLM", slm)
core.displaySLMImage("SLM")
```

**DO NOT re-apply SLM each frame** — once set, it stays active.

### Step 5: Acquire N_FRAMES GFP Frames
```python
from useq import MDASequence
from src.hardware.core import run_events

N_FRAMES = 40  # "After 40 steps: ~3x enrichment"
seq = MDASequence(
    time_plan={"loops": N_FRAMES, "interval": 0.0},
    channels=[{"config": GFP_CHANNEL, "group": group}],
)
trap_imgs = []
run_events(core, list(seq), on_frame=lambda img, ev: trap_imgs.append(img))
```

Each frame = 1 simulation step. 40 frames = 40 steps for full enrichment.

### Step 6: Final Count and Enrichment
```python
pos_final, n_final, _ = detect_bacteria_gfp(gfp_img_final, threshold=thresh)

# Compute enrichment
e_init = compute_enrichment(pos_init, TARGET_Y, TARGET_X, TRAP_RADIUS)
e_final = compute_enrichment(pos_final, TARGET_Y, TARGET_X, TRAP_RADIUS)

enrichment_ratio = e_final['frac_inside'] / e_init['frac_inside']
# OR: e_final['enrichment']  = frac_inside / area_frac (normalized by area)
```

## Key Parameters
```python
TARGET_X = 300    # circle center column (from challenge description)
TARGET_Y = 256    # circle center row (from challenge description)
TRAP_RADIUS = 80  # pixels
N_FRAMES = 40     # illumination steps for ~3x enrichment
GFP_CHANNEL = 'nucleus-channel'  # or 'geminin-channel' if available
```

## Enrichment Computation
```python
area_frac = π * r² / (512 * 512) = π * 80² / 512² ≈ 0.077

# n_inside = bacteria inside circle
# n_total = all bacteria

frac_inside = n_inside / n_total
enrichment = frac_inside / area_frac

# Expected: initial enrichment ≈ 1.0 (uniform distribution)
# After 40 steps: enrichment ≈ 3.0
```

## Grade History
- ch470 (R1): 5/10 — Used BF channel (overcounting 181 vs 80), wrong coords
- ch471 (R2): 3/10 — GFP threshold=2 too low → 18K noise pixels counted as bacteria
- ch472 (R3): 8/10 — GFP>4 + CC (79 detected vs 80 actual), enrichment 2.4×
- ch502 (R1): 7/10 — Enrichment 1.96× (just above 1.5 threshold). Issue: 44 initial vs expected 20 at 20x
  - Used 20x (256px FOV) — didn't see all bacteria!
  - FIX: Use 10x (512px FOV) — ALL bacteria visible
- ch527 (R1): 4/10 — Low enrichment (1.155×). Detection threshold=2.0 wrong, blob parameters too large

## solve_509_bacterial_trap.py Usage
```bash
python scratch/solve_509_bacterial_trap.py 509 http://127.0.0.1:5600
python scratch/solve_509_bacterial_trap.py 510 http://127.0.0.1:5601
python scratch/solve_509_bacterial_trap.py 511 http://127.0.0.1:5602
```

The script reads challenge description target coords (300, 256), r=80.
Adapts channel name from available configs.
Uses P97 vs P95 threshold comparison.

## Pitfalls
1. **Wrong channel**: Always use GFP, NOT BF
2. **Wrong magnification**: Use 10x for full FOV (all bacteria visible)
3. **Target coords**: Read from challenge description, don't search
4. **Threshold too low**: P97 percentile threshold is a good default
5. **Min/max area**: min_area=3 (bacteria are small spots), max_area=50 (at 10x)
6. **Enrichment metric**: Use count-based (n_inside/n_total / area_frac), NOT just brightness ratio

## Bacterial Trap v2 (ch539) — Area-Threshold Counting

### CRITICAL: Use area-threshold, NOT blob_log for rod-shaped bacteria

`blob_log` is designed for ROUND blobs. E. coli are ROD-SHAPED (~3×6px at 10x).
`blob_log` misses many bacteria due to shape mismatch.

### Correct method:
```python
from src.workflows.bacteria_trap import detect_bacteria_area_threshold, count_bacteria_in_circle

# Detect bacteria
n_bact, positions, thresh = detect_bacteria_area_threshold(
    img,
    min_area=8,     # bacteria: 8-120px at 10x
    max_area=120,
    std_multiplier=2.0  # thresh = background + 2*std
)

# Count inside circle
n_in = count_bacteria_in_circle(positions, center_x=300, center_y=256, radius=80)
```

### Enrichment computation
```python
area_frac = np.pi * radius**2 / (512**2)  # = 0.077 for r=80
frac_in = n_in / max(n_total, 1)
enrichment = frac_in / area_frac
```

### Protocol for v2 (ch539)
1. Baseline: 5 snaps, no SLM. Count initial bacteria.
2. Apply SLM circle (area_frac = 7.7%)
3. Accumulate: 50 snaps with SLM active
4. Final count: 3 snaps (median for stability)
5. Report enrichment ratio

### Expected results
- Initial fraction in trap: ~0.077 (area fraction = random distribution)
- After 50 snaps: fraction ~0.24-0.35 (enrichment 3-5×)
- Theoretical max (steady state): 6.7× (= 1/speed_factor = 1/0.15)

### ch539 grade result
- Initial: 269 total, 39 in trap (14.5% — slightly above area_frac = normal random variation)
- After 50 snaps: 345 total, 101 in trap (29.3%)
- Enrichment: 3.82× — within expected 3-5× range!

## Sprint 55: Intensity-Based Enrichment (No Counting Needed)

### CRITICAL LESSON from ch539 grade:
Area-threshold counting (8-120px) was 3× inflated (269 detected vs 87 actual). Each rod-shaped bacterium was fragmented into multiple labeled objects. However, enrichment was correct because the overcounting was systematic (same everywhere).

### Better method: summed fluorescence intensity

```python
from src.workflows.bacteria_trap import measure_bacteria_intensity

# Build circle mask
yy, xx = np.ogrid[:512, :512]
circle_mask = (xx - TARGET_X)**2 + (yy - TARGET_Y)**2 <= TRAP_RADIUS**2

# Measure intensity-based enrichment (no counting artifacts)
result = measure_bacteria_intensity(img, circle_mask)
# enrichment = (intensity_inside / total_intensity) / area_frac
print(f"Enrichment: {result['enrichment']:.2f}×")
```

### Why intensity > counting for rods:
- Total GFP fluorescence ∝ bacterial biomass (proportional to cell number)
- No shape assumption (works for rods, cocci, filaments)
- No fragmentation artifacts (no connected-component issues)
- Robust to clumps (clumped cells still contribute proportional intensity)

### Use counting for:
- Absolute cell count required
- Individual cell tracking
- Cell-by-cell analysis

### Use intensity for:
- Population enrichment ratio (this use case!)
- Any time biomass fraction matters more than exact count
- When detection parameters are uncertain

### Correct counting parameters for rod bacteria:
- min_area=8 px, max_area=120 px at 10x — will overcount (fragmentation)
- Better: min_area=4 px, max_area=50 px (individual fragment size)
- OR: use intensity method to bypass counting entirely

### Updated enrichment formula:
```python
# Count-based (fragment-prone for rods):
enrichment = (n_in / n_total) / area_frac

# Intensity-based (robust to shape):
enrichment = (intensity_in / intensity_total) / area_frac
```

## ⚠️ Real Microscope Considerations

### Bacterial Response Lag
**Simulation assumption:** Bacteria respond instantly to SLM illumination.
**Real hardware:** Bacteria have 100–500 ms phototaxis reaction lag + acceleration ramp time.

**Impact on experiment:**
- At 1 fps: each frame step ≈ 1 second, bacteria response is ~100-500ms lag
- Effective enrichment may develop slower than simulation predicts
- At 10x or faster frame rates: lag becomes negligible

**Adjustments for real hardware:**
- Extend `N_FRAMES` from 40 to 60–80 for full enrichment saturation
- Add `response_lag` parameter (e.g., 200 ms) to predict actual delay
- Consider variable illumination duration per frame if using hardware with adjustable frame intervals
- Validate enrichment kinetics on pilot runs with 5-10 bacteria before committing to full experiments

### SLM Settling Time
**Real SLM devices:** May have 5–20 ms settling time between mask updates.
- For this experiment (frame-rate limited): negligible impact
- But important if rapid SLM switching is needed

### Population Heterogeneity
**Simulation:** All bacteria respond identically to light.
**Reality:** Bacterial response varies (motility, light sensitivity differ between cells).
- Some bacteria may not accumulate even with optimal illumination
- Realistic enrichment: 2–4× (not theoretical 6.7×)
- Repeat experiments to assess population variability
