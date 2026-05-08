# Playbook: Brightfield Cell Analysis

## When to Use
- Sample shows cells on a plain background in brightfield
- Tasks: count cells, measure areas, classify morphology, find brightest/largest

## Step 1: Visual Inspection

Snap and view the image before any code:

```python
from src.core.hardware.core import snap
img = snap(core, "brightfield")
from PIL import Image; Image.fromarray(img).save("/tmp/preview.png")
# View /tmp/preview.png with Read tool
```

Ask yourself:
- Are cells sparse or touching? (touching -> may need watershed or nucleus channel)
- Are cells bright or dark relative to background? (BF cells are typically bright rings)
- Is the focus good? (blurry -> try autofocus first)
- What magnification? (check `get_config(core).pixel_size_um`)

## Step 2: Detection

```python
from src.core.detection.cells import detect_cells
cells = detect_cells(img, threshold_sigma=2.0, min_area_px=30, fill_holes=True)
```

**Parameter tuning by visual assessment:**
- Sparse, high-contrast cells: `threshold_sigma=2.5`
- Dense, moderate contrast: `threshold_sigma=1.5-2.0`
- Very dim cells: `threshold_sigma=1.0` (risk: more noise)
- `fill_holes=True` is essential for BF (cells appear as rings)

**If detection looks wrong:**
1. Save overlay image, visually inspect detected vs actual cells
2. Try different sigma (1.5, 2.0, 2.5, 3.0)
3. If cells are touching, try `count_objects_dt()` on the binary mask
4. If available, switch to nucleus fluorescence for cleaner counting

## Step 3: Area Measurement

- BF areas are systematically underestimated by ~17% at sigma=2.0
- For more accurate area: use `threshold_sigma=1.5` (captures more boundary)
- Area at 10x: typical single cell = 800-1000 px (800-1000 um^2)
- Area at 40x: same cell = 12800-16000 px (800-1000 um^2)
- Always report `area_um2` (pixel_size-corrected), not raw `area_px`

## Step 4: Morphology Classification

Use the shape descriptors from `detect_cells()`:
- `circularity`: 1.0 = perfect circle, <0.7 = elongated/irregular
- `solidity`: 1.0 = convex, <0.8 = blebbed/irregular membrane
- `eccentricity`: 0.0 = round, >0.5 = elongated

**Visual confirmation with LLM vision:**
For subjective classifications (healthy/apoptotic/mitotic), save the cell region and use visual inspection rather than relying solely on computed descriptors.

## Step 5: Multi-Position Scanning

If the sample is larger than one FOV:

```python
from src.core.workflows.scanning import scan_and_detect_mda
from src.core.hardware.core import run_events

gen, on_frame, state = scan_and_detect_mda(
    world_positions, pixel_size=pixel_size_um,
    threshold_sigma=2.5, min_area_px=20, channels=("brightfield",),
)
run_events(core, gen(), on_frame=on_frame)
cells = state["world_positions"]
```

- Overlap tiles by 30%+ to avoid missing edge cells
- Deduplicate with hierarchical clustering at 30-35 px threshold
- Convert all positions to world coordinates before submission

## Population Statistics

- Use **median/MAD** for population stats — mean/std inflated by outliers
- MAD = `median(|xi - median|) * 1.4826` gives outlier-robust std estimate
- Area underestimation worsens for larger cells — threshold cuts proportionally more boundary

## Confluency Estimation

- Membrane channel fill_holes: overestimates ~15-20% (line thickness)
- BF fill_holes: underestimates ~17% (threshold cuts boundary)
- **Geometric mean of both** ≈ ground truth: `sqrt(membrane_pct * bf_pct)`
- Cell count: use BF connectedComponents (membrane merges touching cells)

## When NOT to Use Brightfield

**BF is unreliable for counting small objects** (bacteria, small particles):
- Phase contrast halos create ghost detections at cell boundaries
- Ch470 lesson: BF detected 181 "bacteria" vs actual 80 (2.3x overcount)
- Ch552 lesson: BF phase contrast unreliable for run-tumble tracking

**Use fluorescence instead** when:
- Counting bacteria (bright-on-dark is unambiguous)
- Counting small particles or foci
- Any quantitative count where false positives matter

**Use BF for**:
- Finding large cells (>5µm) for navigation
- Overview/scout images before fluorescence
- Confluency estimation (with membrane channel cross-check)

## Use parameter_advisor for Detection Parameters

```python
from src.core.analysis.parameter_advisor import suggest_detection_params, parameter_report
params = suggest_detection_params(ps, object_type='nucleus')
print(parameter_report(params))
# Use params['min_area_px'], params['threshold_sigma'] etc.
```

## Common Pitfalls

- Forgetting `fill_holes=True` -> cells detected as thin rings with tiny areas
- Using pixel coordinates instead of world coordinates at 40x
- Area underestimation not accounted for in quantitative comparisons
- Edge cells (centroid within 50px of border) have truncated area -> exclude for morphometry
- **Using BF for bacteria/small cell counting** — use fluorescence channel instead
