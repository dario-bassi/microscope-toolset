# Playbook: Yeast Bud Scar Analysis (Calcofluor)

## When to Use
- Calcofluor-stained yeast cells
- Tasks: count bud scars per cell, identify virgin vs budded cells, determine replicative age

## Step 1: Visual Inspection

```python
img = snap(core, "fluorescence")  # or appropriate channel
from PIL import Image; Image.fromarray(img).save("/tmp/yeast.png")
```

Look for:
- Bright cell walls (calcofluor stains chitin)
- Small bright puncta on cell periphery (bud scars)
- Uniform wall fluorescence with NO puncta (virgin cells)

## Step 2: Cell Segmentation

Segment individual yeast cells:
```python
from src.core.detection.cells import detect_cells
cells = detect_cells(img, threshold_sigma=1.5, min_area_px=100, fill_holes=True)
```

## Step 3: Bud Scar Detection (CRITICAL)

**Must use per-cell relative thresholding, NOT global threshold:**

For each cell:
1. Extract the cell's wall pixels (periphery, not interior)
2. Compute that cell's wall baseline brightness
3. Bud scars = puncta brighter than the cell's own wall baseline

```python
# Per-cell approach:
for cell in cells:
    wall_pixels = get_periphery_pixels(img, cell['mask'])
    wall_baseline = np.median(wall_pixels)
    wall_std = np.std(wall_pixels)
    scar_threshold = wall_baseline + 2.0 * wall_std
    scars = wall_pixels > scar_threshold
    # Count connected bright puncta that are small (1-3px)
```

**True bud scars are:**
- Small bright puncta (typically a few pixels across, varies with magnification)
- Located at the cell PERIPHERY (not interior)
- Brighter than the cell's own wall baseline

## Step 4: Virgin Cell Identification

- ~50% of yeast population are virgin cells (never budded)
- Virgin cells have uniform wall fluorescence with NO bright puncta
- If no puncta exceed the per-cell threshold -> virgin cell

## Common Pitfalls

- **Global thresholding**: Background wall fluorescence varies between cells. A global threshold produces massive false positives.
- **Background wall = bud scars**: NOT TRUE. Uniform wall staining is normal calcofluor binding, not bud scars.
- **Interior bright spots**: Bud scars are on the PERIPHERY. Interior bright spots are artifacts.
- **Overcount from touching cells**: Where two cells touch, the wall is doubly bright -- not a bud scar.
