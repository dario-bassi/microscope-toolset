# Blood Smear — WBC Differential Count

**Assumes:** Wright-Giemsa stained smear (RGB), 100x for nucleus-lobe morphology, 10x for survey. Paired with `src/recipes/blood_smear_wbc.py` (`detect_and_classify_wbc_v2` — 2-lobe size cutoff at 0.75× large_wbc_area; tested on ch590).

## Sample Type
Wright-Giemsa stained blood smear. RBCs are small gray discs, WBCs are larger with visible nuclei.

## Workflow
1. **10x survey**: Snap nucleus channel at field center. WBC nuclei appear as bright spots (threshold above background).
2. **Detect WBCs**: Connected components on thresholded nucleus channel. Expect ~20 WBCs per field.
3. **100x classification**: For each WBC, move stage to world coords, snap BF + nucleus at 100x.
4. **Measure features**: From central 200x200 ROI:
   - `n_lobes`: Connected components in nucleus channel (area >= 5px)
   - `cell_area`: Dark region in BF (cell is darker than background)
   - `nuc_area`: Total nuclear pixel area
   - `nc_ratio`: nuc_area / cell_area
   - `nuc_eccentricity`: Largest nuclear component eccentricity
5. **Classify** using decision tree below.

## Classification Decision Tree (100x)

```
IF n_lobes == 1:
    IF cell_area > 18000:     → MONOCYTE (largest WBC, kidney nucleus)
    ELSE:                     → LYMPHOCYTE (small, round, high N:C)
ELIF n_lobes == 2:
    IF nuc_area > 2500 AND eccentricity > 0.5:
                              → EOSINOPHIL (bilobed, large granules)
    ELSE:                     → NEUTROPHIL (2-lobed variant)
ELIF n_lobes >= 3:
                              → NEUTROPHIL (multi-lobed, most common)
```

## Key Morphology at 100x

| Type | Size | Nucleus | Cytoplasm | Frequency |
|------|------|---------|-----------|-----------|
| Neutrophil | Medium (10-14µm) | 2-5 dark lobes | Pale, fine granules | ~60% |
| Lymphocyte | Small (7-10µm) | Single round, dark | Thin rim | ~30% |
| Monocyte | Large (15-20µm) | Kidney/bean-shaped | Abundant, gray | ~5% |
| Eosinophil | Medium (12-15µm) | Bilobed | Large bright granules | ~5% |

## Critical Lessons

- **Cell SIZE is the primary discriminator** at high magnification, not just lobe count
- Relative ordering: Lymphocyte (smallest) < Neutrophil (medium) < Monocyte (largest)
- Lymphocytes: highest N:C ratio (nucleus fills most of cell)
- Monocytes: largest overall cell area
- Eosinophils: 2 lobes + distinctively high nuclear eccentricity
- **Don't use lobe count alone** — 2-lobed neutrophils are common
- The nucleus channel at high magnification cleanly separates lobes (better than BF)
- *Note: Specific pixel thresholds depend on magnification and camera — calibrate per setup*

## RGB Wright-Giemsa (Color Imaging)

When the microscope provides RGB color images:
- **Nucleus color**: Purple (R~75, G~40, B~100). Darker purple = lymphocyte. Lighter purple = monocyte.
- **Eosinophil cytoplasm**: ORANGE (R>210, G<165, B<120) — uniquely identifiable by color alone
- **RBC color**: Salmon pink (R~200, G~170, B~150)
- **Target cells**: Concentric ring pattern. At 10x: detect via high internal intensity range (>55)
  within the cell body (normal RBCs have lower variance). At 40x: bull's-eye clearly visible.
- **Sickle cells**: Dark brown crescents, eccentricity >0.92 and dark intensity (<170)
- At 10x, use binary dilation (4 iterations) to merge multi-lobed nuclei before labeling

## 10x-Only Classification (when higher magnification unavailable)

At 10x, WBCs are 5-10px. Classify using:
1. **Nucleus area** (most discriminating): monocyte >> neutrophil > lymphocyte
2. **Lobe count** (after dilation merge): 3+ = neutrophil, 2 = neutrophil or eosinophil, 1 = lymphocyte or monocyte
3. **Nucleus darkness** (BF darkest pixel RGB): (45,20,70) = lymphocyte, (75,40,100) = neutrophil, (90,55,110) = monocyte
4. **Cytoplasm color** (BF around nucleus): orange = eosinophil (only type with orange)

### ch461 Refined Decision Tree (10x, nucleus channel + BF RGB)
Tested approach that achieved exact match on 20-WBC differential:
```
detect: LoG(sigma=3) + detect_cells(threshold_sigma=2) → merge_nearby(12px)
patch: 25×25px around each WBC

# High-threshold lobe detection (threshold > 80 on nucleus channel)
IF nuc_R > nuc_B + 80 in BF:           → EOSINOPHIL (orange nucleus)
ELIF n_high_lobes >= 2:                → NEUTROPHIL (multi-lobed)
ELIF nuc_area >= 60:                   → MONOCYTE (largest nucleus)
ELIF nuc_area >= 40 AND solidity < 0.85: → MONOCYTE (irregular shape)
ELIF nuc_B < 100 in BF:               → LYMPHOCYTE (very dark nucleus)
ELIF nuc_area < 45 AND solidity > 0.85: → LYMPHOCYTE (small, compact)
ELSE:                                  → NEUTROPHIL (default)
```

### RBC Morphology (ch461 method)
- **Sickle cells**: eccentricity > 0.95 AND solidity < 0.85 AND 25 < area < 120
- **Target cells**: eccentricity < 0.5 AND solidity > 0.85 AND top_25_intensity - bottom_50_intensity > 20
- Exclude pixels near WBC positions (±10px) to avoid false detections

## Sample Size Adequacy
After counting WBCs across FOVs, check if you have enough for reliable proportions:
```python
from src.core.analysis.statistics import estimate_required_n
# Example: 12 neutrophils out of 20 WBCs so far
check = estimate_required_n(12, 20, target_ci_width=0.10)  # ±10%
if not check['sufficient']:
    print(f"Need ~{check['required_n']} total WBCs, have {20}")
    # Acquire more FOVs
```
- WHO recommends 100+ WBCs for a differential; 200+ for rare cell types
- Proportions <5% (eosinophils, basophils) need larger samples for ±2% CI

## Pitfalls
- Lobe-counting thresholds must scale with magnification (100x nuclei are huge)
- 2-lobed cells default to neutrophil (most common), not eosinophil
- Colonies merging at high magnification can confuse cell boundaries
- At 10x, RBCs are only 6-8px — target cell concentric pattern barely resolvable
- Y-axis may be flipped between pixel and world coordinates — verify empirically
- **MONOCYTE must be a distinct category** — don't merge into neutrophil! (ch454 lesson)
  - Monocyte = LARGEST WBC + kidney-shaped nucleus + LIGHTEST purple nucleus at 10x
  - At 10x: largest area_10x AND highest purple_sum (lightest nucleus) = monocyte
- **RBC measurement**: use Canny/Hough circles, NOT thresholding (misses rim)
- **pixel_size at 100x**: verify from hardware — formula 10/mag may be wrong (0.125 not 0.1)
- **Consistency**: ensure per-cell type list matches summary differential counts
