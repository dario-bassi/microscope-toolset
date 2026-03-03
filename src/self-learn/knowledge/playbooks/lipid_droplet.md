# Lipid Droplet & Hepatic Steatosis Playbook

## Sample Types
- **Hepatocytes** (liver cells): steatosis disease model
- **Adipocytes** (3T3-L1): adipogenesis differentiation assay
- **Non-adipose cells**: drug-induced lipid accumulation

## Fluorescent Dyes
| Dye | Filter | Notes |
|-----|--------|-------|
| BODIPY 493/503 | GFP (488/510) | Gold standard, membrane-channel |
| LipidTOX Green/Red | GFP / mCherry | Membrane-channel / nucleus-channel |
| Nile Red | Yellow/green | Broad spectrum, use GFP or Texas Red filter |
| Oil Red O | BF only | Fixed cells, bright field, not compatible with live imaging |

BODIPY 493/503 stains **neutral lipids** (triglycerides). Signal is in membrane-channel (GFP filter).

## Key Microscopy Parameters
- **Magnification**: 20x or 40x for droplet morphometry; 10x for population-level fraction
- **Channel order**: First snap nucleus (DAPI/nuc-ch) for cell finding, then BODIPY for lipid
- At 10x (1 µm/px): lipid droplets are 1-5 px; use LoG detection (min_sigma=0.5, max_sigma=3)
- At 40x (0.25 µm/px): droplets are 4-20 px; use LoG (min_sigma=1, max_sigma=8)

## Steatosis Classification Workflow

### Step 1: Detect all cells (nucleus channel)
```python
from skimage import filters, measure, morphology
from scipy import ndimage

smooth = sk_gaussian(nuc_img, sigma=2)
thresh = filters.threshold_otsu(smooth)
nuc_mask = smooth > thresh
nuc_mask = ndimage.binary_fill_holes(nuc_mask)
labeled_cells, n = ndimage.label(nuc_mask)
```

### Step 2: Define cell body regions (dilate nuclei)
```python
# Hepatocytes: cell ~2-3x nucleus diameter
DILATION = 8  # px at 10x; ~15 px at 20x
cell_regions = {}
for label_id in range(1, n+1):
    nucleus = labeled_cells == label_id
    cell_body = ndimage.binary_dilation(nucleus, iterations=DILATION)
    cell_regions[label_id] = cell_body
```

### Step 3: Measure BODIPY per cell
```python
per_cell_intensity = []
for label_id, region in cell_regions.items():
    bodipy_vals = bodipy_img[region]
    per_cell_intensity.append({
        'label': label_id,
        'mean': bodipy_vals.mean(),
        'p90': np.percentile(bodipy_vals, 90),
        'max': bodipy_vals.max(),
    })
```

### Step 4: Classify steatotic/normal
```python
intensities = np.array([c['mean'] for c in per_cell_intensity])
# Use Otsu for bimodal split (steatotic = high intensity)
threshold = filters.threshold_otsu(intensities)
n_steatotic = sum(c['mean'] > threshold for c in per_cell_intensity)
steatotic_fraction = n_steatotic / len(per_cell_intensity)
```

## Clinical Steatosis Grading (per-cell)
From `src/analysis/lipid_droplet.py` `steatosis_index()`:
- Grade 0 (normal): lipid_fraction < 5%
- Grade 1 (mild): 5-20%
- Grade 2 (moderate): 20-50%
- Grade 3 (severe): > 50%

**Steatotic cell** (for population-level fraction): grade ≥ 1, or use bimodal threshold on BODIPY mean intensity.

## Using lipid_droplet.py

```python
from src.analysis.lipid_droplet import (
    detect_lipid_droplets, droplet_morphology, droplet_count_per_cell,
    lipid_content_score, steatosis_index, lipid_droplet_analysis, population_lipid
)

# Per-cell analysis at 40x
for cell_mask in cell_masks:
    result = lipid_droplet_analysis(bodipy_40x, cell_mask,
                                    min_sigma=1.0, max_sigma=8.0, threshold=0.08)
    grade = result['steatosis_grade']  # 0-3
    n_droplets = result['n_droplets']
    frac = result['lipid_fraction']

# Population summary
summary = population_lipid(all_per_cell_results)
```

## Common Pitfalls

1. **Wrong channel for lipid detection**: BODIPY is in GFP filter (membrane-channel), NOT nucleus-channel
2. **Using BF for lipid quantification**: BF shows dark cells on bright background, oil droplets appear as dark spots — use fluorescence always
3. **Nucleus ≠ cell body**: nuclei are ~1/3 the area of a hepatocyte. Dilate nucleus mask to get cell body
4. **Bimodal assumption failure**: if all cells are steatotic (high-fat diet model), Otsu will still split them. Cross-validate with background intensity as a "normal" reference
5. **LoG scale mismatch**: at 10x, droplets are tiny. Use min_sigma=0.5. At 40x, droplets are larger — min_sigma=1.5

## Steatotic Fraction Reporting
- `steatotic_fraction = n_steatotic / n_cells_total`
- Tolerance typically ±2 cells OR ±15% (use grading criteria from challenge notes)
- Always report both count AND fraction

## Checklist
- [ ] Connected to correct server
- [ ] Snapped ALL channels at 10x first
- [ ] Confirmed which channel is BODIPY (GFP filter = membrane-channel)
- [ ] Detected all cells (nucleus channel)
- [ ] Expanded nucleus to cell body region
- [ ] Measured per-cell mean BODIPY intensity
- [ ] Applied Otsu threshold on cell intensities
- [ ] Reported: n_steatotic, n_normal, n_cells, steatotic_fraction
- [ ] Saved showcase with cell map and intensity histogram
