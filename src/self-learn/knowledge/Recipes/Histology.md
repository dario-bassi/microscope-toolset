# Histology / H&E Stained Tissue Playbook

## Sample Characteristics

- H&E staining: **Hematoxylin** (blue-purple) stains nuclei/DNA; **Eosin** (pink) stains cytoplasm/stroma
- RGB brightfield images (3-component uint8)
- Fixed tissue — no dynamics, no bleaching concerns

## Color Deconvolution

### Standard: skimage `rgb2hed()`
```python
from skimage.color import rgb2hed
hed = rgb2hed(img_rgb)
h = hed[:, :, 0]  # hematoxylin (nuclei) — higher = more stain
e = hed[:, :, 1]  # eosin (cytoplasm/stroma)
d = hed[:, :, 2]  # DAB (if IHC)
```

### Alternative: LAB color space
```python
from src.core.detection import extract_hematoxylin
hema = extract_hematoxylin(img_rgb)  # nuclei bright
```

## Nuclear Segmentation

1. Extract hematoxylin channel
2. Otsu threshold → binary mask
3. Remove small objects (min_size=30 at 10x, 100 at 20x)
4. Fill holes
5. Distance transform + peak_local_max → markers
6. Watershed split touching nuclei

```python
from skimage import filters, morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

h_thresh = filters.threshold_otsu(h)
nuc_mask = h > h_thresh
nuc_mask = morphology.remove_small_objects(nuc_mask, min_size=30)
nuc_mask = ndimage.binary_fill_holes(nuc_mask)
dist = ndimage.distance_transform_edt(nuc_mask)
coords = peak_local_max(dist, min_distance=8, labels=nuc_mask)
markers = np.zeros_like(nuc_mask, dtype=int)
for i, (r, c) in enumerate(coords, 1):
    markers[r, c] = i
labeled = watershed(-dist, markers, mask=nuc_mask)
props = measure.regionprops(labeled)
```

## Tumor Grading (Nottingham System)

### Three components (each scored 1-3):

| Component | Score 1 | Score 2 | Score 3 |
|-----------|---------|---------|---------|
| **Tubule formation** | >75% well-formed | 10-75% | <10% |
| **Nuclear pleomorphism** | Q75/Q25 < 2 | Q75/Q25 2-4 | Q75/Q25 > 4 |
| **Mitotic count** | Index < 0.03 | 0.03-0.10 | > 0.10 |

**Total → Grade**: 3-5 = Grade 1, 6-7 = Grade 2, 8-9 = Grade 3

### Gland Architecture Assessment
- Use nuclear cluster solidity: **solidity < 0.75** = ring-like gland with lumen
- **solidity >= 0.75** = solid nest (no lumen = poorly differentiated)
- Ratio of ring glands / total large clusters → tubule formation score

### Nuclear Pleomorphism
- **CV (coefficient of variation)** of nuclear areas from watershed segmentation
- **Q75/Q25 area ratio**: best single discriminator
- Use multiple FOVs at 20x for representative sampling

### Mitotic Index
- Mitotic figures: condensed chromosomes, small-medium size, high stain, irregular shape
- Criteria: area < 800, hematoxylin > 65th percentile, eccentricity > 0.75 OR solidity < 0.80
- Confirm at 40x when possible

## Necrosis Detection (CRITICAL — ch430 lesson)

**DO NOT skip this. Always run explicit necrosis analysis.**

Necrosis in H&E:
- **Eosinophilic acellular regions** — bright pink, no intact nuclei
- **Ghost nuclei** (karyorrhexis) — pale, fragmented nuclear remnants
- Different from lumens (which are white/pale with no eosin stain)

### Detection algorithm:
1. Extract eosin channel from HED deconvolution
2. Find regions with **moderate-to-high eosin** AND **low hematoxylin**
3. Check that these regions are NOT just gland lumens (lumens have very low eosin)
4. Key: necrosis has ELEVATED eosin (ghost cells); lumens are simply empty

```python
hed = rgb2hed(img)
h, e = hed[:,:,0], hed[:,:,1]
# Necrotic: elevated eosin in nuclear-desert zones
h_thresh = filters.threshold_otsu(h)
nuc_mask = h > h_thresh
nuc_dilated = ndimage.binary_dilation(nuc_mask, iterations=5)
# Regions without nuclei but WITH eosin stain (not empty white)
nuclear_free = ~nuc_dilated
e_thresh = np.percentile(e[e > 0], 50)  # above-median eosin
necrotic = nuclear_free & (e > e_thresh)
necrotic = morphology.remove_small_objects(necrotic, min_size=200)
necrosis_present = necrotic.sum() > 500  # significant area
```

**Lesson from ch430=7**: I checked necrosis by comparing eosin in acellular zones to overall mean, but the threshold (1.5×) was too strict. Necrosis patches have MODERATE eosin elevation, not extreme. Use absolute threshold (above-median eosin) in nuclear-free zones instead.

## Multi-Magnification Workflow

1. **10x**: Architecture overview — count glands, assess pattern, check necrosis
2. **20x**: Nuclear segmentation — areas, pleomorphism (CV, Q75/Q25), 4-tile survey
3. **40x**: Mitotic figure confirmation — verify condensed chromosome morphology

## Additional Features to Report

- **Lymphocyte infiltrate**: Small, dark, round nuclei (smaller than tumor nuclei)
- **Stromal desmoplasia**: Dense collagen around glands
- **Invasion pattern**: Crisp gland borders (pushing) vs infiltrative

## Challenge History
- ch415=8: 40 nuclei, 2 dysplastic identified correctly
- ch430=7: Grade 2 correct, necrosis MISSED (needed explicit eosin analysis)
