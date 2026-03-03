# Playbook: Hemocytometer Cell Counting

## When to Use
- Cell counting with a ruled counting chamber (Neubauer)
- Viability assessment with trypan blue staining
- Cell concentration calculation

## Background
A hemocytometer is a thick glass slide with a known-depth chamber and an etched grid.
The standard Neubauer grid has a 3x3 arrangement of large squares. The 4 CORNER squares
are used for counting — each has a known area and the chamber depth gives a known volume.

## Step 1: Visual Inspection

```python
img_bf = snap(core, "brightfield")
img_tb = snap(core, "trypan-blue")
from PIL import Image
Image.fromarray(img_bf).save("/tmp/hemo_bf.png")
Image.fromarray(img_tb).save("/tmp/hemo_tb.png")
```

Look for:
- 3x3 grid of large squares with dark grid lines
- Center square has fine subdivisions (NOT used for standard counting)
- Live cells: bright circles with dark halo in BF (phase contrast)
- Dead cells: darker/filled circles in BF; BRIGHT in trypan-blue channel
- Cell clumps (2-4 touching cells)

## Step 2: Define Corner Squares

Divide the image into a 3x3 grid. The 4 corner squares are used for counting:
```python
H, W = img.shape[:2]
sq = W // 3  # each square side
corners = {
    'TL': (0, 0, sq, sq),
    'TR': (W - sq, 0, W, sq),
    'BL': (0, H - sq, sq, H),
    'BR': (W - sq, H - sq, W, H),
}
```

## Step 3: Cell Detection Strategy

**LoG on trypan-blue (BEST — proven ch409=8/10)**:
- ALL cells visible in TB: dead=very bright, live=dim gray (~30 vs background ~15-20)
- LoG blob detection (threshold=5, sigma=3-12) naturally separates closely-spaced cells
- Per-corner ROI detection gives better results (local contrast normalization)
- Avoids BF grid line interference entirely

```python
from skimage.feature import blob_log
# Per-corner detection on TB channel
for name, (r0, r1, c0, c1) in corners.items():
    roi = img_tb[r0:r1, c0:c1].astype(float)
    blobs = blob_log(roi, min_sigma=3, max_sigma=12, threshold=5, num_sigma=8)
    # blobs[:, :2] = (row, col), blobs[:, 2] = sigma
```

**Supplement with BF watershed for dense clumps**:
```python
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

filled = binary_fill_holes(binary).astype(np.uint8)
dist = distance_transform_edt(filled)
coords = peak_local_max(dist, min_distance=4, threshold_abs=1.5, labels=filled)
markers = np.zeros_like(filled, dtype=np.int32)
for i, (r, c) in enumerate(coords):
    markers[r, c] = i + 1
ws_labels = watershed(-dist, markers, mask=filled)
```
- Merge TB + BF detections, deduplicate within 5px

## Step 4: Calculations

```python
total_count = sum(corner_counts)  # [TL, TR, BL, BR]
live_count = total_count - dead_count
viability = live_count / total_count  # fraction 0-1
concentration = (total_count / 4) * dilution_factor * 10000  # cells/mL
```

- Dilution factor: typically 2 (1:2 dilution with trypan blue)
- The 10000 multiplier converts from count/0.1µL to cells/mL

## Common Pitfalls

- **Grid lines vs dead cells**: Both are dark in BF. Dead cells are CIRCULAR, grid lines are LINEAR. Use TB channel for dead cell detection.
- **Clump splitting**: Area-based estimation (area / single_cell_area) is crude. Watershed with min_distance=4 works better.
- **Single cell area**: Typical radius 3-7px → area 28-154px². Median ~75px².
- **Boundary cells** (IMPORTANT): Standard hemocytometer convention — count cells touching LEFT and TOP edges, EXCLUDE cells touching RIGHT and BOTTOM. This prevents double-counting across adjacent squares.
- **Background threshold**: BF background ~175, live cells ~220, dead cells ~130, grid lines ~50-80.

## Lessons Learned
- Apply edge exclusion protocol for boundary cells (left/top in, right/bottom out)
- **LoG on TB channel is primary detector** — avoids BF grid line confusion entirely
- Supplement with BF watershed for dense clumps that LoG under-resolves
- Per-corner ROI detection better than whole-image (local contrast normalization)
- Densest corners need extra care — clumps are where most undercounting occurs
- ch409: 105/114 (8/10), viability 80.0% vs GT 78.9%, concentration 525k vs 570k
