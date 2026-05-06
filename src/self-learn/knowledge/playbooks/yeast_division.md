# Playbook: Yeast Cell Division Tracking

## When to Use
- Budding yeast (S. cerevisiae) timelapse
- Tasks: count cells, track division events, measure growth rate / doubling time
- Channels: BF for morphology, GFP (membrane) for cleaner cell detection

## Key Learnings

### Use GFP Channel, Not BF
- BF threshold (e.g., < 110) is too strict for tiny yeast cells — misses ~50% at high density
- GFP (membrane-channel) gives bright rings on dark background — much cleaner
- Global threshold > 10 works well for GFP yeast

### Splitting Budding Pairs: Watershed
Budding cells form pairs that merge into single connected components.
Use watershed segmentation to split:

```python
from skimage import measure, morphology, segmentation
from scipy import ndimage
from skimage.feature import peak_local_max

binary = img_gfp > 10
binary = morphology.opening(binary, morphology.disk(1))
distance = ndimage.distance_transform_edt(binary)
coords = peak_local_max(distance, min_distance=4, threshold_abs=2, labels=binary)
markers = np.zeros_like(binary, dtype=int)
for i, (y, x) in enumerate(coords, 1):
    markers[y, x] = i
markers = morphology.dilation(markers, morphology.disk(1))
ws = segmentation.watershed(-distance, markers, mask=binary)
cells = [p for p in measure.regionprops(ws) if p.area > 5]
```

### FOV Matters
- Clarify which FOV the count refers to (full sample vs visible field)
- Higher magnification sees fewer cells but with more detail
- Use low mag to count ALL cells, higher mag for morphological detail

### Doubling Time Calculation
```python
import math
ratio = n_end / n_start
t_double = total_frames * math.log(2) / math.log(ratio)
```

### Frame Budget
- Multi-channel acquisition means multiple snaps per timepoint
- Keep track of total snaps and elapsed time for accurate timing

## Timelapse Protocol
1. Set 20x objective
2. Snap GFP frame 0, count cells
3. For each subsequent frame: snap BF (visual) + GFP (counting)
4. Track count increase over 15-20 frames
5. Compute: n_start, n_end, n_divisions = n_end - n_start, doubling_time

### Avoid Watershed Oversplitting
- min_distance=4 in peak_local_max oversplits budding pairs → overcount by ~6
- Better: min_distance=7 for yeast at 20x (cells are ~8-10px diameter)
- Also filter by circularity: buds attached to mothers are elongated (low circularity)
- Or use area filter: each split region should be > ~30px area at 20x

## Lessons Learned
- BF threshold alone misses ~50% of yeast at high density — use fluorescence
- Watershed with min_distance too small oversplits budding pairs → ~6 cell overcount
- Growth dynamics (divisions, doubling time) are robust even when absolute count is off
