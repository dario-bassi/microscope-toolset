# Failure Analysis: Cell Segmentation with Out-of-Focus Artifacts

## Challenge Type
Segment fluorescent cells (mCherry) in images contaminated by bright out-of-focus artifacts.

## What Went Wrong

### Error 1: Bright artifacts mistaken for cells
- **Error**: Initial segmentation on mCherry channel detected bright spots as cells. These were out-of-focus artifacts from a different Z plane (e.g., prism reflection or debris).
- **Diagnostic**: Artifacts appear as small, very bright spots without corresponding DAPI signal. Real cells have both mCherry (cytoplasm) and DAPI (nucleus).
- **Fix**: Use DAPI+mCherry co-localization to confirm real cells before segmentation.

### Error 2: Adaptive threshold captured edges only
- **Error**: Local adaptive thresholding on mCherry produced donut/ring shapes (cell edges detected but interiors missed).
- **Root cause**: Cell interiors have moderate, relatively uniform intensity. Adaptive threshold responds to local contrast, which is highest at cell edges.
- **Fix**: Use global threshold on locally-normalized image instead.

### Error 3: binary_fill_holes merged touching cells
- **Error**: Applied `scipy.ndimage.binary_fill_holes` to the cell mask globally. Touching cells formed a connected region, and fill_holes filled the gaps between them into one giant blob (coverage jumped from reasonable to 54% single component).
- **Fix**: Abandoned fill_holes approach. Used Voronoi watershed from nuclei seeds instead, which naturally separates touching cells.

### Error 4: Otsu threshold skewed by artifacts
- **Error**: After local contrast normalization (`img / gaussian_blur(img, sigma=50)`), artifacts had values up to 8.3x. Otsu threshold landed at 1.66, capturing only 3.8% of image.
- **Fix**: Clip normalized image to max 2.0 before computing Otsu. This prevents extreme outliers from biasing the threshold.

### Error 5: Re-labeling fragmented territories
- **Error**: After clipping Voronoi territories to cell mask, ran `measure.label()` which re-labeled connected components → 1686 tiny fragments instead of 138 cells.
- **Fix**: Preserve original Voronoi labels. Just zero out pixels outside the mask without re-labeling: `labels[~mask] = 0`.

## Correct Approach (Voronoi Watershed from Nuclei)

```
1. Detect nuclei from DAPI channel:
   - Background subtract: gaussian(dapi,2) - gaussian(dapi,60)
   - Threshold at 0.35 * otsu
   - Clean (remove_small_objects, remove_small_holes), label, filter by size

2. Co-localize with mCherry to remove artifact nuclei:
   - Normalize mCherry: gaussian(mch,2) / clip(gaussian(mch,50), min=1)
   - Keep nuclei where mean mCherry intensity in nucleus region > 0.75

3. Voronoi watershed on mCherry gradient (NO mask argument):
   - gradient = sobel(gaussian(mch_norm, sigma=2))
   - markers = nuclei labels (renumbered 1..N)
   - territories = watershed(gradient, markers)  # fills entire image

4. Clip to cell body mask:
   - cell_mask = mch_norm > (otsu_on_clipped * 0.85)
   - territories[~cell_mask] = 0  # preserve original labels

5. Keep only nucleus-connected component per territory:
   - For each label: find connected components, keep only the one overlapping the original nucleus
   - This removes disconnected fragments
```

## Key Insights

- **Multi-channel co-localization** is essential when one channel has artifacts. If a nucleus (DAPI) overlaps a cell body (mCherry), it's a real cell.
- **Local contrast normalization** (`img / gaussian_blur(img, large_sigma)`) flattens uneven illumination but amplifies artifacts. Clip extreme values before thresholding.
- **Voronoi watershed** from nuclei seeds is robust for touching cells. Each nucleus gets a territory, clipped to the cell body mask.
- **Never re-label** after clipping a Voronoi/watershed result — preserve original labels and just zero out unwanted pixels.
- **Deep learning** (Cellpose, StarDist) would be superior for this task but requires model weights and dependencies.
