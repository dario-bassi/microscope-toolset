# Decision Pattern: Detection Strategy Selection

## CRITICAL First Step: Match Method to Morphology

Before choosing any method, LOOK at the image and classify the cell morphology:

| Cell appearance | Correct method | WRONG method |
|-----------------|---------------|--------------|
| Large, space-filling (confluent tissue) | threshold + connected components | LoG blob detection (overcounts 10x+!) |
| Small bright dots on dark background | LoG blob_log or peak_local_max | Connected components (misses dim cells) |
| Touching/overlapping circles | Watershed (distance_transform + markers) | Simple CC (merges touching pairs) |
| Elongated structures (worms, gametocytes) | Ridge detection or PCA on CC | Circular blob detection |
| Sparse, well-separated | Simple threshold + CC | Complex watershed (unnecessary) |

**LoG on confluent tissue**: LoG blob detection on confluent tissue can overcount by 10x+ because it finds edges and texture in space-filling cells. Use threshold + CC instead.

**Gate the LoG response, not the raw intensity, for puncta on textured backgrounds.** If you're using `peak_local_max` on a LoG-filtered image, putting the threshold on the *raw* image's `mean + k·std` is fragile when the bright puncta themselves dominate the std (the gate ends up sitting inside the puncta band, letting cell-edge gradient hits through). Threshold the LoG response itself (`threshold_abs=…` on `-∇²G(σ)`): it's already band-passed, noise is whitened, and real puncta sit cleanly above gradient/noise responses. Optionally OR with a cell-mask gate from DAPI/membrane to drop residual extracellular false positives.

**Intensity-weighted centroids**: Pixel-level intensity weighting overestimates dispersion. Segment individual cells as connected components, track each centroid separately.

## Decision Tree

```
What are you detecting?
|
+-- Cells on plain background
|   |-- Sparse, good contrast -> detect_cells(sigma=2.0-2.5, fill_holes=True)
|   |-- Dense/touching -> count_objects_dt() on binary mask
|   |-- Very dim -> detect_cells(sigma=1.0-1.5)
|   +-- Need accurate count -> nucleus fluorescence if available
|
+-- Fluorescent spots
|   |-- Sparse -> detect_fluorescence(sigma=2.0)
|   |-- Dense/touching -> count_blobs_log(sigma=4)
|   +-- Cross-match with BF -> crossmatch_channels()
|
+-- Tissue (confluent)
|   +-- segment_tissue(use_otsu=True, tissue_mode=True)
|
+-- Neurons
|   |-- Soma detection -> detect_somata()
|   +-- Branch points -> count_branch_points() with CC assignment
|
+-- Blood cells
|   |-- RBCs -> threshold + count_objects_dt()
|   +-- Parasites -> per-RBC brightness comparison
|
+-- Plant cells
|   |-- Fluorescence (bright walls) -> segment_cell_walls()
|   +-- Brightfield (dark walls) -> invert image first!
|
+-- Point source
|   +-- detect_point_source(smooth_sigma=3)
|
+-- H&E tissue
|   +-- detect_nuclei_hae() (LAB color space + LoG)
```

## Context-Aware Parameters (use parameter_advisor)

**Always** use `suggest_*` functions to set pixel-domain parameters from physical context.
Hardcoded pixel values that work at 10x will fail at 40x.

```python
from self_learn.analysis.parameter_advisor import (
    suggest_watershed_params, suggest_detection_params, suggest_log_params,
    suggest_tracking_params, suggest_nc_params,
)
from self_learn.hardware.core import get_pixel_size

ps = get_pixel_size(core)  # µm/px

# Watershed splitting (nuclei, cells)
wp = suggest_watershed_params(ps, object_type='nucleus')
result = watershed_split(binary, min_distance=wp['min_distance'])

# Cell detection (threshold + CC)
dp = suggest_detection_params(ps, object_type='yeast')
cells = detect_cells(image, min_area_px=dp['min_area_px'])

# LoG blob detection
lp = suggest_log_params(ps, expected_diameter_um=8.0)
n = count_blobs_log(image, sigma=lp['sigma'])

# Tracking
tp = suggest_tracking_params(ps, object_type='bacterium', dt_s=0.3)
tracks = build_tracks(positions, max_dist=tp['max_dist_px'])

# N:C ratio measurement
np_ = suggest_nc_params(ps)
nc = compute_nc_ratio(image, nuc_mask, cyto_mask, guard_band=np_['guard_band'])

# Multi-FOV sampling (for proportion estimation)
from self_learn.analysis.parameter_advisor import suggest_sampling_params
sp = suggest_sampling_params(expected_fraction=0.15, objects_per_fov=50)
print(f"Need {sp['min_fovs']} FOVs for ±10% CI")
```

## When Detection Fails (0 cells or suspicious count)

1. **Try different sigma**: sweep [1.0, 1.5, 2.0, 2.5, 3.0]
2. **Check image quality**: is it in focus? well exposed?
3. **Visual inspection**: save image, use LLM vision to count manually
4. **Try different method**: switch from threshold to LoG, or vice versa
5. **Change channel**: if BF fails, try fluorescence
6. **Adjust exposure**: if image is too dim/bright
7. **Refocus**: run autofocus sweep

## Confidence Indicators

- **High confidence**: SNR > 10, detected count matches visual estimate, consistent across parameters
- **Medium confidence**: SNR 3-10, count varies +/-20% across sigma values
- **Low confidence**: SNR < 3, count varies wildly, visual inspection disagrees with detection
