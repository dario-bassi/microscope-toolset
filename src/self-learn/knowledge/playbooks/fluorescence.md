# Playbook: Fluorescence Analysis

## When to Use
- Fluorescence imaging (nucleus, membrane, GFP, GCaMP, etc.)
- Tasks: count labeled cells, measure intensity, classify bright/dim, cross-match with BF

## Step 1: Visual Inspection

```python
core.setConfig("Fake", "nucleus-channel")  # or appropriate channel
core.snapImage()
img = core.getImage()
from PIL import Image; Image.fromarray(img).save("/tmp/fluor.png")
```

Look for:
- Bright spots on dark background (typical fluorescence)
- Spot density: sparse or overlapping?
- Background: is it uniform or has autofluorescence gradient?

## Step 2: Detection

```python
from src.detection.cells import detect_cells, count_blobs_log
# For well-separated spots:
cells = detect_cells(img, threshold_sigma=2.0, min_area_px=15, fill_holes=True)

# For dense/touching spots:
n = count_blobs_log(img, sigma=4, peak_thresh=1.0)
```

## Step 3: Intensity Measurement

**CRITICAL: Define ROI consistently with what the analysis expects.**
- Nucleus-only ROI: bright core pixels only
- Cell body ROI: includes surrounding dimmer cytoplasm
- Using different ROI sizes gives very different mean intensities

```python
from skimage.measure import regionprops
# Measure per-cell intensity from labeled mask
props = regionprops(label_mask, intensity_image=img)
intensities = [p.intensity_mean for p in props]
```

## Step 4: Classification

For multi-class intensity classification:
```python
from src.analysis.intensity import classify_intensities
result = classify_intensities(intensities, method='gap')
```

## Step 5: Cross-Channel Matching

Match cells detected in different channels by nearest-neighbor position:
```python
from src.analysis.tracking import match_frames
matches = match_frames(bf_positions, nuc_positions, max_dist=50.0)
```

## Common Pitfalls

- **ROI size matters**: Nucleus-only vs cell-body ROI gives drastically different intensities. Be consistent.
- **Per-frame thresholding in Z-stacks**: Compute global threshold across all frames, don't re-threshold each frame.
- **Nucleus channel for spatial analysis**: Overlapping spots shift centroids. Use BF for better centroid accuracy.
- **Fluorescence for counting, BF for positions**: Best of both worlds when both channels available.
- **Background masking**: Mask out background pixels (intensity < noise floor) before computing mean ROI intensity.
