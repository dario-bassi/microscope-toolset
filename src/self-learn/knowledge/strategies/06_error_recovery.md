# Error Recovery

> **Note**: Code examples below use conceptual pseudocode for diagnostics.
> In production code, use `snap(core, channel=ch)` for single frames and
> `run_events()` for all multi-frame loops.

Systematic troubleshooting for common microscopy agent failures. When something
goes wrong, do not guess randomly. Follow the diagnostic trees below.

---

## Symptom: 0 Cells Found

The most common failure. Work through causes in order of likelihood.

### Diagnostic Tree

```
0 cells detected
  |
  +-- Is the image black? --> See "Black Image" below
  |
  +-- Is the image all background?
  |     +-- Wrong XY position? Check core.getXPosition(), getYPosition()
  |     +-- Wrong well? Verify well index and stage coordinates
  |     +-- Sample not present? Snap BF and visually inspect
  |
  +-- Image has cells but detection misses them?
        +-- Wrong channel? Try all available channels
        +-- Threshold too high? Print histogram, check percentiles
        +-- Wrong scale? Check pixel_size, object size in pixels
        +-- Objects too small? Lower min_size / min_area filter
        +-- Objects too dim? Lower intensity threshold
        +-- Wrong method? Blob detection on confluent tissue fails
```

### Fix Sequence

```python
# Step 1: Check the image is not empty
core.setConfig(group, "BF")
core.snapImage()
img = core.getImage().copy()
print(f"Image stats: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
Image.fromarray(img).save("/tmp/debug_raw.png")

# Step 2: Try all channels
for ch in channels:
    core.setConfig(group, ch)
    core.snapImage()
    ch_img = core.getImage().copy()
    print(f"{ch}: min={ch_img.min()}, max={ch_img.max()}, std={ch_img.std():.1f}")

# Step 3: Check threshold on best channel
plt.hist(img.ravel(), bins=100)
plt.savefig("/tmp/histogram.png")

# Step 4: Try very permissive detection
from skimage.measure import label, regionprops
mask = img > np.percentile(img, 50)  # very low threshold
labels = label(mask)
print(f"Permissive detection: {labels.max()} objects")
```

---

## Symptom: Too Many Cells

Detection returns 10x or more than expected.

### Diagnostic Tree

```
Too many detections
  |
  +-- Over-segmentation?
  |     +-- Watershed splitting single cells? Lower seed sensitivity
  |     +-- Skeleton has spurious junctions? Erode before skeletonize
  |
  +-- Noise detected as cells?
  |     +-- Raise intensity threshold
  |     +-- Add minimum area filter
  |     +-- Median-filter the image before detection
  |
  +-- Wrong method for morphology?
  |     +-- LoG blobs on confluent tissue? Use threshold + CC instead
  |     +-- Peak detection on smooth regions? Raise min_distance
  |
  +-- Wrong scale assumption?
        +-- Check pixel_size: query core.getPixelSizeUm()
        +-- Feature size in pixels vs expected size
```

### Fix Sequence

```python
# Check: are detections real?
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray")
for prop in regionprops(labels):
    y, x = prop.centroid
    ax.plot(x, y, "r.", markersize=2)
ax.set_title(f"{len(regionprops(labels))} detections")
plt.savefig("/tmp/debug_detections.png", dpi=150)

# Often the fix is: increase min_area
props = regionprops(labels)
areas = [p.area for p in props]
print(f"Area stats: min={min(areas)}, median={np.median(areas):.0f}, max={max(areas)}")
# Filter small noise
real_cells = [p for p in props if p.area > expected_min_area]
```

---

## Symptom: Black Image

Image is entirely black or near-zero.

### Diagnostic Tree

```
Black image
  |
  +-- Channel not set correctly?
  |     +-- Check core.getAvailableConfigs(group) for valid names
  |     +-- Config group may be "Fake" not "Channel"
  |
  +-- Exposure too low?
  |     +-- Check core.getExposure()
  |     +-- Try core.setExposure(100)
  |
  +-- Wrong position? (sample not under objective)
  |     +-- Move stage to known position
  |
  +-- Z-focus completely off?
  |     +-- Run Z-scan to find focus
  |
  +-- SLM blocking light?
  |     +-- Check SLM mode and mask
  |
  +-- Camera issue?
        +-- Check core.getImageWidth() returns expected value
        +-- Verify camera device is initialized
```

### Fix Sequence

```python
# Step 1: Check channel config
groups = core.getAvailableConfigGroups()
print(f"Config groups: {groups}")
for g in groups:
    configs = core.getAvailableConfigs(g)
    print(f"  {g}: {configs}")

# Step 2: Set channel explicitly
core.setConfig(group, channels[0])
core.setExposure(100)
core.snapImage()
img = core.getImage().copy()
print(f"After explicit config: max={img.max()}")

# Step 3: Z-scan
z_device = core.getFocusDevice()
z_current = core.getPosition(z_device)
for dz in [-20, -10, 0, 10, 20]:
    core.setPosition(z_device, z_current + dz)
    core.snapImage()
    sharpness = np.var(np.array(core.getImage(), dtype=float))
    print(f"Z={z_current + dz:.1f}: sharpness={sharpness:.0f}")
```

---

## Symptom: Wrong Coordinates

Detected positions do not match expected locations.

### Diagnostic Tree

```
Wrong coordinates
  |
  +-- Pixel vs world confusion?
  |     +-- pixel_to_world() converts (row, col) to (x_um, y_um)
  |     +-- World coords are in micrometers, pixel coords in pixels
  |
  +-- Objective mismatch?
  |     +-- Coords at 10x != coords at 40x for same physical point
  |     +-- Check current pixel_size: core.getPixelSizeUm()
  |
  +-- X/Y axis swap?
  |     +-- Images are (row, col) = (y, x)
  |     +-- Stage is (x, y)
  |     +-- Common bug: passing (row, col) as (x, y)
  |
  +-- Origin confusion?
        +-- Image origin is top-left
        +-- Stage origin depends on calibration
        +-- Y-axis may be inverted between image and stage
```

### Fix: Coordinate Conversion

```python
# Image pixel (row, col) to world coordinates (x_um, y_um)
pixel_size = core.getPixelSizeUm()
stage_x = core.getXPosition()
stage_y = core.getYPosition()
img_w = core.getImageWidth()
img_h = core.getImageHeight()

def pixel_to_world(row, col):
    """Convert image pixel to world coordinates."""
    x_um = stage_x + (col - img_w / 2) * pixel_size
    y_um = stage_y + (row - img_h / 2) * pixel_size
    return x_um, y_um

# Always verify with a known landmark
```

---

## Symptom: Detection Works on First Frame but Fails Later

### Common Causes

1. **Photobleaching**: fluorescence signal decreases over time
   - Fix: use adaptive threshold relative to current frame, not first frame

2. **Z-drift**: sample drifts out of focus
   - Fix: monitor sharpness metric, refocus when it drops

3. **Cell movement**: cells leave FOV or change morphology
   - Fix: track objects across frames, handle appearance/disappearance

4. **Stage drift**: XY position shifts
   - Fix: use fiducial markers or cross-correlation registration

### Robust Detection Pattern

```python
def robust_detect(img, method="adaptive"):
    """Detection that adapts to varying image quality."""
    # Adaptive threshold based on THIS frame's statistics
    bg = np.percentile(img, 25)
    signal = np.percentile(img, 95)

    if signal - bg < 10:
        return []  # No signal in this frame

    threshold = bg + 0.3 * (signal - bg)  # Relative, not absolute
    mask = img > threshold
    labels = label(mask)
    props = regionprops(labels)

    # Filter by size (robust to noise)
    valid = [p for p in props if min_area < p.area < max_area]
    return valid
```

---

## Symptom: Measurement Outside Expected Range

### Sanity Check Table

| Measurement | Typical Range | Red Flag |
|------------|---------------|----------|
| Cell count (10x FOV) | 5-200 | 0 or >1000 |
| Cell diameter | 8-30 um | <3 or >100 um |
| Migration speed | 5-50 um/hr | 0 or >200 |
| Fluorescence ratio | 0.5-5.0 | <0.1 or >50 |
| Branch points | 1-10 per cell | 0 or >50 |
| Zone of inhibition | 10-30 mm | 0 or >50 mm |

### When Values Are Suspicious

```python
def sanity_check(measurement, name, expected_range):
    """Warn if measurement is outside expected range."""
    lo, hi = expected_range
    if measurement < lo or measurement > hi:
        print(f"WARNING: {name}={measurement} outside expected [{lo}, {hi}]")
        print("Investigate before submitting!")
        return False
    return True

sanity_check(cell_count, "cell_count", (5, 500))
sanity_check(speed, "speed_um_per_hr", (1, 100))
```

---

## General Recovery Patterns

### Retry with Backoff

```python
def retry_with_variants(img, methods):
    """Try multiple detection methods until one works."""
    for method_name, method_fn, expected_range in methods:
        result = method_fn(img)
        if expected_range[0] <= len(result) <= expected_range[1]:
            print(f"{method_name} succeeded: {len(result)} objects")
            return result
        print(f"{method_name} gave {len(result)}, trying next...")

    # All methods failed -- return best guess
    return result
```

### Refocus and Retry

```python
def refocus_and_retry(core, group, channel, detect_fn):
    """If detection fails, try refocusing first."""
    # Initial attempt
    core.setConfig(group, channel)
    core.snapImage()
    result = detect_fn(core.getImage().copy())

    if len(result) == 0:
        # Refocus
        best_z = autofocus(core, group, channel)
        core.setPosition(core.getFocusDevice(), best_z)
        core.snapImage()
        result = detect_fn(core.getImage().copy())

    return result
```

### Switch Method

```python
# Method-morphology matching: if first method fails, try alternatives
if count > 10 * expected:
    # Over-counting: switch from blob detection to connected components
    result = threshold_and_cc(img)
elif count == 0:
    # Under-counting: switch from threshold to adaptive/Otsu
    from skimage.filters import threshold_otsu
    t = threshold_otsu(img)
    result = threshold_and_cc(img, threshold=t)
```

---

## Pre-Submission Checklist

Before submitting any result:

- [ ] Value is within expected biological range
- [ ] Units are correct (um not px, unless px requested)
- [ ] Data types are Python-native (int not np.int64)
- [ ] Visual overlay confirms detection quality
- [ ] Edge cases handled (0 cells, 1 cell, max cells)
- [ ] JSON-serializable (no numpy arrays in output dict)

```python
# Convert numpy types to Python types for JSON
def sanitize(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
```
