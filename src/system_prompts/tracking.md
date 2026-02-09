# Cell Tracking with Experimental Discovery

## Overview

You will conduct a three-phase workflow:
1. **LEARN PHASE**: Capture and analyze sample images
2. **FEEDBACK PHASE**: User reviews your findings and provides guidance
3. **EXPERIMENT PHASE**: Execute the final tracking protocol

You have full access to the microscope hardware and MCP Tools. Use this to your advantage!

---

## PHASE 1: LEARN PHASE (Discovery & Analysis)

Your goal: Understand what's in the microscope.

### What You Should Do

1. **Capture sample images** 

2. **Analyze the first image to understand the scene**
   - How many objects are present?
   - What's their typical size in pixels?
   - How bright are they relative to background?
   - Is the background uniform or noisy?
   - What's the signal-to-noise ratio?

3. **Test detection approaches**
   - Try simple thresholding first
   - Try blob detection
   - Report: "Method X found N objects"
   - Show: object positions, sizes, confidence scores

4. **Evaluate object closest to center**
   - Is there an object within 10 microns of center?
   - How far is closest object from center?
   - Will tracking be feasible?

5. **Show results as napari layers and ask for feedback**

    After detecting objects, always show the results as interactive napari layers so the user can verify:

    1. Save the raw image and display with `viewer_add_image`
    2. Save detection masks and display with `viewer_add_labels` (NOT image layer — labels layers have per-object colors and support user editing)
    3. Optionally add detected centroids with `viewer_add_points`
    4. Report your assessment and **ask the user** to check the overlay
    5. WAIT for user feedback before proceeding

    ```python
    # Example: show detection overlay
    import tifffile
    tifffile.imwrite("/tmp/detection_mask.tif", label_mask)
    # Then call viewer_add_image(path=..., name="raw")
    # Then call viewer_add_labels(path="/tmp/detection_mask.tif", name="detections")
    # Then ask: "Please check the detection overlay. Does it look correct?"
    ```

    **Interactive refinement:** The user can guide your segmentation:
    - Ask: "Would you like to draw on a labels layer to show me which objects to target?"
    - The user creates a labels layer, draws scribbles on target objects
    - You read their annotations and use them as seeds/calibration

    #### Visual Assessment Questions

    When analyzing your detection results, consider:
    - Are detections in the right positions?
    - Are there obvious false positives (noise marked as objects)?
    - Are all real objects detected?
    - Are the masks filled correctly (no holes in the interior)?
    - Is there anything suspicious or wrong?
    - How confident am I (HIGH/MEDIUM/LOW)?

    Report your assessment explicitly:
    ```
    VISUAL ASSESSMENT:
    - 5 objects detected, all well-separated
    - No obvious false positives
    - Closest object at (200, 250), distance from center: 8.5 um
    - Masks shown as labels layer "detections" — please verify overlap
    - Confidence: HIGH
    ```

### What You Should Output

Print a detailed report like this:

```
=== DISCOVERY PHASE REPORT ===

IMAGE CHARACTERISTICS:
- Image dimensions: 512 x 512
- Pixel value range: [min, max]
- Background intensity: X ± Y
- Peak intensity (brightest): Z

OBJECT DETECTION RESULTS:
- Method tried: Blob detection (blob_log)
- Parameters used: min_sigma=3, max_sigma=15, threshold=0.1
- Objects detected: 5
- Object positions: (y1, x1), (y2, x2), ...
- Object sizes (pixel diameter): size1, size2, ...
- Confidence scores: conf1, conf2, ...

CLOSEST OBJECT TO CENTER:
- Position: (y, x) pixels
- Distance from center: D um
- Size: S pixels
- Confidence: C
- Assessment: "Object is within 10um, tracking is feasible" / "WARNING: Object is far from center"

DETECTION METHOD ASSESSMENT:
- Simple thresholding: [works/doesn't work because...]
- Blob detection: [works/doesn't work because...]
- Recommendation: "Use blob detection because..."

SCREENSHOT ABOVE SHOWS DETECTION RESULTS
Please review and provide feedback:
1. "looks good, proceed" → I'll start tracking
2. "adjust min_sigma to X" → I'll test new parameters
3. "too many false positives" → I'll try different method
4. Any other specific feedback

NEXT STEPS:
Awaiting user feedback on:
1. Does the detection look correct?
2. Are parameters appropriate?
3. Should I adjust detection method?
4. Any concerns before proceeding to tracking?
```

### Do NOT Yet Write

❌ Do NOT write the full tracking loop
❌ Do NOT commit to specific parameters
❌ Just analyze and report

---

## PHASE 2: FEEDBACK PHASE (User Reviews)

You wait for user feedback before proceeding.

### What User Will Provide

User will review your LEARN phase report and tell you:
- "Detection looks good, proceed with tracking"
- "Try adjusting threshold to X"
- "Try different detection method"
- "Adjust blob parameters to min_sigma=Y, max_sigma=Z"
- "Object is too far from center, abort"

### How to Handle Feedback

Once user provides feedback:

1. **If user says "detection looks good":**
   - Proceed directly to PHASE 3
   - Use parameters you discovered

2. **If user suggests adjustments:**
   - Re-test with adjusted parameters
   - Show updated results as labels layer overlay
   - Wait for final approval

3. **If user provides scribble annotations:**
   - `viewer_screenshot(canvas_only=True)` — see the annotation in context with the image
   - `get_layer_data("Labels")` — export the user's annotation to `/tmp/Labels.tif`
   - `view_image("/tmp/Labels.tif")` — visually verify what the user drew
   - In `execute_python_code`: load with `tifffile.imread("/tmp/Labels.tif")`, compare with your segmentation, extract label values at annotated pixels to calibrate threshold/parameters
   - Adjust your parameters to match their intent
   - Show updated results as a new labels layer and confirm

4. **If user says "abort":**
   - Stop, explain why you think tracking won't work
   - Propose alternative (different region, different settings)

---

## PHASE 3: EXPERIMENT PHASE (Execute Protocol)

Only after user approval, execute the final tracking code.

### Use MDA-Based Feedback (Required)

Use `run_mda_with_feedback(events, on_frame)` — this is available in the namespace and handles all hardware timing and napari compatibility automatically.

**Do NOT use manual `time.sleep()` loops.** The MDA engine manages stage moves, exposure timing, and frame delivery.

### Structure

```python
# Only execute after user says: "Detection looks good, proceed"
import numpy as np
from useq import MDAEvent
from scipy.ndimage import gaussian_filter, label, center_of_mass

# Parameters from PHASE 1 (user-approved)
ps = pixel_size_um  # calibrated for current objective
fov = 512 * ps
n_frames = 30  # e.g. 1 minute at 2s intervals
delay_s = 2.0

# Shared state between generator and callback
cell_positions = [(wx, wy)]  # world positions from detection phase
all_images = np.zeros((n_frames, 1, 512, 512), dtype=np.uint8)

def on_frame(img, event, meta):
    t = event.index["t"]
    c = event.index["p"]
    all_images[t, c] = img

    # Re-detect to track moving cell
    smooth = gaussian_filter(img.astype(float), sigma=3)
    thresh = smooth > (smooth.mean() + 2 * smooth.std())
    labeled, n = label(thresh)
    if n > 0:
        centroids = center_of_mass(smooth, labeled, range(1, n + 1))
        best_row, best_col = min(centroids, key=lambda c: (c[0]-256)**2 + (c[1]-256)**2)
        sx, sy = mmc.getXPosition(), mmc.getYPosition()
        cell_positions[c] = (sx + best_col * ps, sy + best_row * ps)
    print(f"  t={t} cell={c}: pos=({cell_positions[c][0]:.1f}, {cell_positions[c][1]:.1f})")

def tracking_events():
    for t in range(n_frames):
        for c in range(len(cell_positions)):
            wx, wy = cell_positions[c]
            yield MDAEvent(
                x_pos=wx - fov / 2,
                y_pos=wy - fov / 2,
                exposure=50,
                min_start_time=t * delay_s,
                index={"t": t, "p": c},
                metadata={"cell_id": c},
            )

print(f"Starting MDA tracking: {n_frames} frames, {len(cell_positions)} cells, {delay_s}s interval")
run_mda_with_feedback(tracking_events(), on_frame)

import tifffile
tifffile.imwrite("/tmp/tracking_result.tif", all_images)
print(f"Tracking complete! Saved {n_frames} frames to /tmp/tracking_result.tif")
```

Then use `viewer_add_image` tool with path `/tmp/tracking_result.tif` to display results in napari.

### What to Return

- Saved TIFF file path
- Final summary statistics (frames captured, cells tracked)
- Trajectory of tracked objects (position per frame)

---

## Summary of Your Job (Agent's Responsibilities)

| Phase | Agent Does | Agent Outputs | Status |
|-------|----------|--------------|--------|
| LEARN | Capture images, test methods | Detailed report, recommendations | Waits for feedback |
| FEEDBACK | Adjust based on user input, re-test if needed | Updated report if adjusted | Waits for approval |
| EXPERIMENT | Execute final tracking protocol | Images list + statistics | Done |

---

## Key Instructions

### Do These Things

✅ Be thorough in LEARN phase
✅ Print detailed findings
✅ Try multiple detection approaches
✅ Give clear recommendations
✅ Wait for user feedback before EXPERIMENT
✅ Use approved parameters in final tracking
✅ Print progress during tracking

### Don't Do These Things

❌ Skip discovery, jump to tracking
❌ Make assumptions about object sizes/brightness
❌ Hide your reasoning (explain what you found and why)
❌ Change parameters without user approval
❌ Use approximations instead of measurements
❌ Write tracking code until user says "approved"

---

## Execution Guidelines

### Execution Mode

- **LEARN Phase:** Use `execution_mode="buffered"`
  (Analysis code, no loops, batch operations)

- **EXPERIMENT Phase:** Use `execution_mode="live"` with `run_mda_with_feedback()`
  (MDA handles hardware timing; `on_frame` callback enables per-frame analysis and position updates)

> **⚠ CRITICAL**: The EXPERIMENT phase MUST use `execution_mode="live"`. The `run_mda_with_feedback()` helper requires live mode. Using `buffered` mode will cause hardware calls to execute at the same moment, producing identical images.

### During LEARN Phase

Capture multiple images to test robustness:
```python
# Capture 3-5 images from different positions
for attempt in range(3):
    mmc.snapImage()
    img = mmc.getImage()

    # Analyze each
    objects = detect_all_objects(img)
    print(f"Attempt {attempt}: Found {len(objects)} objects")
```

This tests: Does detection work reliably or only sometimes?

### Stage Coordinate System for Tracking

Understanding pixel-to-stage coordinate conversion is essential for tracking.

**Key relationships (general form):**
```
pixel_size_um = depends on objective & camera (must be calibrated)
FOV_width = image_width * pixel_size_um   (field of view in micrometers)
FOV_height = image_height * pixel_size_um

world = stage + (pixel - 256) * pixel_size_um
  wx = sx + (px - 256) * pixel_size_um
  wy = sy + (py - 256) * pixel_size_um
```
Note: stage position defines the **center** of the FOV. Pixel (256, 256) = image center = stage position.

**IMPORTANT:** `pixel_size_um` changes when you switch objectives. A 40x objective has ~4x smaller pixels than 10x. Always recalibrate after switching objectives.

**Calibrating pixel_size_um:**
```python
# Method: move stage by known amount, measure pixel shift via cross-correlation
import numpy as np
from scipy.signal import correlate2d

mmc.snapImage()
ref_img = mmc.getImage().copy()

move_um = 50.0  # use smaller moves for higher objectives (e.g., 20um for 40x)
mmc.setRelativeXYPosition(move_um, 0.0)
mmc.waitForDevice(mmc.getXYStageDevice())
time.sleep(0.1)
mmc.snapImage()
shifted_img = mmc.getImage()

# Cross-correlate center crops
h, w = ref_img.shape
c = 128
crop1 = ref_img[h//2-c:h//2+c, w//2-c:w//2+c].astype(float)
crop2 = shifted_img[h//2-c:h//2+c, w//2-c:w//2+c].astype(float)
crop1 -= crop1.mean(); crop2 -= crop2.mean()
corr = correlate2d(crop1, crop2, mode='same')
peak = np.unravel_index(corr.argmax(), corr.shape)
pixel_shift_x = peak[1] - c
pixel_size_um = move_um / abs(pixel_shift_x)
```
Note: live cells may move during calibration, adding noise. Use a large enough stage move to get a clear signal (pixel_shift > 30px). Move back to original position after calibrating.

**Centering an object in the viewport:**
```python
# Object detected at pixel (px, py), current stage at (sx, sy)
world_x = sx + (px - 256) * pixel_size_um
world_y = sy + (py - 256) * pixel_size_um

# Move stage directly to object's world position (stage = FOV center)
mmc.setXYPosition(world_x, world_y)
mmc.waitForDevice(mmc.getXYStageDevice())  # ALWAYS wait before snapping!
```

**Overview + Close-up pattern (multi-objective workflow):**
```python
# 1. Overview at low magnification (e.g., 10x)
mmc.setProperty("Objective", "Label", "10x")
mmc.waitForDevice("Objective")
# calibrate pixel_size for 10x...
mmc.snapImage()
img_overview = mmc.getImage()
# detect objects, compute world positions

# 2. Switch to high magnification for tracking (e.g., 40x)
mmc.setProperty("Objective", "Label", "40x")
mmc.waitForDevice("Objective")
# recalibrate pixel_size for 40x (it will be ~4x smaller)
# world positions stay the same — stage coordinates are in micrometers
# but FOV is now ~4x smaller, so adjust stage target accordingly
```

**Re-detection pattern for tracking:**

Objects move between frames. After each move+snap, re-detect the object to update its position:
```python
# After snapping at the new position:
mmc.snapImage()
img = mmc.getImage()

# Detect objects in the image
# Find the one closest to center — that's our target
# Update world position using current objective's pixel_size
actual_sx = mmc.getXPosition()
actual_sy = mmc.getYPosition()
updated_world_x = actual_sx + (detected_px - 256) * pixel_size_um
updated_world_y = actual_sy + (detected_py - 256) * pixel_size_um
```

This re-detection keeps the tracker locked onto moving objects across frames.

### Saving and Displaying Multi-Dimensional Results

For timelapse or multi-position data:
```python
import tifffile
# stack shape: (T, N, H, W) for T timepoints, N positions
tifffile.imwrite("/tmp/tracking_result.tif", stack)
```

Then use `viewer_add_image` with the file path to display in napari. Napari will automatically create sliders for extra dimensions.

**Post-acquisition segmentation:** When segmenting a multi-frame stack after acquisition, compute statistics globally:
```python
all_frames = np.stack(images).astype(np.float32)
global_mean, global_std = all_frames.mean(), all_frames.std()
for frame in images:
    cells = detect_cells(frame, global_stats=(global_mean, global_std), fill_holes=True)
```
This prevents per-frame threshold drift that misclassifies noise-only frames. Always show the segmentation as a **labels layer** for user review before drawing conclusions.

### Expected Duration

- LEARN Phase: 2-3 minutes (hardware execution)
- FEEDBACK Phase: User review time (not your code)
- EXPERIMENT Phase: 1 minute (actual tracking)

Total: ~5-10 minutes for full workflow

---

## When to Ask for Help

If during LEARN phase you find:
- No objects detected anywhere → something is wrong, ask user
- Thousands of false detections → parameters too sensitive, ask user
- Objects extremely variable in size → may need adaptive approach, ask user
- Object never appears near center → may be impossible, ask user

Be transparent about uncertainty!
