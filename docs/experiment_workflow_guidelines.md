# Experiment Workflow Guidelines

These are practical tips for running microscopy experiments through the MCP tools. They apply to tracking, timelapse, stimulation, and any multi-step acquisition workflow.

For coordinate system details, execution modes, MDA usage, and SLM setup, see `microscope_tool.md`.

---

## General Workflow: Discover → Confirm → Execute

Most experiments follow the same three phases. The key insight is to **close the loop with the user** before committing to a long acquisition.

### 1. Discovery

Snap an image and **look at it first** before writing any analysis code.

**Look before you code:** Use `snap_image` to capture, then `viewer_screenshot(canvas_only=True)` or `view_image` to visually inspect the raw image. This tells you things that inform your entire analysis strategy:
- Are there cells at all, or is the field empty?
- Roughly how many — a handful or hundreds?
- How large are they — filling half the FOV or tiny dots?
- What's the morphology — round, elongated, clustered, isolated?
- What's the contrast like — bright on dark, dark on bright, low SNR?
- Are there artifacts — dust, out-of-focus regions, uneven illumination?

This visual assessment guides which detection approach to use (simple threshold vs blob detection vs more sophisticated methods), what parameter ranges make sense, and whether the current position/objective is even suitable.

**Then write analysis code** informed by what you saw. After detection runs, visually verify the result — use `view_image` with a segmentation overlay to cross-check that code output matches reality.

```
# Typical discovery flow:
# 1. snap_image → viewer_screenshot(canvas_only=True) → look at the raw image
# 2. Based on what you see, write detection code in execute_python_code
# 3. Save results: tifffile.imwrite("/tmp/labels.tif", labels)
# 4. Show in napari: viewer_add_labels(path="/tmp/labels.tif", name="detections")
# 5. Verify: view_image(image_path="/tmp/snap.tif", overlay_path="/tmp/labels.tif") — do masks match cells?
```

**Sweep parameters on the same image** rather than re-snapping. One snap, multiple analysis passes — avoids sample motion between tests.

**Report what you found** with enough detail for the user to make decisions: cell count, sizes, SNR, which parameters worked best, and your recommendation.

### 2. User Feedback

Present findings and wait for the user's go-ahead. Common interactions:

- **User approves** → proceed with discovered parameters
- **User adjusts parameters** → re-run detection, show updated overlay, confirm again
- **User draws annotations** → read their labels layer with `get_layer_data`, use the annotation to calibrate your detection (e.g., extract intensity values at annotated pixels to set thresholds)
- **User says it's not working** → propose alternatives (different region, different objective, different approach)

The annotation workflow is particularly powerful:
```
get_layer_data("Labels") → /tmp/Labels.tif
→ load in execute_python_code
→ compare with your segmentation
→ adjust parameters to match user intent
→ show updated result as new labels layer
```

### 3. Execution

Run the actual acquisition with user-approved parameters. Use `run_mda_with_feedback()` with `execution_mode="live"` for any multi-frame workflow.

Print progress periodically (e.g., every 10-50 frames) so the user can monitor. Save intermediate results to disk as you go — if something fails at frame 500, you still have frames 0-499.

After acquisition, show results in napari and visually verify before reporting to the user.

---

## Tips for Robust Experiments

### Contrast and visibility

Microscopy images often have low contrast — a 16-bit camera might have background at ~200 and cells at ~300 out of 65535. `view_image` auto-scales contrast using percentiles (1st–99.9th) to handle this and ignore hot pixels, but it may not always show the structures you care about. If the image looks too dark or washed out, use `viewer_screenshot(canvas_only=True)` instead — the user can adjust contrast limits in napari's layer controls, and the screenshot captures exactly what they see.

### Vision API and code serve different roles

The vision API gives you qualitative understanding — spatial distribution, morphology, whether something looks like a cell or an artifact. Code gives you precise measurements — centroids, areas, intensities. Use them in sequence:

1. **Look first** (`viewer_screenshot` / `view_image`) → form expectations about what detection should find
2. **Run code** → get precise metrics
3. **Look again** (`view_image` with overlay) → verify code output matches what you see
4. If they disagree, trust what you see and adjust the code

This catches things metrics alone miss:
- Over-segmentation (one cell split into fragments — code says "3 objects" but you see 1 cell)
- False positives on dust or artifacts
- Threshold too high/low for the actual contrast
- Missed cells at image edges or in clusters

### Calibration

When `pixel_size_um` isn't known, calibrate by cross-correlation:

```python
# Move stage by a known amount, measure pixel shift
mmc.snapImage()
ref = mmc.getImage().copy()

move_um = 50.0  # smaller for high-mag objectives
mmc.setRelativeXYPosition(move_um, 0.0)
mmc.waitForDevice(mmc.getXYStageDevice())
mmc.snapImage()
shifted = mmc.getImage()

# Cross-correlate center crops
from scipy.signal import correlate2d
h, w = ref.shape
c = 128
crop1 = ref[h//2-c:h//2+c, w//2-c:w//2+c].astype(float)
crop2 = shifted[h//2-c:h//2+c, w//2-c:w//2+c].astype(float)
crop1 -= crop1.mean(); crop2 -= crop2.mean()
corr = correlate2d(crop1, crop2, mode='same')
peak = np.unravel_index(corr.argmax(), corr.shape)
pixel_shift = peak[1] - c
pixel_size_um = move_um / abs(pixel_shift)
```

Use a large enough stage move (pixel shift > 30px) for a clean signal. Move back afterward. Recalibrate whenever you switch objectives.

### Re-detection during tracking

Cells move between frames. After each snap, re-detect the target and update its world position. The `on_frame` callback in MDA-based workflows is the natural place for this — detect, update shared state, and the generator uses the updated position for the next event.

### Multi-frame segmentation

When segmenting a timelapse stack, compute statistics globally across all frames first, then apply a single threshold everywhere. Per-frame normalization fails on frames with no cells (noise gets segmented) or dominant cells (threshold too high). See `microscope_tool.md` for details.

### Saving results

Save all intermediate outputs (timelapse TIFF, label masks, tracks) so post-hoc analysis is always possible without re-acquiring. Show segmentation as **labels layers** (not image layers) — they support per-object colors and user editing in napari.
