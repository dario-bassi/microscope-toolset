# SLM/DMD ↔ Camera Calibration

## Overview

Computes an affine transform between SLM (Spatial Light Modulator) pixel coordinates and
camera pixel coordinates. This enables mapping any camera-space region (e.g., from
segmentation) to an SLM pattern for targeted illumination, optogenetics, FRAP, etc.

## When to Calibrate

The affine transform is **specific** to the current hardware configuration. Re-calibrate after:
- Changing the **objective** (magnification, light path)
- Changing the **channel** or config group (different optical paths)
- Changing **camera binning** or ROI settings
- Physically realigning the SLM or its relay optics

## Key Concept: SLM Conjugate Plane

On many microscopes, the SLM is **not** conjugate to the sample focal plane. SLM patterns
may only be visible on the camera at a **different Z position**. Before calibrating:

1. Upload a test pattern (grid of dots) to the SLM.
2. Scan through Z while imaging — look for the position where the pattern is sharpest.
3. Use Laplacian variance as a focus metric — peak = conjugate plane.

If the SLM pattern is never visible at any Z, check the optical path (relay optics, port
selection, light path switches).

## Reusable Code

```python
from src.self_learn.hardware.slm_calibration import (
    find_slm_conjugate_z,
    calibrate_slm,
    camera_mask_to_slm,
    save_calibration,
    load_calibration,
)
```

### Step 1: Find the conjugate Z plane

```python
z_conj = find_slm_conjugate_z(
    core, slm_device="Mosaic3",
    channel_group="TTL_ERK", channel_config="CyanStim",
    z_center=3500,    # start near sample focus
    z_range=1500,     # scan +/- this range
    z_step=50,        # coarse step (refine later if needed)
)
```

### Step 2: Calibrate (3 points + 3-point verification)

```python
# With MDA (preferred — handles shutter/SLM coordination automatically):
calib = calibrate_slm(
    core, slm_device="Mosaic3",
    channel_group="TTL_ERK", channel_config="CyanStim",
    z_conjugate=z_conj,
    run_mda_fn=run_mda_with_feedback,  # from MDA helpers
)

# Without MDA (fallback — uses manual shutter control):
calib = calibrate_slm(
    core, slm_device="Mosaic3",
    channel_group="TTL_ERK", channel_config="CyanStim",
    z_conjugate=z_conj,
)
```

### Step 3: Save / Load

```python
save_calibration(calib, "dmd_calibration.json")
calib = load_calibration("dmd_calibration.json")
```

### Step 4: Use in experiments

```python
# Convert any camera-space mask to SLM coordinates
fwd = calib["dmd_to_camera"]["matrix"]
slm_mask = camera_mask_to_slm(segmentation_mask, fwd, slm_w=800, slm_h=600)

# Upload to SLM
core.setSLMImage(slm_device, slm_mask)
core.displaySLMImage(slm_device)
```

## How It Works

1. **3 asymmetric points** are displayed sequentially on the SLM (triangle pattern).
   Each is paired with a dark frame for background subtraction.
2. Centroids are detected in each dark-subtracted camera image via thresholding +
   connected component labeling + center of mass.
3. The 3 point correspondences (slm_x, slm_y) → (cam_x, cam_y) yield an **exact**
   affine solution (6 unknowns, 6 equations).
4. **3 independent test points** at different positions verify the transform.
   If all predicted-vs-actual errors are < 5 px, the calibration passes.

## Why 3 Points and Not a Grid?

- An affine has 6 degrees of freedom → 3 points give an exact solution.
- Sequential display ensures unambiguous matching (no sorting/orientation ambiguity).
- The verification step with 3 *different* points catches any systematic error.
- A grid of dots can additionally be used for a least-squares fit if higher
  robustness is desired, but requires careful matching (see Common Mistakes).

## Important: SLM Timeout

Some SLMs (e.g., Andor Mosaic III) **turn off after a timeout** (typically ~2 minutes).
Always re-upload the mask before each use during experiments. If using MDAEvent with
SLMImage, this is handled automatically per frame.

## Common Mistakes

1. **Using a symmetric pattern without sequential verification** — a regular grid has
   rotational/reflection symmetry, so you cannot determine axis flips without displaying
   dots one at a time or using an asymmetric marker.
2. **Imaging at the wrong Z** — SLM patterns are invisible at the sample plane if the SLM
   is not conjugate to it. Always find the conjugate Z first.
3. **Not re-uploading the SLM mask** — the SLM may turn off or reset between acquisitions.
4. **Wrong channel/group** — only channels routed through the SLM will show the pattern.
   Other channels may bypass it entirely.
5. **Camera acquisition errors** — some cameras (e.g., PVCAM) fail on `snapImage()`. The
   calibration module falls back to sequence acquisition automatically.
6. **Confusing coordinate conventions** — camera centroids from `center_of_mass()` are
   (row, col) = (y, x). The affine uses (x, y). Pay attention to order.
