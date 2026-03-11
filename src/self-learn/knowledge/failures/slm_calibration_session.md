# SLM/DMD Calibration — Debugging Session

## Session: DMD-Camera Affine Calibration

Calibrating an SLM (Andor Mosaic III DMD) to camera coordinate mapping required solving
multiple problems before the calibration succeeded. Key learning: **always verify the
physical optical path first before debugging software**.

---

## Problem 1: Wrong SLM Device Name

### Symptom
```
RuntimeError: No device with label "SLM"
```

### Root Cause
Assumed the SLM device was named `"SLM"` — the actual device label was `"Mosaic3"`.
Device labels are set during Micro-Manager configuration and are not standardized.

### Solution
Query loaded devices at runtime:
```python
slm_device = core.getSLMDevice()  # Returns the configured SLM label
```
Never hardcode device names.

---

## Problem 2: Wrong Channel Group

### Symptom
MDA ran but produced no illumination difference between SLM-on and SLM-off frames.
Warning in logs: `Preset "CyanStim" of configuration group "Channel" does not exist`.

### Root Cause
Used `"Channel"` as the group name. The actual group was `"TTL_ERK"`.
The MDA engine silently skipped channel configuration, so the wrong light path was active.

### Solution
Query available groups first:
```python
groups = core.getAvailableConfigGroups()
for g in groups:
    presets = list(core.getAvailableConfigs(g))
    print(f"{g}: {presets}")
```
The SLM channel must be in the correct group — in this case `TTL_ERK` containing `CyanStim`.

### Lesson
**Silent failures are the worst.** The MDA engine warned but did not error when the
channel group was wrong. Always check that your channel is actually being set by
verifying signal levels change between SLM-on and SLM-off.

---

## Problem 3: No Reflective Surface in the Light Path

### Symptom
DMD patterns were completely invisible on the camera at the sample focal plane.
All images showed uniform illumination regardless of DMD pattern.
Dark-subtracted difference images had near-zero contrast (max ~5 counts).

### Root Cause
**There was no glass slide or reflective surface on the microscope stage.**
The DMD illumination was going through the objective into empty space with nothing
to reflect or scatter the patterned light back to the camera.

### Solution
User placed a glass coverslip on the stage. The coverslip surface reflects enough
light to image the DMD pattern — but only at the correct Z plane (see Problem 5).

### Lesson
**For SLM calibration without fluorescent samples, you need a reflective surface**
(glass slide, coverslip, mirror slide, or even a piece of paper). The SLM pattern
is an illumination pattern — without something to scatter/reflect that light back
through the objective, the camera cannot see it.

---

## Problem 4: Symmetric Pattern Cannot Determine Orientation

### Symptom
Initial calibration used a regular 6×8 grid of dots. The affine fit looked good
(low RMSE) but the X-axis was incorrectly assigned as flipped.

### Root Cause
A regular grid has rotational and reflection symmetry. When sorting detected
centroids by position, there is no way to distinguish between the correct
matching and a horizontally-flipped matching — both produce valid affine transforms
with similar residuals.

### Solution
Display dots **sequentially** (one at a time) at **asymmetric** positions.
An L-shaped pattern (3 dots: top-left, top-right, bottom-left) unambiguously
determines the axis orientation because:
- If DMD x+ maps to camera x+, the horizontal pair spreads in the same direction
- If flipped, they spread in opposite directions

### Lesson
**Never use a symmetric pattern alone for calibration.** Either:
1. Display dots one at a time (sequential = unambiguous matching), or
2. Use an asymmetric marker (e.g., L-shape, or different dot sizes)

---

## Problem 5: SLM Not Conjugate to Sample Focal Plane

### Symptom
Even with a reflective surface, the DMD grid pattern was invisible at Z≈3500 um
(normal sample focus). Images showed uniform illumination with no spatial structure.

### Root Cause
The DMD is optically conjugate to a different Z plane (~4600 um for the 20x
objective), about 1100 um above the sample focus. At the sample plane, the DMD
pattern is completely out of focus and appears as uniform illumination.

### Solution
Performed a Z-scan from 3300-4800 um with the grid pattern, measuring Laplacian
variance (image sharpness) at each step. The conjugate plane showed a dramatic
peak in both sharpness and contrast.

### Lesson
**Always find the SLM conjugate Z plane first.** The offset depends on the
objective and relay optics. Use `find_slm_conjugate_z()` before calibrating.

---

## Problem 6: DMD Mask Not Persisting Between Frames

### Symptom
In the first Z-scan, only frame 0 had signal (mean=9194), while all subsequent
frames had very low signal (mean=131). Appeared as if only the first acquisition
worked.

### Root Cause
The DMD mask was uploaded once before the scan loop but was cleared or timed out
between frames. The shutter opening/closing cycle may have reset the DMD state.

### Solution
Re-upload the DMD mask before **every** acquisition frame:
```python
for z in z_positions:
    core.setSLMImage(device, mask)
    core.displaySLMImage(device)
    # ... then acquire
```

### Lesson
**Never assume the SLM mask persists.** Some SLMs reset on shutter events or
after a timeout. Always re-upload before each acquisition. When using MDAEvent
with `slm_image=SLMImage(...)`, this is handled automatically per frame.

---

## Problem 7: Camera snapImage() Failure

### Symptom
```
RuntimeError: Camera image buffer read failed.
```

### Root Cause
PVCAM cameras can fail on `snapImage()` due to a buffer management issue.

### Solution
Fall back to single-frame sequence acquisition:
```python
try:
    core.snapImage()
    img = core.getImage()
except RuntimeError:
    core.clearCircularBuffer()
    core.startSequenceAcquisition(1, 0, True)
    while core.isSequenceRunning():
        time.sleep(0.05)
    img = core.popNextImage()
```
The `slm_calibration.py` module handles this fallback automatically.

---

## Problem 8: Saturation at Conjugate Plane

### Symptom
At the DMD conjugate Z plane, images were saturated (max=65535) even at 5ms exposure.
Could not detect dot centroids accurately.

### Root Cause
At the conjugate plane, the DMD pattern is sharply focused — all light is concentrated
into small spots instead of spread across the field. Much less exposure is needed.

### Solution
Reduced exposure to 2ms. For different microscopes, start low and increase until
dots are bright but not saturated.

---

## Problem 9: Ambient Room Light

### Symptom
Dark frames had unexpectedly high values (mean=570 instead of expected ~105).
This added 465 counts of noise to all difference images.

### Root Cause
Room lights were on. Even with the microscope enclosure, stray light leaked in.

### Solution
Turn off room lights during calibration.

---

## Summary: Calibration Debugging Checklist

1. ✅ Verify SLM device name: `core.getSLMDevice()`
2. ✅ Verify channel group has the SLM channel: check `getAvailableConfigGroups()`
3. ✅ Ensure a reflective surface is on the stage (glass, coverslip, mirror)
4. ✅ Turn off room lights
5. ✅ Find the SLM conjugate Z plane (Z-scan with grid pattern)
6. ✅ Use low exposure to avoid saturation at conjugate plane
7. ✅ Re-upload SLM mask before every frame (or use MDAEvent.slm_image)
8. ✅ Use sequential/asymmetric dots for orientation — never rely on symmetric grids alone
9. ✅ Handle snapImage() failures with sequence acquisition fallback
10. ✅ Verify calibration with independent test points before saving
