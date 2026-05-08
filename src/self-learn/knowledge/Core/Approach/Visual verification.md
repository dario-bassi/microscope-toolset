# Visual Verification

The agent is an LLM with vision. Use this superpower to sanity-check automated
results, classify ambiguous objects, and catch errors that pure numerical
analysis would miss.

---

## When to Use Vision

### Always Verify Visually
- First detection on a new sample (are the circles on real cells?)
- When automated count differs from expectation by >30%
- When classification is ambiguous (ring vs trophozoite, live vs dead)
- After changing detection parameters
- Before submitting final results

### Vision Is Not a Substitute For
- Quantitative measurement (pixel intensities, distances)
- Reproducible thresholds (human-like variability)
- High-throughput counting (too slow for 1000+ objects)
- Sub-pixel accuracy

---

## Saving Images for Inspection

Save annotated images to /tmp/ for the LLM to read.

### Raw Image

```python
from PIL import Image
import numpy as np

core.snapImage()
img = core.getImage().copy()

# Save as PNG (auto-scales 16-bit to 8-bit if needed)
if img.dtype == np.uint16:
    img_8bit = (img / img.max() * 255).astype(np.uint8)
else:
    img_8bit = img

Image.fromarray(img_8bit).save("/tmp/raw_image.png")
# Then read /tmp/raw_image.png with Claude's vision
```

### Detection Overlay

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(img, cmap="gray")

# Overlay detected objects as circles
for obj in detected_objects:
    circle = plt.Circle((obj.x, obj.y), obj.radius,
                        color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.text(obj.x, obj.y - obj.radius - 3, f"{obj.id}",
            color="yellow", fontsize=8, ha="center")

ax.set_title(f"Detected: {len(detected_objects)} objects")
plt.tight_layout()
plt.savefig("/tmp/detection_overlay.png", dpi=150)
plt.close()
```

### Side-by-Side Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(bf_img, cmap="gray")
axes[0].set_title("Brightfield")

axes[1].imshow(nuc_img, cmap="hot")
axes[1].set_title("Nucleus channel")

axes[2].imshow(mask, cmap="gray")
axes[2].set_title(f"Segmentation mask ({n_objects} objects)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("/tmp/comparison.png", dpi=150)
plt.close()
```

---

## What to Look For

### Cell Counting Verification
- Are circles centered on actual cells?
- Are there uncircled cells (false negatives)?
- Are there circles on debris/noise (false positives)?
- Are touching cells split correctly or merged?

### Segmentation Quality
- Do mask boundaries follow cell edges?
- Are there holes inside cells?
- Are background patches included in the mask?
- Is watershed splitting touching objects correctly?

### Classification Verification
- Malaria: ring parasites are small thin circles inside RBCs;
  trophozoites are large dense blobs filling >1/3 of the RBC
- Live/dead: live cells are bright in calcein; dead cells are dim
- Sample type: neurons have branching processes; blood cells are round;
  fibroblasts are elongated

### Spatial Sanity
- Are detected positions within the expected region?
- Do world coordinates map to the right image locations?
- Is the scale correct (cells should be ~10-30 um, not 1 um or 300 um)?

---

## Classification with Vision

For ambiguous cases, save a cropped region and ask the LLM directly.

### Crop and Save Individual Objects

```python
for i, obj in enumerate(ambiguous_objects):
    y0 = max(0, int(obj.y - 30))
    y1 = min(img.shape[0], int(obj.y + 30))
    x0 = max(0, int(obj.x - 30))
    x1 = min(img.shape[1], int(obj.x + 30))

    crop = img[y0:y1, x0:x1]
    Image.fromarray(crop).save(f"/tmp/object_{i}.png")
```

Then read each crop and classify: "Is this a ring-stage parasite or a
trophozoite? The ring is a small thin circle; the trophozoite fills most
of the RBC."

### Confidence Calibration

When automated classification gives low confidence:
1. Save the ambiguous object as a crop
2. Visually inspect with LLM vision
3. If vision agrees with automation, proceed
4. If vision disagrees, investigate: wrong threshold? wrong feature?

---

## Resolving Conflicts

When automated results conflict with visual inspection:

### Automated Says More Than Visual

```
Automated: 50 cells detected
Visual: ~20 cells visible
Likely cause: over-segmentation, noise as cells, wrong scale
Action: increase min_size filter, raise threshold, check morphology
```

### Automated Says Fewer Than Visual

```
Automated: 5 cells detected
Visual: ~20 cells visible
Likely cause: threshold too high, wrong channel, touching cells merged
Action: lower threshold, try different channel, add watershed splitting
```

### Automated Classification Disagrees

```
Automated: classified as trophozoite (dark_fraction=0.12)
Visual: looks like a ring (small, thin)
Likely cause: threshold for troph too low
Action: raise dark_fraction threshold to 0.30 for trophozoite
```

---

## Practical Tips

1. **Always use `matplotlib.use("Agg")`** before importing pyplot.
   The agent has no display; interactive backends will hang.

2. **Set DPI to 150** for overlays. Lower is blurry; higher is slow.

3. **Use contrasting colors**: red circles on grayscale, yellow text on dark
   background. Avoid blue on dark images.

4. **Label objects with IDs** in overlays so you can reference specific
   detections in your analysis.

5. **Save intermediate steps**: raw image, binary mask, labeled regions,
   final overlay. This makes debugging much easier.

6. **Keep /tmp/ clean**: use descriptive filenames like
   `/tmp/chN_well2_bf_overlay.png`, not `/tmp/img.png`.

7. **Multi-channel composites**: overlay fluorescence as color on BF grayscale
   for spatial context.

```python
# RGB composite: BF as gray, nucleus as red, GFP as green
composite = np.stack([
    (nuc_img / nuc_img.max() * 255).astype(np.uint8),
    (gfp_img / gfp_img.max() * 255).astype(np.uint8),
    (bf_img / bf_img.max() * 128).astype(np.uint8),
], axis=-1)
Image.fromarray(composite).save("/tmp/composite.png")
```

---

## Anti-Patterns

- **Skipping visual verification**: "The number looks right" is not enough.
  Look at the image. Every time.
- **Trusting automation blindly**: Automated detection fails in novel samples.
  Verify on first frame.
- **Not saving overlays**: Without overlays, you cannot debug detection errors.
- **Low-resolution saves**: DPI < 100 makes individual cells unresolvable.
- **Verifying only the final result**: Check intermediate steps too (mask,
  labels, features).

## See also

- [[Core/Approach/Confidence assessment]] — the decision framework that visual verification feeds.
- [[Core/Approach/Image quality]] — LLM-vision prompts for focus / exposure / artifact diagnosis.
- [[Core/Approach/Cell classification]] — LLM-vision prompts for cell-state calls.
- [[Core/Approach/Pre-submission checklist]] — includes "save the showcase image" as a mandatory step.
