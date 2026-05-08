# Playbook: Malaria Blood Smear Analysis

## When to Use
- Giemsa-stained thin blood smear
- Tasks: count parasitized RBCs, classify parasite stages (ring, trophozoite, gametocyte), compute parasitemia

## Step 1: Visual Inspection

```python
img = snap(core, "brightfield")
from PIL import Image; Image.fromarray(img).save("/tmp/malaria.png")
```

Look for:
- Pink/pale RBCs filling the field
- Small dark dots inside some RBCs (ring-stage parasites)
- Banana-shaped forms (gametocytes)
- Blue-purple chromatin dots

## Step 2: RBC Detection

```python
from src.core.detection.cells import detect_cells
rbcs = detect_cells(img, threshold_sigma=1.5, min_area_px=50, fill_holes=True)
```

Or for dense fields, use distance-transform counting:
```python
from src.core.detection.cells import count_objects_dt
n_rbcs = count_objects_dt(rbc_mask, filter_size=5, dt_threshold=3)
```

## Step 3: Infected Cell Detection

**Don't use dark_fraction threshold** -- too insensitive for ring parasites.

Better approach -- per-cell brightness comparison:
- Infected RBCs are darker than normal RBCs due to parasite
- Compare each RBC's mean intensity to the population median
- Threshold: RBC mean intensity significantly below population median → likely infected
- Exact threshold depends on staining quality — calibrate per slide

Plus chromatin dot detection:
- Ring parasites have small, very dark dots (chromatin)
- Look for local minima within each RBC mask that are significantly darker than the RBC mean

## Step 4: Stage Classification (CRITICAL)

**RING forms are by far the most common** (~70% in P. falciparum infections). Previous errors: classifying rings as trophozoites because thresholds were too low.

Use watershed segmentation to isolate individual RBCs for per-cell measurements, then classify:

### Ring-stage
- **Most common** -- default to RING when in doubt (~60%+ in peripheral blood)
- Thin dark rim within RBC with 1-2 small chromatin dots
- Dark fraction relatively low (rings have thin parasites, not large dense bodies)
- **Biological prior**: In P. falciparum, trophozoites/schizonts sequester in deep vasculature — rings dominate peripheral smears. If ring ≈ troph count, your threshold is too strict.
- Bias toward RING when uncertain — raise troph threshold until ring >> troph

### Trophozoite
- Large AMORPHOUS dense body filling >1/3 of RBC
- High dark fraction (significantly more than rings)
- Roughly circular, not elongated
- Active feeding stage — less common in peripheral P. falciparum smears

### Gametocyte
- Banana/crescent-shaped (distinctive morphology)
- High eccentricity (highly elongated)
- Large parasite area
- Use PCA elongation: eigenvalue ratio > 3
- Sexual stage, no other blood cell has this shape

### Schizont
- Fills most of the RBC
- Multiple dark bodies (merozoites) visible
- Late stage, about to burst

**Classification decision tree:**
1. Check eccentricity first: highly elongated + parasitized → gametocyte
2. If n_dark_bodies >= 4 → schizont
3. If large dense amorphous body filling >1/3 RBC → trophozoite
4. **Everything else → RING** (most common stage, 60%+ in peripheral blood)
5. Sanity check: ring count should be >> trophozoite count in P. falciparum

## Step 5: Parasitemia Calculation

```
parasitemia = n_infected / n_total_rbcs * 100
```

## Common Pitfalls

- **Ring parasites are subtle**: Small dark fraction within an already-pale RBC. Global thresholding misses them.
- **Brightness comparison is key**: Compare each RBC's mean intensity to population median.
- **Don't confuse platelets with parasites**: Platelets are outside RBCs and very small.
- **Staining variation**: Giemsa intensity varies across the smear. Use per-cell relative measurements.

## LLM Vision Alternative

For difficult cases, save individual RBC crops and use LLM vision to classify:
"Is this RBC infected? Look for small dark dots (chromatin), signet-ring shapes, or any internal structure that differs from a normal uniform pink RBC."
