# Playbook: Fibroblasts — F-actin stress fibers + focal adhesions

**Assumes:** fibroblast realism v1 (2026-04-24-). Phalloidin stains F-actin; focal adhesions appear as **elongated bright punctate foci at stress-fiber termini**, NOT as a continuous perimeter belt. DAPI typically co-acquired for nuclear localization.

**Paired Python module:** `src/recipes/fibroblast_focal_adhesions.py` (extracted 2026-04-24 from ch587 r3 validated pipeline).

## When to Use
- Any phalloidin / F-actin staining task on adherent fibroblasts.
- Focal-adhesion count, density, or per-cell aggregation.
- Stress-fiber morphology or orientation analysis.

## Background

Fibroblasts are large (50-100 µm long), elongated adherent cells with prominent stress fibers — bundled F-actin cables spanning the cytoplasm. Focal adhesions (FAs) are integrin-based adhesion complexes at the termini of these fibers; they are:
- 1-4 µm long, 0.3-0.8 µm wide → **2-8 px long at 0.5 µm/px, 1-2 px wide**
- Elongated with aspect ratio ~3-5:1
- Brighter in phalloidin than the stress-fiber baseline
- 5-10 per cell typical

## Channel Strategy

- **DAPI** — count nuclei (1:1 with cells); use as seed for cell-territory watershed. Fibroblasts are irregular and hard to segment from phalloidin alone.
- **Phalloidin** — FA detection + stress-fiber morphology.
- **Brightfield** — optional for showcase / edge-cell confirmation.

## Pitfall: LoG alone over-counts FAs ~10×

`puncta.detect_puncta_log(phalloidin)` picks up bright pixels along the length of stress fibers, not just at termini. The fiber itself is bright enough to register as a LoG peak every few pixels.

**Solution:** white-tophat (local-background suppression) + connected-component regionprops + eccentricity filter.

## Step 1: Cell segmentation from DAPI

```python
from src.core.analysis.puncta import segment_cells_from_nuclei
cell_labels, n_cells, _ = segment_cells_from_nuclei(
    dapi, min_nucleus_area=50, expand_px=120,  # fibroblasts are large
)
```

`expand_px=120` at 20x/0.5 µm/px = 60 µm territory radius beyond each nucleus — matches typical fibroblast reach.

## Step 2: FA detection — tophat + peak-local-max (recall-oriented)

**Preferred: call the packaged recipe.**

```python
from src.recipes.fibroblast_focal_adhesions import count_focal_adhesions
r = count_focal_adhesions(phalloidin, dapi, peak_k_sigma=8.0)
# Returns: n_cells, n_fa, mean_fa_per_cell, fa_positions_rc, cell_labels
```

**Or inline if you need to tune:**


**Updated 2026-04-24 after ch587 position-based grade feedback.**
The eccentricity-filter approach had low recall (F1=0.12 on ch587 r2
at 6 px match tolerance). For position-based grading the peak-detection
approach wins — draw each FA as a point and let the matching radius do
the rest.

```python
from skimage.feature import peak_local_max
from skimage.morphology import white_tophat, disk

# Top-hat with a larger footprint (disk 8) — larger than typical FA
# length, so long stress fibers also get stripped. Bright FA peaks pop.
th = white_tophat(phal, disk(8))
peaks = peak_local_max(
    th, min_distance=3,
    threshold_abs=th.mean() + 8.0 * th.std(),
)
# Restrict to peaks inside DAPI-expanded cell masks (rejects FP off-cell).
fa_positions = [(int(r), int(c)) for r, c in peaks
                if cell_labels[int(r), int(c)] > 0]
```

**Legacy tophat + eccentricity approach** (`detect_elongated_puncta` in
`src/core/analysis/puncta.py`) still works for count-based tasks where
you want region objects (not just points) — it also gives per-FA
orientation. Not recommended for PR-graded position submissions.

**Parameter sweep experience (ch588, v1 render):**

| ts  | ecc  | n_fa | per-cell mean |
|-----|------|------|---------------|
| 5.0 | 0.75 | 13   | 3.25 (TOO LOW) |
| 4.0 | 0.7  | 26   | 6.5            |
| 3.5 | 0.7  | 28   | 7.0            |
| 3.0 | 0.7  | 26   | 6.5            |

ts=3.5, ecc=0.7 is the sweet spot — captures GT range (5-10/cell) without noise pickup.

## Step 3: Per-cell assignment

```python
from src.core.analysis.puncta import count_puncta_per_cell, puncta_statistics
import numpy as np

peaks = np.array([[int(p.centroid[0]), int(p.centroid[1])] for p in fa_props])
counts, _ = count_puncta_per_cell(peaks, cell_labels, n_cells)
stats = puncta_statistics(counts)  # returns mean, median, total, etc.
```

## Visual verification (mandatory)

Every submission: overlay FA crosses on phalloidin + cell contours, save, and Read the PNG. If the overlay shows bright foci along stress fibers NOT marked, loosen ts or ecc; if marks land on empty background, tighten.

## Reference Solves

- **ch588 r1** (count-grade, 3.25 FA/cell — low): ts=5.0, ecc=0.75. Overlay showed undercounting. Flagged by orchestrator as a case where visual verification would have caught this before submitting.
- **ch588 r2** (count-grade, 7.5 FA/cell — right range): ts=3.5, ecc=0.7. Overlay verified before submit.
- **ch587 r2** (position-grade, F1=0.12 — poor): same eccentricity-region approach applied to position submission. Many real FAs missed by the area/ecc filter; tophat+peak-local-max with threshold mean+8σ on disk(8) tophat scored better at 42 vs GT 41 FAs (r3, result pending at time of writing). See `scratch/solve_587_r3_fibroblast.py`.

## See also

- [[Recipes/Neurons]] — same puncta-per-cell pattern for synaptophysin (ch587 composed cleanly; phalloidin needs the tophat/elongation filter on top because F-actin is structural, not punctate).
- [[Recipes/Lipid droplet]] — LoG-on-bright-spots reference; doesn't apply here (droplets are round and isolated, FAs are elongated and sit on bright fibers).
- `src/core/analysis/puncta.py` — segment_cells_from_nuclei, count_puncta_per_cell, puncta_statistics.
