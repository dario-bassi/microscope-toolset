# Adaptive Acquisition

> **Note**: Code examples below use conceptual pseudocode. All multi-frame
> loops MUST use `run_events(core, generator(), on_frame=callback)` in practice.
> See `knowledge/pymmcore/mda_patterns.md` for the correct MDA patterns.

## Pattern: Survey -> Decide -> Zoom -> Measure

```
10x survey -> detect ROIs -> rank -> switch to 40x -> re-detect -> measure
```

## Step 1: Low-Mag Survey (10x)

```python
set_objective(core, 10)  # Integer, NOT string "40x"
fov_um = core.getImageWidth() * core.getPixelSizeUm()
positions = grid_positions(center=(0, 0), nx=5, ny=5, fov_um=fov_um, overlap=0.1)

roi_candidates = []
for (x, y) in positions:
    core.setXYPosition(x, y)
    img = snap()
    score = compute_interest_score(img)  # Cell density, fluorescence, phenotype
    if score > threshold:
        roi_candidates.append((x, y, score))
```

## Step 2: Rank and Select

```python
roi_candidates.sort(key=lambda r: r[2], reverse=True)
top_rois = roi_candidates[:n_rois]
```

Do NOT commit to the first interesting spot. Survey the full area first.

## Step 3: Switch to High Mag (40x)

```python
set_objective(core, 40)  # FOV shrinks significantly at higher magnification
```

**Critical**: You MUST re-detect at 40x. A 10x centroid is ~5um accurate,
which is a 20-pixel error at 40x. The 10x position gives the neighborhood only.

## Step 4: Re-Detect and Measure

```python
for (wx, wy, _) in top_rois:
    core.setXYPosition(wx, wy)
    img_40x = snap()
    detections_40x = detect_cells(img_40x)  # Fresh detection at high res
    for det in detections_40x:
        results.append(measure_cell(det, img_40x))
```

## MDASequence + Adaptive Generator

```python
from useq import MDASequence, Position
survey = MDASequence(
    stage_positions=[Position(x=x, y=y) for (x, y) in positions],
    channels=["BF", "nucleus"],
)

def adaptive_generator(survey_results):
    for roi in rank_rois(survey_results):
        yield AcquisitionEvent(x=roi.x, y=roi.y, objective=40)
```

## Magnification Selection

| Obj | Use Case                 |
|-----|--------------------------|
| 10x | Survey, counting, tiling |
| 20x | Intermediate, tracking   |
| 40x | Morphology, subcellular  |

Always query `core.getPixelSizeUm()` for actual values — they vary by instrument.

## Common Pitfalls

- **Trusting low-mag positions at high-mag**: Always re-detect after zooming.
- **Incomplete survey**: Survey first, decide second.
- **Objective switching failures**: Verify with `core.getState('Objective')`.
- **FOV mismatch in counts**: 10x sees 16x the area of 40x.
