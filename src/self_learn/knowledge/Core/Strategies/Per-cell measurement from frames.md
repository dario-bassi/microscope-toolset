# Per-cell measurement from frames

The modern (post-2026-04-27) substitute for "read sim ground truth via
`bridge.get_cell_state`". Every per-cell quantity an experiment needs —
position, intensity in channel A, intensity in channel B, area, shape
class — has to be **derived from camera frames** by image segmentation
+ per-ROI feature extraction. The read-and-filter pattern still
applies; only the source of the measurements changes.

This is the workflow real wet labs use, and it's exactly what
[[Core/Approach/Transferability contract]] requires: camera frames in,
device commands out, no shortcut to internal state.

## The pipeline

For a single FOV:

```python
from self_learn.detection.cells import detect_cells
from self_learn.detection.fluorescence import measure_intensity

core.setConfig("Channel", reporter_channel)
core.snapImage()
img_reporter = core.getImage()

cells = detect_cells(img_reporter, threshold_sigma=2.5, min_area_px=50,
                     pixel_size_um=core.getPixelSizeUm())
# cells = [{centroid_px, area_px, bbox, ...}, ...]

per_cell = measure_intensity(img_reporter, cells)
# per_cell = [{cell_id, mean, max, integrated, ...}, ...]
```

For a multi-channel measurement (the typical replacement for the
old `intracellular_drug_conc_a` + `intracellular_drug_conc_b`
dual-field reads):

```python
channels = ("reporter_a", "reporter_b")
imgs = {}
for ch in channels:
    core.setConfig("Channel", ch); core.waitForConfig("Channel", ch)
    core.snapImage(); imgs[ch] = core.getImage()

# Segment once on the brighter / nuclear channel; reuse the masks.
cells = detect_cells(imgs["reporter_a"], pixel_size_um=core.getPixelSizeUm())
per_cell = [
    {
        "cell_id": c["cell_id"],
        "centroid_xy": c["centroid_px"],
        "intensity_a": _mean_in_bbox(imgs["reporter_a"], c["bbox"]),
        "intensity_b": _mean_in_bbox(imgs["reporter_b"], c["bbox"]),
    }
    for c in cells
]
```

Filter as a list comprehension (or `pandas.DataFrame.query` if you
prefer). The predicate shape is the same as the historical
[[#Read-and-filter (HISTORICAL)]] (AND-of-windows, asymmetric
band+ceiling, ratio-band) — see `feedback_dual_field_filter.md` for
the predicate vocabulary.

```python
band = [c for c in per_cell
        if 50 <= c["intensity_a"] <= 80
        and c["intensity_b"] < 30]
```

## Sources of measurement

| Quantity (sim era → real era) | Modern source |
|---|---|
| `bridge.get_cell_state` per-cell `centroid_xy` | `detect_cells(img)` returns `centroid_px`; convert to world units via `pixel_to_world` |
| per-cell `expression` (single reporter) | mean intensity in segmentation mask of that reporter channel |
| per-cell `intracellular_drug_conc` | indirect — needs a reporter responsive to the drug (FRET sensor, fluorescent indicator); read its intensity |
| per-cell `cumulative_illumination` | track in the agent: integrate per-snap exposure × power × cell-mask-coverage. Don't expect the camera to tell you. |
| `nuclear_fraction` (membrane vs nucleus) | two-channel segmentation: nucleus mask from DAPI, full-cell mask from membrane stain, ratio = nuc_intensity / full_intensity |

## Latency considerations

Reading sim ground truth was free of acquisition cost. Camera reads
are not — every snap advances time on a real-time engine
(`feedback_snap_budget`). When the protocol needs many per-cell
measurements at high cadence:

- Batch into a single multi-channel snap when possible (set channel,
  snap; set channel, snap; ...) rather than re-snapping the same
  channel for each cell.
- For closed-loop control, snap once per cycle and use the
  segmentation across the whole population — don't snap per cell.
- If the cell density is low, consider an MDA position-list pass
  (`MDASequence(stage_positions=...)`) so segmentation runs on
  cell-centred sub-FOVs rather than scanning the full chip.

## Related

- [[Core/Approach/Transferability contract]] — why this pipeline is
  the only legitimate one.
- [[Core/Strategies/Operating the solve loop]] § 5 — the now-
  HISTORICAL read-and-filter origin (`bridge.get_cell_state` →
  `Population.filter`).
- [[Core/Strategies/Measurement methodology]] — broader
  intensity / counting / sweep-vs-snapshot heuristics.
- `src/core/detection/cells.py` — `detect_cells`, `find_bright_centroid`.
- `src/core/detection/fluorescence.py` — `measure_intensity`,
  `crossmatch_channels`.
- `src/core/analysis/features.py` — `classify_cells`,
  `nearest_neighbor_distances`.
- `src/core/workflows/adaptive.py` — `pixel_to_world`,
  `survey_cells` for tying detections to stage coordinates.
