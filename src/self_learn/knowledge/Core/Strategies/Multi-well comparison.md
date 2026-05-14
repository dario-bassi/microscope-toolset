# Multi-Well Comparison

> **When to use:** When comparing phenotype or signal intensity across multiple wells relative to a control condition.

> **Note**: Code examples below use conceptual pseudocode. All multi-position
> loops MUST use `MDASequence(stage_positions=[...])` + `run_events()` in practice.

## Core Principle: Survey ALL Wells First

Never classify a well in isolation. A "low" signal is only meaningful relative to control.

```
survey all wells -> extract metrics -> compare to control -> classify
```

## Step 1: Survey with Identical Settings

```python
well_data = {}
for well_id, x, y in get_well_positions():
    core.setXYPosition(x, y)
    images = {}
    for ch in ["BF", "nucleus", "membrane"]:
        core.setConfig("Channel", ch)
        images[ch] = snap()
    well_data[well_id] = images
```

Do NOT change exposure between wells -- invalidates intensity comparisons.

## Step 2: Extract Metrics

```python
for well_id, images in well_data.items():
    nuc = images["nucleus"]
    cell_count = int(count_cells(nuc))
    mean_fl = float(np.mean(nuc[nuc > background_threshold]))  # Mask background!
    metrics[well_id] = {"cell_count": cell_count, "mean_fluorescence": mean_fl}
```

Convert `numpy.int64` to Python `int` before JSON serialization.

## Step 3: Compare to Control

```python
control = metrics["A1"]
for well_id, m in metrics.items():
    suppression = m["mean_fluorescence"] / control["mean_fluorescence"]
    growth = m["cell_count"] / control["cell_count"]
```

## Classification Decision Tree

```
1. Low-variance band in BF rows (std < 5)?        -> "wound_healing"
2. Cell count > 2.5x control?                      -> "overgrowth"
3. Fluorescence < 0.5x control?                    -> "drug_treated"
4. Similar to control?                             -> "control"
```

### Wound Detection
```python
row_std = np.std(bf, axis=1)
wound_present = np.sum(row_std < 5) > 20  # 20+ rows of uniform gap
```

### Drug Effect
Drug-treated looks IDENTICAL to control in BF. You MUST check fluorescence.
Suppression ratio ~ 0.3 means 70% drug suppression.

## Common Pitfalls

- **Classifying before surveying**: Maybe ALL wells are dim today.
- **BF-only classification**: Drug effects invisible in brightfield.
- **Background in fluorescence mean**: Mask out zeros; including background halves values.
- **Numpy types in JSON**: `json.dumps()` fails on `numpy.int64`. Cast first.
- **Single-field bias**: For borderline results, snap multiple positions per well.
