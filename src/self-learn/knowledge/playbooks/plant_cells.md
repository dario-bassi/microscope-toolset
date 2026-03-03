# Playbook: Plant Cell Analysis

## When to Use
- Plant tissue (onion epidermis, leaf, etc.)
- Tasks: segment cell walls, measure dimensions, detect plasmolysis, find nucleus position

## Step 1: Visual Inspection

```python
img = snap(core, "brightfield")
from PIL import Image; Image.fromarray(img).save("/tmp/plant.png")
```

Determine imaging mode:
- **Fluorescence**: Cell walls appear BRIGHT (calcofluor, propidium iodide)
- **Brightfield**: Cell walls appear DARK (transmitted light)

This distinction is CRITICAL for segmentation.

## Step 2: Cell Wall Segmentation

```python
from src.detection.plant_cells import segment_cell_walls
```

**For fluorescence (bright walls):**
```python
walls = segment_cell_walls(img, mode="fluorescence")
```

**For brightfield (dark walls -- MUST INVERT):**
```python
walls = segment_cell_walls(255 - img, mode="fluorescence")  # invert first
# OR
walls = segment_cell_walls(img, mode="brightfield")  # if supported
```

## Step 3: Individual Cell Extraction

After wall segmentation, the spaces between walls are individual cells:
```python
from scipy.ndimage import label
cell_regions, n_cells = label(~wall_mask)  # invert walls to get cells
```

## Step 4: Plasmolysis Detection

Plasmolyzed cells show:
- Cell membrane pulled away from cell wall
- Dark gaps between membrane and wall
- Visible as bright ring (wall) with smaller bright region (retracted protoplast) inside

Use LLM vision: "Is this plant cell plasmolyzed? Look for a gap between the cell wall and the contracted protoplast inside."

## Common Pitfalls

- **`segment_cell_walls()` assumes BRIGHT walls**: In brightfield, walls are DARK. Must invert the image first.
- **Wall thickness varies**: Use morphological operations to standardize.
