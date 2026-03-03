# Z-Stack / 3D Imaging Playbook

## When to use
- Organoids, spheroids, thick tissue sections
- Any sample where structures span multiple focal planes
- Volumetric measurements (cell counting in 3D, organoid size)
- Finding the best focal plane for 2D analysis

## Acquisition

### MDA Z-stack
```python
from useq import MDASequence
from src.hardware.core import run_events

seq = MDASequence(
    z_plan={"range": 50, "step": 2},  # ±25 µm, 2 µm steps
    channels=[{"config": "GFP"}, {"config": "BF"}],
)
results = run_events(core, list(seq))
# results is list of (image, event) tuples
```

### Adaptive Z-range
1. Start with `autofocus_mda()` to find current focus
2. Snap BF at several Z positions to find sample extent
3. Set Z-range to cover the sample, not the full stage range

## Analysis with src/analysis/zstack.py

### Projections
```python
from src.analysis.zstack import project_mip, project_mean, project_std

# Maximum intensity projection — shows all bright structures
mip = project_mip(stack)

# Mean projection — reduces noise, good for uniform structures
mean = project_mean(stack)

# Std projection — highlights structures that change with Z (focus map)
std_map = project_std(stack)
```

### Finding best focus
```python
from src.analysis.zstack import find_focus_plane

result = find_focus_plane(stack, metric='laplacian')
best_z = result['best_z']
focus_scores = result['scores']
# Laplacian variance is best for general focus detection
# 'gradient' works well for edge-rich images
# 'std' works for images with varying content
```

### Volume measurement
```python
from src.analysis.zstack import measure_volume

result = measure_volume(stack, threshold=100,
                        pixel_size_xy=0.5, pixel_size_z=2.0)
volume_um3 = result['volume_um3']
# For spheroids: compare with 4/3 π r³ for sphere validation
```

### Cross-sections
```python
from src.analysis.zstack import slice_orthogonal

xz = slice_orthogonal(stack, axis='xz', position=256,
                       pixel_size_xy=0.5, pixel_size_z=2.0)
# xz['image'] is the cross-section, xz['extent'] for plotting
```

### Z-profiles
```python
from src.analysis.zstack import z_profile

# Profile at a specific point
result = z_profile(stack, position=(128, 128))
# result['peak_z'] — where this point is brightest
# result['fwhm_z'] — axial extent (None if not computable)

# Profile over an ROI
mask = np.zeros((512, 512), bool)
mask[100:200, 100:200] = True
result = z_profile(stack, roi_mask=mask)
```

## Common patterns

### Organoid lumen finding
1. Acquire Z-stack in BF
2. Use `project_std()` to find planes with sharp features
3. Look for central dark region (lumen) in best-focus plane
4. Use `ring_radii()` from ring.py to measure inner/outer radii

### 3D cell counting
1. MIP → count cells in projection (may merge overlapping cells)
2. Better: process each Z-plane independently, deduplicate across Z
3. Use `to_world_coords()` to map detections to 3D positions

### Volumetric measurement
1. Threshold each plane (use Otsu on MIP for consistent threshold)
2. `measure_volume()` sums voxels above threshold
3. Correct for anisotropic voxels (pixel_size_z ≠ pixel_size_xy)

## Pitfalls
- **Anisotropic voxels**: Z-step is often 2-5× larger than XY pixel size
- **Bleaching**: Minimize fluorescence Z-stack frames — use BF for structural Z-scanning
- **Large stacks**: >100 planes × 512×512 = 25 MB — process plane-by-plane if memory limited
- **MIP artifacts**: overlapping structures merge in MIP — use per-plane analysis for counting
