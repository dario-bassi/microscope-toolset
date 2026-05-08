# Playbook: Disk Diffusion ZOI Measurement

## When to Use
- Antibiotic susceptibility testing (Kirby-Bauer)
- Tasks: measure zone of inhibition (ZOI) diameter around antibiotic discs
- Classify susceptibility as S/I/R using CLSI breakpoints

## Step 1: Visual Inspection

```python
img = snap(core, "plate-image")  # or "brightfield"
from PIL import Image; Image.fromarray(img).save("/tmp/zoi.png")
```

Look for:
- Circular agar plate with bacterial lawn (slightly brighter than bare agar)
- 4 antibiotic paper discs (bright white dots, ~6px radius)
- Clear zones around each disc (darker/smoother than lawn)
- Disc positions are typically symmetric around plate center

## Step 2: Disc Detection

Paper discs appear as small bright dots. Find them via:
```python
# Option 1: peak_local_max on smoothed image for bright spots
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
smoothed = gaussian_filter(img.astype(float), sigma=3)
peaks = peak_local_max(smoothed, min_distance=60, threshold_abs=threshold)

# Option 2: If symmetric layout, discs at known quadrant positions
# Typical: 4 discs at ~1/3 and ~2/3 of plate diameter
```

## Step 3: Texture-Based Zone Detection (BEST METHOD)

**Key insight**: Intensity difference between lawn and zone is only ~5%. But TEXTURE difference is ~60%:
- Bacterial lawn = NOISY (high local std, textured colony growth)
- Bare agar (zone) = SMOOTH (low local std, uniform)

```python
from scipy.ndimage import uniform_filter, gaussian_filter

# Compute local texture (standard deviation in small window)
img_f = img.astype(np.float64)
win = 9  # window size — tune to match colony texture scale
mean = uniform_filter(img_f, size=win)
sq_mean = uniform_filter(img_f ** 2, size=win)
local_std = np.sqrt(np.maximum(sq_mean - mean**2, 0))

# Optional: ratio-normalize for illumination correction
# Use sigma >> zone radius to avoid normalizing zones out
bg = gaussian_filter(img_f, sigma=max(img.shape)//6)
ratio = img_f / np.maximum(bg, 1)
# Then compute local_std on ratio image
```

## Step 4: Radial Profile from Each Disc

For each disc center, compute azimuthal average of local_std vs distance:
```python
from scipy.signal import savgol_filter

yy, xx = np.ogrid[:H, :W]
dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
radii = np.arange(5, max_r)
profile = np.array([local_std[(dist >= r-1) & (dist < r+1)].mean() for r in radii])
smooth = savgol_filter(profile, window_length=11, polyorder=2)
```

## Step 5: ZOI Edge Detection (CRITICAL)

**ZOI edge = steepest gradient in texture profile** (where lawn texture appears).

```python
gradient = np.gradient(smooth)
edge_idx = np.argmax(gradient[10:]) + 10  # skip disc region
zoi_radius = radii[edge_idx]
zoi_diameter = 2 * zoi_radius
```

**Why texture works better than intensity**:
- Intensity profile: zone ~195, lawn ~200 (2.5% difference) → hard to threshold
- Texture profile: zone ~1.3, lawn ~2.3 (77% difference) → clear step edge

**Avoid 50% intensity threshold**: Overestimates ZOI by ~40% (gave ZOI=158 vs GT=113).

## Step 6: Classification (CLSI Breakpoints)

```python
# Example breakpoints (check challenge-specific values):
breakpoints = {
    'Ampicillin':    {'S': 100, 'R': 60},   # S >= 100, R <= 60
    'Tetracycline':  {'S': 76,  'R': 52},
    'Gentamicin':    {'S': 90,  'R': 60},
    'Erythromycin':  {'S': 70,  'R': 44},
}

def classify(diameter, bp):
    if diameter >= bp['S']: return 'S'
    elif diameter <= bp['R']: return 'R'
    else: return 'I'
```

## Antibiotic-to-Disc Mapping
- If not specified, assume order matches breakpoint table listing
- Typical layout: UL, UR, BL, BR (reading order)
- Verify by checking zone sizes against expected resistance patterns

## Common Pitfalls

- **Intensity-only detection fails**: Lawn/zone intensity difference is <5% — use texture instead
- **50% threshold overestimates**: The transition from clear to lawn is gradual
- **Steepest gradient is the standard**: Clinical ZOI = where the edge is sharpest
- **Disc center masking**: Exclude disc area (~6px radius) from radial profile — discs have very high texture (bright spot against agar)
- **Asymmetric zones**: Measure at multiple angles and average, or report min/max
- **Flat-field correction pitfall**: If sigma is similar to zone size, correction normalizes zones out
- **CLAHE pitfall**: Tile-based enhancement creates uneven contrast across plate
