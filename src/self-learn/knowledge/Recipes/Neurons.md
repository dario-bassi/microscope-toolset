# Playbook: Neuron Analysis

## When to Use
- Sample shows neurons with somata (cell bodies) and dendrites
- Tasks: count neurons, classify types (pyramidal/stellate), count branch points, measure neurite length

## Step 1: Visual Inspection

```python
img = snap(core, "brightfield")
from PIL import Image; Image.fromarray(img).save("/tmp/neurons.png")
```

Look for:
- Large bright somata (cell bodies, 20-50px diameter at 10x)
- Branching dendrites extending from somata
- Multiple neurons? How close together? (affects branch assignment)

## Step 2: Soma Detection

**Preferred (2026-04-24+, ch588 r4 validated at F1=0.67):** use aggressive
morphological opening on MAP2 to strip dendrites to soma cores.

```python
from src.recipes.neuron_puncta import segment_soma_cores
somata = segment_soma_cores(map2, opening_radius=7, min_area=50)
# Each entry: skimage regionprops (centroid, area, etc.)
```

The opening radius should exceed typical dendrite half-width (~3 px at 0.5 µm/px).
`disk(7)` (~14 px footprint) works at standard 20x imaging; scale with px size.

**Fallback** (older API, radius-based filter): `src.core.detection.neurons.detect_somata(img, min_radius=5, max_radius=30)` — has pixel-scale defaults and may over-detect on dense dendritic trees. Use only when a distance-transform-based result is preferred.

## Step 2b: Puncta Per Neuron (full pipeline)

```python
from src.recipes.neuron_puncta import count_puncta_per_neuron
r = count_puncta_per_neuron(
    map2, synaptophysin,
    opening_radius=7,
    puncta_thresh_factor=0.03,  # permissive — catches dim lognormal tail
)
# Returns: n_neurons, n_puncta, puncta_per_neuron, *_positions_rc
```

## Step 3: Branch Point Counting (CRITICAL)

```python
from src.core.detection.neurons import count_branch_points
```

**MUST use connected-component assignment, NOT Voronoi:**
- Skeletonize the image to get 1-pixel-wide dendrites
- Find junction pixels (3+ neighbors in 8-connectivity)
- Assign each junction to the soma whose skeleton connected component it belongs to
- Voronoi partitioning FAILS when neurons are close together (junctions from one neuron's dendrites get assigned to the wrong soma)

**"Most branched" = most dendrite BIFURCATIONS (junction points), not most primary processes from soma.**

## Step 4: Neuron Classification

Visual classification works better than pure computation here:
- **Pyramidal**: one dominant apical dendrite + basal dendrites, triangular soma
- **Stellate**: dendrites radiating in all directions, round soma
- **Bipolar**: two main processes extending from opposite poles

Save a zoomed crop of each neuron and use LLM vision to classify.

## Step 5: Neurite Length

```python
from src.core.detection.neurons import estimate_neurite_length

# Pass the MAP2 channel directly; the helper thresholds + distance-
# transforms it. Returns total neurite length in pixels.
length_px = estimate_neurite_length(map2_channel, threshold=15)
length_um = length_px * pixel_size_um
```

- Skeleton-area / mean-width approximation gives neurite length without
  requiring skimage.morphology.skeletonize (broader compatibility).
- Convert pixels to microns using `pixel_size_um`.

## Optogenetics / Calcium Imaging (ChR2 + GCaMP)

### Channel Identification (CRITICAL)
GCaMP is a GFP-based calcium indicator. It shares the GFP filter set with structural
markers like MAP2. **Check which channel carries calcium signal** — it may be the same
channel as the structural marker, not a separate "calcium" channel.

- Snap each channel and measure at known soma ROIs
- Apply SLM to one neuron and re-snap — the channel showing intensity change = calcium channel
- Structural signal (MAP2) provides constant baseline F₀; GCaMP adds on top during firing

### Protocol: Before/After SLM Stimulation
```python
from useq import MDAEvent, SLMImage
from src.core.hardware.core import run_events, make_slm_circle

# 1. Detect somata from structural channel (MAP2/GFP)
# 2. Select target neuron (largest, most centered)
# 3. Create SLM mask targeting soma
slm_mask = make_slm_circle((cx, cy), radius=20, size=512)
slm = SLMImage(data=slm_mask, device="SLM")

# 4. Baseline: N frames WITHOUT SLM (spaced 2s for clean sampling)
baseline = [MDAEvent(channel={"config": "nucleus-channel", ...},
                     min_start_time=i*2.0) for i in range(8)]

# 5. Stimulation: N frames WITH SLM (each snap activates ChR2)
stim = [MDAEvent(channel={"config": "nucleus-channel", ...},
                 slm_image=slm, min_start_time=i*1.5) for i in range(10)]

# 6. Recovery: N frames WITHOUT SLM (watch calcium decay)
recovery = [MDAEvent(channel={"config": "nucleus-channel", ...},
                     min_start_time=i*2.0) for i in range(8)]

# Run each phase via run_events() with on_frame callback
```

### Analysis: ΔF/F
- **ΔF/F = (F_stim - F_baseline) / F_baseline** — standard calcium imaging metric
- Measure mean intensity within soma ROI (labeled region or circular mask)
- Target neuron: expect massive ΔF/F (>500% for strong ChR2 expression)
- Control neurons: near 0% (only rare spontaneous events at freq ~0.02)
- Recovery: calcium decays to baseline within ~4s (2 frames at 2s interval)

### Timing Considerations
- Allow 10s+ for residual calcium to decay before starting baseline
- Space baseline frames sufficiently (2s) for independent samples
- SLM stimulation is observation-coupled: blue light only during exposure
- In realtime mode: dynamics continue between snaps

## Connectivity Mapping (SLM Stimulation Protocol)

Map directed synaptic connections between neurons using ChR2/GCaMP.

### Protocol (scored 10/10 on ch440, refined for real-time in ch462)
```
For each neuron i in [0..N-1]:
  1. Snap GCaMP baseline (all neurons at resting level)
  2. Create SLM mask: circle r=20px centered on neuron i's soma
  3. Apply SLM + snap GCaMP via MDAEvent(slm_image=...) (stim frame)
  4. Snap 6 post-stim GCaMP frames with time.sleep(0.1) between
     (100ms = 1 synaptic hop delay for GCaMP6f)
  5. Wait 4s for refractory period (calcium decay half-life ~500ms)
```

### Connection Detection (real-time protocol, ch462=8/10)
- **1-hop direct**: post-1 frame neuron goes from resting (<50) to activated (>120)
- **2-hop indirect**: appears in post-2 frame (200ms after stim)
- **CRITICAL (ch462 lesson, cost 2 points)**: Must distinguish 1-hop from 2-hop!
  Checking only post-1 amplitude catches ALL responders including 2-hop cascades
  that propagate quickly. Use BOTH timing AND amplitude:
  - Direct: responds in post-1 (100ms) AND NOT in post-0 (stim frame)
  - Indirect: may also appear in post-1 if timing resolution is ~100ms
  - Better: check if neuron is ALREADY responding in stim frame (latency < 100ms = direct),
    or only in post-1 (latency ~100ms = could be direct or fast 2-hop)
  - Best approach: require post-1 response AND post-2 should show DECAY (not further increase)
    for direct connections. 2-hop neurons continue to increase from post-1 to post-2.
- **Previously-stimulated neurons**: if baseline > 50, can't tell if new activation → exclude
- Direction: stim source → post-1 responder = directed edge

### Reusable Module
```python
from src.core.workflows.optogenetics import connectivity_mapping
result = connectivity_mapping(core, soma_positions,
    channel_config='nucleus-channel', slm_radius=20,
    n_post_frames=6, post_interval=0.1, refractory_wait=4.0)
adjacency = result['adjacency']
n_connections = result['n_connections']

from src.core.analysis.functional_connectivity import graph_metrics
metrics = graph_metrics(adjacency)
hub = metrics['hub']  # highest total degree neuron
```

### Key Lessons
- **SLM mask radius**: r=20 for robust stimulation. Stimulated neuron should reach ~190.
- **Refractory**: 4s wait between stimulations is sufficient. Direct SLM stimulation
  produces longer-lasting elevation than synaptic activation.
- **Elevated baseline masking**: Sequential stimulation means the PREVIOUS neuron
  may still be elevated. Check `baseline < resting_threshold` before counting as new response.
  This can mask 1-2 connections per run.
- **Real-time vs observation-coupled**: With real-time dynamics, use time.sleep()
  for actual propagation delay. Do NOT snap empty frames to advance time.
- **Bidirectional connections**: A→B + B→A creates reciprocal pairs. Use `reciprocity`
  metric from graph_metrics().

### Analysis Module
```python
from src.core.analysis.functional_connectivity import (
    cascade_connectivity,  # Infer directed graph from stimulation cascade
    graph_metrics,         # Degree, clustering, hub detection
    correlation_matrix,    # Pairwise correlation from spontaneous activity
    lagged_correlation,    # Directionality from time delays
    build_adjacency,       # Threshold correlations into binary graph
)
```

## Common Pitfalls

- **Voronoi assignment for branch points**: WRONG when neurons overlap. Always use connected-component assignment.
- **Counting primary processes instead of bifurcations**: "branch points" means skeleton junctions, not processes leaving the soma.
- **Spurious small somata**: Filter by minimum radius (>5px at 10x).
- **Voronoi picks wrong neuron**: When neurons are close, Voronoi assigns junctions from overlapping dendrites to the wrong soma, changing which neuron appears "most branched".
- **Wrong calcium channel**: GCaMP signal may appear in structural marker channel (same GFP filter). Always verify by test stimulation before running full experiment.
- **Not waiting for calcium decay**: Rapid sequential stimulations won't show clear baseline. Wait ≥10s between phases.
