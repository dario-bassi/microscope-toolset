# Workflow: Neural Circuit Connectivity Mapping

## Overview
Map functional connectivity in a neural network using optogenetic stimulation
(ChR2) and calcium imaging (GCaMP). Systematically stimulate each neuron via
SLM and observe which downstream neurons respond.

## Protocol

### Acquisition
Per neuron stimulation cycle:
1. **Baseline** (SLM off) — image all neurons at resting intensity
2. **Sustained SLM stimulation + rapid imaging** — target neuron fires under
   ChR2 activation, cascade propagates to downstream neurons:
   - Within ~0.5s: target fires
   - After 1 synaptic delay (~0.5s): direct (1-hop) connections fire
   - After 2 delays (~1s): indirect (2-hop) connections fire
3. **SLM off + recovery** — wait for calcium to decay to baseline before
   stimulating the next neuron (several seconds, depends on indicator)

### Key Principles
- **SLM stays ON during observation** — target must sustain firing while you
  image downstream responses. Turning off too early kills the cascade.
- **Image fast enough** to resolve cascade timing — frame rate should be
  faster than the synaptic delay (~100-500ms intervals)
- **Sufficient recovery time** between stimulations — calcium must return
  to baseline before stimulating the next neuron
- **ΔF/F detection** — scale-invariant, not dependent on absolute thresholds

### Detection
```python
from self_learn.workflows.optogenetics import infer_connectivity_dff

# After collecting stim_data via on_frame callback:
result = infer_connectivity_dff(state, n_neurons, dff_threshold=2.0, direct_only=True)
# result['adjacency'] = {src: [tgt, ...]}
# result['n_connections'] = int
# result['details'] = [{'src', 'tgt', 'dff', 'peak_snap', 'baseline', 'peak'}]
```

- **ΔF/F threshold 2.0** = response must be ≥3× baseline intensity
- **direct_only=True** filters out multi-hop connections (peak at snap 2+)
- **peak_snap=0** → direct (1-hop) connection
- **peak_snap≥1** → indirect (multi-hop relay)

### MDA Generator Pattern
```python
from useq import MDAEvent
from self_learn.hardware.core import run_events, make_slm_circle

clear_mask = np.zeros((512, 512), dtype=np.uint8)

def connectivity_events():
    for stim_idx in range(n_neurons):
        cx, cy = soma_positions[stim_idx]

        # Baseline (SLM off)
        core.setSLMImage("SLM", clear_mask)
        core.displaySLMImage("SLM")
        yield MDAEvent(channel=ch_kw, exposure=50,
                      metadata={'neuron': stim_idx, 'phase': 'baseline'})

        # SLM on for 3 observation snaps
        slm_mask = make_slm_circle((cx, cy), radius=20, size=512)
        core.setSLMImage("SLM", slm_mask)
        core.displaySLMImage("SLM")
        for si in range(3):
            yield MDAEvent(channel=ch_kw, exposure=50,
                          metadata={'neuron': stim_idx, 'phase': 'slm_on',
                                    'snap_idx': si})

        # Decay (SLM off, 8 snaps)
        core.setSLMImage("SLM", clear_mask)
        core.displaySLMImage("SLM")
        for d in range(8):
            yield MDAEvent(channel=ch_kw, exposure=50,
                          metadata={'neuron': stim_idx, 'phase': 'decay'})

run_events(core, connectivity_events(), on_frame=on_frame)
```

## Common Pitfalls

1. **Turning SLM off too early** — downstream neurons need sustained stimulation
   of the source to show cascade. Turning off after 1 snap kills the cascade.

2. **Using SLMImage in MDA events** — fails over pymmcore-proxy (numpy array
   serialization error). Use manual `setSLMImage`/`displaySLMImage` between
   generator yields instead.

3. **Absolute intensity thresholds** — `_infer_adjacency` uses resting/response
   thresholds that don't generalize across samples. Use `infer_connectivity_dff`
   with normalized ΔF/F instead.

4. **Including false positive neurons** — detect from fluorescence channel
   (GCaMP baseline), not just brightfield. Filter out detections with near-zero
   fluorescence signal.

5. **Including indirect connections** — 2-hop relays peak at snap 2+.
   Use `direct_only=True` or filter by `peak_snap == 0`.

## Module Functions
- `detect_somata_fluorescence(img)` — CC-based soma detection
- `connectivity_mapping(core, positions, ...)` — full acquisition + detection
- `connectivity_mapping_events(core, positions, ...)` — MDA-native (returns generator)
- `infer_connectivity_dff(state, n, dff_threshold=2.0)` — ΔF/F detection
- `_infer_adjacency(state, n, resting_thresh, response_thresh)` — legacy absolute thresholds
