# Multi-channel, multi-position MDA survey

> **When to use:** When acquiring multiple fluorescence channels at multiple stage positions in a single coordinated MDA sequence.

## The pattern

Most "scan N positions × M channels, then analyze per-position" workflows
should use a **single `MDASequence`** that visits every position, snapping
every channel at each — not a `for pos in positions: move_to; for ch in
channels: snap` loop.

```python
from self_learn.workflows.batch import multichannel_scan

r = multichannel_scan(
    core,
    positions=[(0, 0), (256, 0), (0, 256), (256, 256)],
    channels=['DAPI', 'bodipy-channel', 'membrane'],
    analyze_fn=lambda imgs: {
        'n_cells':  count_nuclei(imgs['DAPI']),
        'droplets': count_droplets(imgs['bodipy-channel']),
    },
)

# r['per_tile'] is a list; each entry has:
#   'position': (x, y)
#   'channels': {'DAPI': np.ndarray, 'bodipy-channel': np.ndarray, ...}
#   (+ whatever analyze_fn returned)
# r['aggregate'] is aggregate_results() over analyze_fn outputs.
```

The channel group is auto-discovered from the core when `group=None`.

## Why MDA over snap-loops

1. **Hardware engines expose MDA events**: autofocus-per-position,
   hardware-triggered acquisition, SLM patterns, and metadata are native
   to `MDAEvent` but silent in a snap loop. Real microscopes (Nikon Ti,
   Zeiss Axio, custom) often require MDA for correct sequencing.
2. **Portable across real and simulated cores**: `run_events(core, seq)`
   dispatches via `core.mda.run()`, the same entry point used by pymmcore-plus
   on real hardware. A snap loop works only if the backend is lenient.
3. **Per-position bucketing is free**: `event.index` gives `(p, c, t, z, ...)`
   so sorting frames by position / channel / timepoint is a dict lookup.
4. **Preflight checks catch bad channel names before acquisition starts**.
5. **One call = one clear log line** instead of interleaved move / snap
   messages. Easier to debug.

## When NOT to use it

- **Closed-loop / adaptive**: if the next position depends on the current
  frame's analysis, use an *MDA generator* that yields events based on
  `shared_state` — not a plain `MDASequence`. See `adaptive_survey_mda`
  and `Feedback control.md`.
- **Single-position single-channel snap**: one-off preview snaps are
  fine as `snap(core, channel=...)`; MDA has overhead that isn't worth
  it for a lone frame.

## Related helpers

- `workflows.batch.tile_and_analyze` — grid-based single-channel
  scan (computes positions from a `(rows, cols)` grid).
- `workflows.multi_scale.multi_position_events` — lower-level
  helper that just builds the event list (no acquisition, no analysis).
- `hardware.config.resolve_channel_group(core, group)` — used
  internally by multichannel_scan to default `group=None` to the runtime
  channel group.

## References

- Implementation: `workflows/batch.py::multichannel_scan`.
- Example use case: hepatocyte steatosis imaging (DAPI+bodipy+membrane at 10x).

## Literature

- [[Papers/Almada 2019]] — multiplexed
  labelling rounds (STORM / DNA-PAINT) extend the channel axis *beyond*
  the spectrally separable fluorophores by exchanging probes between
  imaging rounds. Each fluid-exchange cycle adds a new effective channel
  at the same excitation wavelength.
- [[Papers/Lutz 2018, Rames 2023]] — two
  controller-driven multiplexing strategies for DNA-PAINT: PRIME-PAINT
  (fluidic exchange — wash and inject the next imager strand) and
  Quencher-Exchange-PAINT (exchange a quencher rather than the imager
  itself, faster cycles, no flow). Each round is one effective channel;
  the cycle count is bounded by photobleaching and by the controller's
  ability to track ROI registration across washes. Sister to ch606's
  live → fix → probe Fluidics state device, generalised to N rounds.
