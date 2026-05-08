# MDA silent truncation on the proxy

## What goes wrong

You ask the proxy for a 30-frame `MDASequence` timelapse with `interval=3.0`. The runner logs `t=0..29` cleanly. Your `on_frame` callback collected 19 frames. No exception. No visible warning unless you look hard. Downstream analysis (FRAP recovery curve, drift trajectory, growth rate) is fit to the truncated series and the answer is wrong by exactly the right amount to look plausible.

The pitfall is that the **runner's t-index logging is on the proxy server's side; the `frameReady` callback is on yours**. When the WebSocket bridge between them blips — `ConnectionClosedOK`, `WebSocketDisconnect`, a flickering server, or a network hiccup mid-stream — the runner finishes its event list internally on the server (logging each `t={idx}` to the proxy stdout) but a subset of frames never make it across the wire to fire the local handler. The `run_events` retry-to-manual fallback re-establishes connection but **does not replay the events that fired during the disconnect window** to the original handler.

Empirical: a counter=53 probe against `http://127.0.0.1:5605` reproduced this on the second of three test acquisitions. A 10-frame × 5-s plan (45 s wall) silently delivered 8 frames; a 30-frame × 3-s plan (87 s) on the same server delivered all 30. The truncation is **stochastic**, not a hard timeout.

## When it bites

Long-ish or medium-length MDAs over a flaky proxy connection are the typical surface area:

- FRAP recovery curves, where missing the *late* frames silently tightens the apparent half-life.
- Drift-tracking timelapses, where missing intermediate frames lets the Hungarian linker conclude the sample didn't move much.
- Calcium / cardiac timelapses where one missing wave makes the period look slower.
- Any submission whose answer is a *function of N frames* and where N is silently low.

The ch348 r2 grade (8/10) attributed one missing field to this exact pattern. ch599 silently lost frames during the long MDA before the apparent-vs-underlying GT issue became the story.

## Why it's a pitfall, not a bug

`run_events` retries with `_manual_run` when it sees a recognised network error (`ConnectionClosed*`, `WebSocketDisconnect`, `TimeoutError`). The retry IS doing its job — the manual fallback re-snaps the rest of the event list. But the retry path was added before the `expected_frames` accounting from sprint #15, and there's no end-to-end "did the callback fire N times?" cross-check inside `run_events` itself.

Each layer is correct in isolation. The pitfall is in the gap between them.

## Right fix

Use `src/core/utils/mda_diagnostics.py:run_events_checked(core, events, expected_frames=N, on_frame=cb)` (sprint #15). It returns `(results, report)` where `report.is_truncated` is the cross-check the bare `run_events` is missing:

```python
from src.core.utils.mda_diagnostics import run_events_checked

results, report = run_events_checked(
    core, list(seq), expected_frames=30, on_frame=on_frame,
)
if report.is_truncated:
    raise RuntimeError(
        f"MDA truncated: asked {report.expected_frames}, "
        f"got {report.actual_frames} ({report.truncation_report})"
    )
```

For recipes that submit frame-count-sensitive metrics, the `expected_frames` kwarg is already wired into `analyze_frap` (and similar). Use it.

As of sprint #15.1 (counter=54), the following recipes have explicit
truncation guards: `frap_background_correction` (via `analyze_frap`'s
`expected_frames` kwarg), `bacteria_trap` (accumulation + final phases
block, baseline warns), `temperature_experiment` (measurement phase
blocks, equilibration warns; both v1 and v2), `event_driven_modality_switch`
(detail scan blocks, warmup warns), `cybergenetic_per_cell_control`
(post-hoc baseline-frame check; the adaptive feedback phase has no static
N and is not guarded). `adaptive_sted_burst` is exempt — each call is
`loops=1` and a truncation surfaces as `IndexError` immediately. Snap-loop
recipes (`plate_reader_ic50`) are out of scope: they don't go through
the MDA engine and the WebSocket-mid-MDA-drop pattern doesn't apply.

For one-off scripts where you can't afford to add the import, the same check inline:

```python
captured = []
run_events(core, list(seq), on_frame=lambda img, ev, m=None: captured.append(ev.index['t']))
expected = len(list(seq))
if len(captured) < expected:
    raise RuntimeError(f"MDA truncated: expected {expected}, got {len(captured)}")
```

## Defensive habits

- Never trust `len(captured)` blindly — log the gap.
- Watch for the `UserWarning: MDA engine error (...) — retrying with manual acquisition` line in stderr; if it appears, count frames before believing the result.
- For long-running timelapses, prefer one shorter MDA over a single big one, with `expected_frames` checked between segments. The retry path runs once per `run_events` call.

## See also

- `src/core/utils/mda_diagnostics.py` — the `run_events_checked` countermeasure and `MDAFrameTruncation` report dataclass.
- [[Core/Strategies/Closed-loop autofocus]] — long-running drift-correction loops are the canonical target for this pitfall; `track_focus_brownian` already counts its corrective sweeps and exposes `n_snaps_total`.
- [[Core/Strategies/Adaptive acquisition]] — generator-driven MDAs should pair with `expected_frames` accounting.
- [[Core/Pitfalls/Sim-state vs rendered count asymmetry]] — sister pitfall on the answer side; this one is on the *acquisition* side.
- `src/core/utils/render_vs_submit.py` — agent-side numerical guard that catches the *downstream* effect (re-derived value disagrees with submitted value because of missing frames).
- [[Core/Approach/Pre-submit guard architecture]] — explains why this acquisition-layer guard sits orthogonal to the answer-shape guards (preflight / render_vs_submit) and why mixing them into the orchestrator was rejected.
