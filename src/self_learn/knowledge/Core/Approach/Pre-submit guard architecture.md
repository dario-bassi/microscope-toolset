# Measurement Verification Architecture

This note describes a layered architecture for verifying experiment
results before reporting them. The goal: catch cases where the
answer *feels* right but the rendered image disagrees.

## The failure modes the guards address

Three classes of bug share one symptom: the analysis is *internally
consistent* (the numbers match the code's own intermediate values),
but the result disagrees with what's actually in the rendered image
or what the acquisition engine actually delivered.

| Failure class | What went wrong | What guard catches it |
|---|---|---|
| Apparent-vs-underlying count | Submitted internal state count, image shows different | render_vs_submit (scalar relative-error) |
| List-element disagreement | One element of a result list disagrees with rendered data | render_vs_submit (list-shape, max-element delta) |
| MDA truncation | Acquisition silently delivered fewer frames than requested | mda_diagnostics (`expected_frames`) |

## The three-tier ladder

```
┌─────────────────┐  cheap, deterministic, scalar/range checks
│ 2a. preflight   │  → NaN / Inf / missing key / sign error / fractional
│                 │     count. Run first, always, on every result.
└─────────────────┘
         ↓ pass
┌─────────────────┐  deterministic, image-on-input, re-derives from
│ 2b. render_vs_  │  your own detector. Catches the
│     submit      │  apparent-vs-underlying class.
└─────────────────┘
         ↓ pass
┌─────────────────┐  Visual + code review. Catches what 2a/2b miss —
│ 2c. visual      │  rendered overlay doesn't match the result, code
│     review      │  uses the wrong primitive, etc.
└─────────────────┘
         ↓ pass → record results
```

Each guard catches a distinct failure class. They compose because
they're in increasing order of cost: a preflight error short-circuits
2b, and a 2b block short-circuits 2c. The reverse — running 2c first
— would burn budget on results 2a would catch in milliseconds.

## Why three modules instead of one

1. **Different cost profiles.** preflight is microseconds;
   render_vs_submit re-runs a detector (milliseconds); visual review
   is human/LLM time (seconds). A monolithic call hides the cost; a
   layered API makes each cost visible at the call site.
2. **Different failure surfaces.** Shape-based checks (preflight) want
   a dict; image-based checks (render_vs_submit) want an image +
   detector callable; visual review wants overlay images and analysis
   code. Cramming them into one signature creates a ~15-arg function
   where every recipe sets some args and ignores others.
3. **Different owners.** Preflight rules drift from incident reports;
   render_vs_submit thresholds are recipe-specific; visual review is
   qualitative and judgment-based. Separate modules let owners ship
   independently.

## The orchestrator (opt-in)

`run_presubmit_guards` is the opt-in one-call form. It composes
2a + 2b + 2c-prep with early-exit-on-block and aggregates severity
(error > block > flag > ok). It is **not** mandatory — direct callers
of `run_preflight` / `render_vs_submit_check` keep working.

The orchestrator stays opt-in because:
- The recipe author knows which checks apply. A recipe with no scalar
  count doesn't benefit from render_vs_submit.
- The orchestrator's value is consolidation when *all three* fire on
  the same result. That's a substantial fraction of recipes but not all.

## Re-detect contract (the load-bearing piece)

`render_vs_submit_check` requires `recipe_re_detect_fn` — your *own*
detector, closing over the same kwargs used internally. The check is:
"the value I'm about to record equals what my detector finds when
re-run on the rendered image."

A *generic* detector defeats the purpose — it would be checking
against a *different* detector. The contract is: if the detector finds
the same value you computed, you're consistent with the image. If it
finds a different value, the disagreement is exactly the bug.

**Two independent estimators rule:** any time you have two independent
estimators of the same quantity, you have a guard. Agreement validates
both; disagreement is diagnostic. Never naively combine disagreeing
estimators — investigate the root cause instead.

## The MDA-truncation guard

`mda_diagnostics.run_events_checked` operates one layer below the
answer-shape guards. It catches a different bug — the proxy returned
fewer frames than requested — and the recipe is upstream of the guards.

```python
from self_learn.utils.mda_diagnostics import run_events_checked
frames = run_events_checked(core, events, expected_frames=N, on_frame=cb)
```

Per-phase block/warn policy:
- N-frame-sensitive metrics (kinetic fits, time-resolved enrichment,
  peak-of-stack sharpness) → `raise_on_mismatch=True`.
- Discard / threshold-locking phases (warmup, equilibration, baseline)
  → `raise_on_mismatch=False, warn_threshold=0.5`.

## What's outside the architecture

1. **Does not gate the result automatically.** Every guard returns a
   result; the *caller* decides whether to record, abort, or escalate.
2. **Does not auto-tune thresholds.** `rel_tol_flag=0.05`,
   `rel_tol_block=0.20`, `recall_floor=0.8` are all explicit kwargs.
   The guard's job is to be *physically* meaningful.

## See also

- [[Core/Approach/Experiment verification checklist]] — the procedural
  checklist that orders 2a/2b/2c.
- [[Core/Pitfalls/Sim-state vs rendered count asymmetry]] — the
  apparent-vs-underlying failure mode.
- [[Core/Pitfalls/MDA silent truncation on proxy]] — the truncation
  pitfall and its diagnostic.
- `utils/preflight.py`, `utils/render_vs_submit.py`,
  `utils/presubmit.py`, `utils/mda_diagnostics.py` — the modules.
