# Pre-submit guard architecture

The 2026-04-26 evening sprints (sprints #14–#23 + audit closures) hardened the agent against a recurring failure mode: *the answer feels right, but the rendered image disagrees with the value being submitted*. The architecture that emerged is layered — three guards in increasing-cost order, plus an opt-in orchestrator that composes them. This note captures the *why* of that shape so future contributors don't re-relitigate the design.

## The failure modes the guards were designed against

Three lower scores motivated the layering, each surfacing a different class of bug:

| Failure | Where the answer broke | What guard catches it |
|---|---|---|
| ch593 r3 (4/10) | per-element list — one well's IC50 disagreed with the rendered curve | render_vs_submit (list-shape, max-element delta) |
| ch599 (5/10) | scalar count — agent submitted "192 sim cells", image had 187 | render_vs_submit (scalar relative-error) |
| ch348 r2 (8/10) | MDA truncation silently shortened the FRAP recovery curve | mda_diagnostics (`expected_frames`) |

These are three orthogonal bugs that share one symptom: the agent is *internally consistent* (the analysis matches its own intermediate numbers), but the answer disagrees with what's actually in the rendered image or what the engine actually delivered.

## The three-tier ladder

```
┌─────────────────┐  cheap, deterministic, scalar/range checks
│ 2a. preflight   │  → NaN / Inf / missing key / sign error / fractional
│                 │     count. Run first, always, on every submission.
└─────────────────┘
         ↓ pass
┌─────────────────┐  deterministic, image-on-input, re-derives from
│ 2b. render_vs_  │  the recipe's own detector. Catches the
│     submit      │  apparent-vs-underlying class.
└─────────────────┘
         ↓ pass
┌─────────────────┐  LLM subagent, expensive, visual + code review.
│ 2c. pre-submit- │  Catches what 2a/2b miss — rendered overlay
│     review skill│  doesn't match the answer, code uses the wrong
│                 │  primitive, etc.
└─────────────────┘
         ↓ pass → submit_solution
```

Each guard catches a distinct failure class. They compose because they're in increasing order of cost: a preflight error short-circuits 2b (no need to run the detector if NaN is in the answer), and a 2b block short-circuits 2c (no need to spend an LLM call on a numerically-wrong answer). The reverse — running 2c first — would burn budget on submissions 2a would catch in milliseconds.

## Why three modules instead of one

Early in the session there was a temptation to fold all three into a single monolithic `pre_submit_check.py`. That was rejected because:

1. **Each module has a different cost profile.** preflight is microseconds; render_vs_submit re-runs a detector (milliseconds); pre-submit-review dispatches an LLM (seconds + $$). A monolithic call hides the cost; a layered API makes each cost visible at the call site.
2. **Each module has a different failure surface.** Shape-based checks (preflight) want a dict; image-based checks (render_vs_submit) want an image + detector callable; LLM checks (pre-submit-review) want a *directory* of inputs. Cramming them into one signature creates a ~15-arg function where every recipe sets some args and ignores others.
3. **Each module has a different "owner."** preflight rules drift from incident reports; render_vs_submit thresholds are recipe-specific; pre-submit-review prompt+overlays are agent-skill-shaped. Separate modules let owners ship independently.

## The orchestrator (sprint #22)

`run_presubmit_guards` is the *opt-in* one-call form. It composes 2a + 2b + 2c-prep with early-exit-on-block and aggregates severity (error > block > flag > ok). It is **not** mandatory — direct callers of `run_preflight` / `render_vs_submit_check` keep working.

The orchestrator stays opt-in because:

- The recipe author already knows which checks apply. A recipe that submits no scalar count doesn't benefit from render_vs_submit; one that submits no positions doesn't benefit from the position-list path. Forcing the union path through every recipe is overhead.
- The orchestrator's value is consolidation when *all three* fire on the same submission. That's a substantial fraction of recipes but not all.
- An opt-in API makes the migration story cleaner: existing recipes keep working; new recipes pick up the orchestrator when they want.

## Re-detect contract (the load-bearing piece)

`render_vs_submit_check` requires `recipe_re_detect_fn` — the recipe's *own* detector, closing over the same kwargs the recipe used internally. The check is "the value I'm about to submit equals what my detector finds when re-run on the rendered image." A *generic* detector defeats the purpose — it would be checking against a *different* detector, which is a different question.

This contract is what makes 2b useful. The grader feedback for ch599 was specifically: "the count you submitted is the constructor's count, not what the renderer shows." A re-derivation closes that gap because if the detector finds the same value the agent submitted, the agent is consistent with the renderer; if it finds a different value, the disagreement is exactly the bug.

The same shape generalises beyond render_vs_submit: any time you have **two independent estimators of the same quantity**, you have a guard. Agreement validates both; disagreement is diagnostic. ch609's three-round resilience arc (5 → 6 → 10) was driven by this — the Brenner pre-scan estimate of `tissue_z` (`z_focus_approx`) and the off-axis-LED-shift estimate (`-dx/slope`) are independent measurements; comparing them validated the slope calibration in the same way render_vs_submit validates a count submission. See [[Core/Strategies/Closed-loop autofocus]] ch609 entry and the durable feedback memory `feedback_two_estimator_cross_check.md` (rule: agree = validated, disagree = bug, never naively combine).

## What's outside the architecture

Three things this layered guard does **not** do, and why:

1. **It does not invoke the LLM from Python.** The pre-submit-review skill is dispatched via the agent's normal skill mechanism. The orchestrator only *prepares* the input directory at `/tmp/ch<N>_pre_submit/` and returns a path. This keeps the architecture deterministic and the LLM call on the user's terms.
2. **It does not gate `submit_solution`.** Every guard returns a result; the *caller* decides whether to ship, abort, or escalate. Coupling submission to guard outcome would force the architecture to predict caller policy.
3. **It does not auto-tune the guards' thresholds.** `rel_tol_flag=0.05`, `rel_tol_block=0.20`, `recall_floor=0.8`, etc. are all explicit kwargs. Auto-tuning by mining `session_audit` was considered and rejected — it would couple the guard's correctness to the historical mix of challenges, and the guard's job is to be *physically* meaningful, not statistically calibrated.

## The MDA-truncation guard sits orthogonal

`mda_diagnostics.run_events_checked` (sprint #15) operates one layer below the answer-shape guards. It catches a different bug — the proxy returned fewer frames than the recipe asked for — and the recipe is upstream of the guards. Sprint #15.1 rolled it out across the 4 acquisition-touching recipes with per-phase block/warn policy:

- N-frame-sensitive metrics (Q10 fits, time-resolved enrichment, peak-of-stack sharpness) → `raise_on_mismatch=True`.
- Discard / threshold-locking phases (warmup, equilibration, baseline) → `raise_on_mismatch=False, warn_threshold=0.5`.

The split exists because losing 1 of 5 baseline frames is fine but losing 1 of 3 final-snap frames corrupts a median. The recipe author writes the policy; the orchestrator lives at the answer layer above and trusts the recipe's frame count.

## Discoverability

Three discovery paths into this architecture:

1. **`status_summary()` → fast pulse.** Recent scores, running servers, message-queue inbox.
2. **`python -m src.core.utils.session_audit` → drift check.** Per-recipe / per-core-module score distributions, brittle-recipe flags, rolling-window trend (sprint #23).
3. **`scratch/template_auto_solve.py` → starter code.** `auto_recipe` → run → `run_presubmit_guards` → `submit_with_showcase`. (The historical `first_contact` composer paired `auto_recipe` with `bridge_preflight`, but the bridge RPC surface was closed 2026-04-27; only the image-level half survives.)

## See also

- [[Core/Approach/Pre-submission checklist]] — the procedural checklist that orders 2a/2b/2c.
- [[Core/Pitfalls/Sim-state vs rendered count asymmetry]] — the specific pitfall render_vs_submit was built against.
- [[Core/Pitfalls/MDA silent truncation on proxy]] — the truncation pitfall and the diagnostic that surfaced it.
- `src/core/utils/preflight.py`, `render_vs_submit.py`, `presubmit.py`, `mda_diagnostics.py` — the modules.
