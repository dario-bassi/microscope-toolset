# Session retrospective

When you've been working on a project long enough that scores accumulate across many challenges, eyeballing the trend stops working — there are too many recipes, too many runs, and too many one-off tweaks to hold in your head. `src/core/utils/session_audit.py` (sprint #23 + #23.1) mines `logs/challenges/<id>_<slug>/{submission,grade}.json` and folds the history into structured statistics so you can ask one of three questions deliberately.

## The four questions session_audit answers

1. **Where am I overall?** `overall.mean / median / stdev / pass_rate / range` from `audit_session(...)["overall"]`.
2. **Which recipes are brittle?** `brittle_recipes(distribution, min_n=2, stdev_threshold=1.5)` returns recipes with high score variance — those are the candidates for hardening (extra tests, parameter audit, recipe-knowledge note refresh).
3. **What's drifting?** `trend_per_recipe(rows, window=5)` compares the last `window` submissions per recipe to the historical baseline and tags each as `improving` / `stable` / `degrading` / `insufficient_data`.
4. **Am I using the platform?** `inline_violation_rate(rows, window=20)` (sprint added 2026-04-28) returns the fraction of recent submissions with >30 code lines and zero `from src.` imports — a proxy for NON_NEGOTIABLES rule 4 ("no inline analysis >30 lines") drift. The format_report tags the rate with ⚠ at ≥20%; below that it's a routine signal.

## When to call the audit

- **Every 5 challenges** (per CLAUDE.md "Reflection & Knowledge"). The cheap pulse of `status_summary()` shows recent scores; the audit shows the *distribution* underneath them.
- **After a sprint that touched a recipe.** If sprint #15.1 added truncation guards to four recipes, the audit reveals whether the next four runs of those recipes show a `degrading` flag (the guards started firing on real submissions) or `stable` (the guards were a no-op in practice).
- **Before a knowledge-note refresh pass.** Recipes with `brittle` flags need their pitfall sections updated; recipes with `insufficient_data` flags need either more diverse test challenges or are too narrow.

## How to read the live snapshot

```
$ python -m src.core.utils.session_audit --window 5
=== session audit (n_graded=243) ===
overall: mean=7.45/10  median=8  stdev=1.94  pass_rate=73.3%  range=[2, 10]

Top 10 recipes by submission count:
  __inline_solve__               n=243  mean=7.49  stdev=1.95  range=[2, 10]
  frap_background_correction     n=  1  mean=8.00  stdev=0.00  range=[8,  8]
  ...

Top 10 utils by submission count:                  ← sprint #30
  __inline_solve__               n=247  mean=7.49  ...
  slm_masks                      n=  1  mean=10.00 ...
```

The `__inline_solve__` bucket dominating is expected — most legacy solves predate the recipe layer and inlined detection. Sprint #30 added the parallel `by_utils` distribution (`group_by="utils"`) which buckets `src.core.utils.*` imports separately from `src.recipes.*`; recent sprint #25-#33 utility-driven solves show up there once challenges exercise them. The bucket was renamed `__no_recipe__` → `__inline_solve__` for accuracy ("no extracted module attributed", not "no recipe at all"). Drift detection is most informative on `by_core_module` (autofocus, cells, lipid_droplet) where the 17 src/recipes/*.py imports show up early.

## What the audit will NOT tell you

The score is a *grade-bench-side* signal — useful for spotting brittleness but not for diagnosing root cause. When a recipe shows a `degrading` flag, the audit hands off to:

- The grader feedback for the most recent low-scoring rounds (`logs/challenges/<id>/grade.json` `feedback` field).
- The relevant pitfall note(s) in `Core/Pitfalls/`.
- The recipe's paired knowledge note in `Recipes/`.
- For per-call truncation: the `MDA silent truncation on proxy` pitfall + `mda_diagnostics.run_events_checked` countermeasure.

The audit is a *navigator*, not a diagnostician.

## See also

- `src/core/utils/session_audit.py` — `iter_history`, `score_distribution` (now with `group_by="utils"`), `brittle_recipes`, `trend_per_recipe`, `inline_violation_rate`, `audit_session` (now returns `by_utils` + `brittle_utils` + `utils_trends` + `inline_violations`), `format_report` (now renders Top 10 utils + NON_NEG #4 rate), `_main` CLI entry.
- `tests/test_session_audit.py` — 27 tests covering iter / aggregate / trend / format / inline-violation paths.
- [[Core/Approach/Pre-submit guard architecture]] — the layered guards whose effect the audit measures over time.
- CLAUDE.md "Reflection & Knowledge" — the procedural cadence the audit fits into.
