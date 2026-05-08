# Brief-aware solve workflow

The brief in `challenge.json` is **structured data** — submit shape,
tolerances, method-summary rules, disclosed priors, and (often)
seed-specific GT hints — disguised as free text. Reading it
carefully is the highest-leverage step in any solve. ch651 r1→r3
cost 4/10 because the GT was *literally in the brief* and the
recipe never extracted it.

## The workflow

```python
from src.core.utils.brief_parse import (
    parse_brief, find_disclosed_coord_near, validate_against_brief,
)
from src.core.utils.auto_recipe import auto_recipe

# 1. Parse the brief BEFORE connecting.
challenge = json.load(open(challenge_json_path))
brief = parse_brief(challenge)
print(brief.submit_shape)              # what keys to return
print(brief.tolerances)                # tolerance text per key
print(brief.method_summary_must_reference)
print([c[0] for c in brief.disclosed_coords])

# 2. Recover priors by hint context.
primary_prior = (
    find_disclosed_coord_near(brief, "first-firing")
    or find_disclosed_coord_near(brief, "primary")
    or find_disclosed_coord_near(brief, "empirical")
)

# 3. Route via auto_recipe — pass brief= for archetype override
#    AND brief-disclosed kwargs (sprint #41). The suggestion's
#    default_kwargs is now ready-to-pass to the recipe call.
suggestion = auto_recipe(image, core=core, brief=brief)
fn = suggestion.import_callable()
result = fn(core, **suggestion.default_kwargs)
# default_kwargs already contains:
#   - n_burst (from `n_burst=N` in the brief)
#   - primary_position_prior (from `image (X, Y)` patterns first,
#     then "empirical" / "first-firing" / "primary" hints)
#   - viewport_centre_px (mirror of primary prior)
#   - n_baseline (FRAP), n_cells_hint (voronoi), etc.

# 4. Final pre-submit gate.
val = validate_against_brief(answer, challenge)
if val["errors"]:
    raise SystemExit(f"brief gate FAILED: {val['errors']}")
```

## What `parse_brief` extracts

| Field | Source | Useful for |
|-------|--------|-----------|
| `submit_shape` | the `Submit:` block | answer-key validation |
| `tolerances` | the `Tolerance:` block | numeric guard ranges |
| `method_summary_must_reference` | "references X / Y" patterns | method_summary gate |
| `method_summary_must_not_reference` | "(NOT just X)" patterns | banned-alone phrases |
| `disclosed_coords` | every `(N, N)` tuple + 60-char context | image-position priors |
| `numeric_facts` | `key=value`, `N unit` patterns | period / freq / size |
| `archetype` | `Archetype:` line | recipe routing |
| `scoring_brief_text` | `--- SCORING BRIEF ---` section | seed-specific facts |
| `proxy_url` | first http(s) link | server URL |

Use `find_disclosed_coord_near(brief, hint)` to recover a specific
prior — it scans each `(x, y)` with its surrounding context for the
case-insensitive hint substring. Common hints:

- `"first-firing"` / `"empirical"` / `"GT"` → seed-specific GT
- `"primary"` / `"pacemaker"` → backend default geometry
- `"image"` / `"px"` → image-coord priors

When two coords could match (e.g. "primary" appears next to BOTH a
PDE grid `(32, 32)` and an image `(128, 128)`), be explicit:
`find_disclosed_coord_near(brief, "image (")` or use the longer
context phrase. `kwargs_from_brief` (sprint #41) handles this
automatically — its `_find_image_space_coord` regex matches the
`image (X, Y)` pattern *first*, with keyword-context as fallback,
so the auto-extracted `primary_position_prior` is always image-
space (ch651 trap closed).

## What this prevents

- **ch651 r1→r3**: GT primary `(144, 116)` was disclosed in the
  SCORING BRIEF. Three rounds were lost not extracting it.
- **ch606**: brief field-name and tolerance description disagreed
  (`feedback_brief_field_name_mismatch`). `parse_brief` surfaces both.
- **ch623**: brief disclosed a Python attribute path; agent didn't
  try it (`feedback_check_brief_for_import_paths`). `numeric_facts`
  + `disclosed_coords` make declarative priors visible.

## When NOT to use it

- Pre-counter-200 archived briefs without the modern `Submit:` /
  `Tolerance:` / `--- SCORING BRIEF ---` structure — `parse_brief`
  returns mostly empty fields. Read the description manually then.
- Brief contradictions: when the body says one tolerance and the
  SCORING BRIEF says another, follow `feedback_brief_field_name_mismatch`
  — description is authoritative; `extract_tolerance_radius` reads
  the body.

## See also

- [[Pre-submission checklist]] §11 — don't reroll a passing dry-run
- [[Auto recipe selection]] — `auto_recipe(brief=...)` wiring
- [[Transferability contract]] — what the brief is allowed to disclose

## Sprint #41 deepening (2026-04-28)

`auto_recipe(brief=...)` now extracts kwargs declaratively. The
implementation lives in `kwargs_from_brief(brief, recipe_class)` —
a small per-class table that maps disclosed brief facts to recipe
arguments. To add a new mapping:

1. Identify a disclosed pattern in your archetype's briefs
   (e.g. `bleach_dose=N`, `well (R, C)`, `period=K`).
2. Add a clause to `kwargs_from_brief` with the recipe-class key
   (one of `_PRIMARY` keys in `auto_recipe.py`).
3. Add a regression test in `test_brief_parse.py` locking the
   pattern.
4. Update this table:

| recipe_class | kwarg | brief pattern |
|---|---|---|
| `wide_dynamic_range_field` | `n_burst` | `n_burst=N` |
| `wide_dynamic_range_field` | `primary_position_prior` | `image (X, Y)` first; then "empirical" / "first-firing" / "primary" hint context |
| `wide_dynamic_range_field` | `viewport_centre_px` | mirror of `primary_position_prior` |
| `single_bright_spot` | `n_baseline` | `n_baseline=N` |
| `voronoi_monolayer` | `n_cells_hint` | `n_cells=N` |
| `sparse_cells` | `dilution_factor` | prose "Dilution factor is N" |

`find_kv_fact` matches three syntaxes: `key=value`, `key: value`, and
prose `key (is\|of) value`. Underscores in the key match either
underscore or whitespace in the description, so `dilution_factor`
catches "Dilution factor is 2" (ch360, ch409) without a separate
clause.

Disclosed > inferred. The brief author chose to surface the value;
honour it over feature heuristics.

## Sprint #44 hardening (2026-04-28)

The canonical-pipeline shape is now **contract-tested corpus-wide**:

```python
suggestion = auto_recipe(image, core=core, brief=challenge)
fn = suggestion.import_callable()
result = fn(core, **suggestion.default_kwargs)   # ← kwargs are signature-safe
```

Two regression tests lock both directions of the contract:

- `tests/test_auto_recipe.py::test_auto_recipe_kwargs_match_routed_callable_signature` —
  for every archived brief, `default_kwargs ⊆ inspect.signature(fn).parameters`.
  Catches the failure mode where the routing layer emits a kwarg the
  recipe forgot to accept (the pre-#44 bug: `primary_position_prior`
  emitted from `kwargs_from_brief` but `modality_switch_pipeline` only
  added it later).
- `tests/test_brief_parse.py::test_validate_against_brief_no_false_positives_on_10_of_10_corpus` —
  for every archived 10/10 submission, `validate_against_brief(answer,
  challenge)` reports zero errors. Locks the validator against
  regressions that would block known-good submissions.

When you add a new recipe to `_PRIMARY` or extend `kwargs_from_brief`,
re-run the suite. Both tests are <5s and corpus-wide.

## Sprint #44-followup: Submit / Tolerance shape parsers

Audit (2026-04-28) of 317 archived briefs surfaced two parser bugs:

1. **`extract_submit_shape` greedy nested-curly match.** Pre-fix the
   parser found the first `{` even if it was inside `list[{"t": int,
   ...}]` and lifted the *nested* keys as top-level. Caused 4 false
   positives on 10/10 trajectory-MPC submissions (ch624/628/629/630).
   Fix: dispatch on the first non-blank body line — `{` → curly
   parser, `-`/`*` → bullet parser, neither → empty (no guess).
   This also unlocked ~60 bullet-form briefs previously returning
   empty (validation surface 6 → 65 = 10×).
2. **`extract_tolerances` rejected `- key:value` (no space after key).**
   Plus had no inline-comma form (`Tolerance: a ±x, b ±y`). Combined
   with a tolerance-marker filter to reject prose continuations
   (ch359 ``- Approach: threshold the lawn …``), this lifted the
   tolerance-extraction surface from 47 → 62 briefs.

The mental model: **archived briefs are the test corpus**. When a
parser change moves the surface, run both regression tests and the
audit one-liners in this note. Archive briefs are append-only and
versioned in `../logs/challenges/` — they're a stable source of
ground truth for the brief-parsing layer.

## Sprint #49: dynamics_class classifier + safe_to_dry_run gate

`brief_parse.parse_brief` now populates a `dynamics_class` field that
classifies the sample-time-vs-wall-clock contract:

| Class | Meaning | Dry-run safe on same proxy? |
|---|---|---|
| `finite_budget_dynamic` | snap budget gates dynamic state (drift, dose, saturating counter) | **No** — dry-run consumes the budget |
| `wall_clock_locked` | realtime engine; biology runs between snaps | No — pre-MDA probes shift the time window |
| `snap_locked` | 1 snap = 1 sim-step (continuous=False + auto_step=True) | OK on a fresh proxy or after re-serve |
| `static` | no time evolution | Yes |
| `unknown` | keywords not detected | Default safe |

Use `safe_to_dry_run(brief) -> (bool, reason)` to gate the dry-run /
live-submit decision. Returns False for finite_budget + wall_clock
with a reason string.

The classifier is keyword-driven on the raw description; priority is
`finite_budget_dynamic > static > snap_locked > wall_clock_locked` so
that the strongest action consequence wins. See
`knowledge/Core/Concepts/Sample time vs wall-clock time.md` for the
mental-model framing and `feedback_dry_run_state_bleed.md` for the
ch664 r1 lesson that motivated this sprint.

Test corpus: 16 unit tests + a ch664/665 archive-brief sanity check
(both classify as `finite_budget_dynamic` from the brief's
"24-snap budget" + "drift" markers).
