# Auto recipe selection

When you connect to an unknown challenge, the first thing to do — even before deciding on the workflow — is **look at the sample**. The sample-classifier (`src/core/utils/sample_classifier.py`) extracts 8 cheap features (entropy, FFT periodicity, edge density, CC stats, sparsity) and ranks 7 sample classes. The `auto_recipe()` selector turns that ranking into a concrete recipe pick + pre-filled kwargs.

## When to call

Right after the first `core.snapImage()`. Don't run a workflow until you've called this — even if you think you know the sample. The classifier catches dispatch mistakes a tired agent makes (treating a phase-contrast tessellation as a plate-reader grid because both look "structured", etc.).

```python
from src.core.utils.auto_recipe import auto_recipe

core.snapImage()
preview = core.getImage()
suggestion = auto_recipe(preview, core=core)
print(f"{suggestion.recipe_module}.{suggestion.callable_name}  "
      f"(class={suggestion.classifier_class} "
      f"conf={suggestion.classifier_confidence:.2f})")
print(f"Rationale: {suggestion.rationale}")
```

The returned `RecipeSuggestion` is a frozen dataclass with: `recipe_module`, `callable_name`, `default_kwargs`, `rationale`, `classifier_class`, `classifier_confidence`, `fallbacks` (tuple of alt suggestions), boolean flags (`requires_timelapse`, `requires_slm`, `requires_modality_device`, `requires_objective_states`, `requires_dm_device`), and the raw `features` dict for debugging.

Scenario-level signals (dose budgets, per-cell state) used to come
from `bridge_preflight` / `first_contact`, but those primitives are
removed (bridge RPC surface closed 2026-04-27). All scenario context
must now be inferred from the brief text + camera frames + standard
device-property reads. See [[Core/Approach/Transferability contract]].

## Mapping table

| Classifier class | Primary recipe | Pre-filled kwargs |
|---|---|---|
| `plate_reader_grid` | `plate_reader_ic50.fit_compound_pair` | — |
| `voronoi_monolayer` | `two_population_stain.classify_pipeline` | `min_area_px`, `n_cells_hint` |
| `phase_contrast_tessellation` | `two_population_stain.classify_pipeline` | `min_area_px`, `n_cells_hint` |
| `sparse_cells` | `hemocytometer.hemocytometer_count` | `min_area`, `max_area`, `dilution_factor` |
| `single_bright_spot` | `frap_background_correction.analyze_frap` | `frap_roi_size`, `n_baseline` |
| `wide_dynamic_range_field` | `event_driven_modality_switch.modality_switch_pipeline` | `n_burst`, `primary_position_prior`, `viewport_centre_px` |
| `sim_protocol` | `sim_protocol.run_sim_protocol` | `mode`, `channel` |
| `unknown` | (no primary) | top-3 weak fallbacks |

`sim_protocol` is **not** image-classifier reachable — frequency
content isn't a sample-classifier feature. It's only routed via the
`brief=` archetype path (see below).

When `core` is supplied, the selector also reads `MicroscopeConfig` (channel list, pixel size, magnification, SLM device, objective state labels) to **refine** the pick:

- Phase-contrast tessellation + a phalloidin/F-actin channel → swap primary to `fibroblast_focal_adhesions`.
- High magnification (≥ 40×) + wide-dynamic class → demote `event_driven_modality_switch`, promote `two_population_stain` (event-driven needs the low-mag scout).
- SLM device present → unlock `bacteria_trap` and `cybergenetic_per_cell_control` as fallbacks.
- Objective labels containing `sted` / `confocal` / `lattice` → unlock `adaptive_sted_burst`.
- Wright/Giemsa channel → unlock `blood_smear_wbc` fallback.
- MAP2 / synaptic-marker channel → unlock `neuron_puncta` fallback.
- Any `min_area_px` kwarg is mirrored into `min_area_um2` using `pixel_size_um²` so the recipe can pick whichever unit it prefers (see [[Core/Strategies/Physical-unit thresholds]]).

## Brief-aware dispatch (`brief=` parameter)

Sprint #41 wired `parse_brief` → `auto_recipe` so disclosed values
in the brief drive both the archetype override AND the
`default_kwargs` pre-fill:

```python
from src.core.utils.brief_parse import parse_brief
from src.core.utils.auto_recipe import auto_recipe

brief = parse_brief(challenge)               # StructuredBrief / dict / str all OK
suggestion = auto_recipe(image, core=core, brief=brief)
```

What the brief drives:

- **Archetype override** (`_ARCHETYPE_TO_CLASS` map): `frap` →
  `single_bright_spot`, `voronoi` → `voronoi_monolayer`, `hemocytometer`
  → `sparse_cells`, `plate_reader` → `plate_reader_grid`,
  `modality_switch` → `wide_dynamic_range_field`, `sim` → `sim_protocol`.
  Beats the image classifier when the brief explicitly says so.
- **Recipe-specific kwargs** (via `kwargs_from_brief`): each class has
  its own extractor — disclosed coords (`primary_position_prior` for
  modality_switch), numeric facts (`n_burst`, `n_baseline`,
  `n_cells`, `dilution_factor`), and submit-shape mode picks
  (`mode=demod_3phase` vs `mode=orientation` for `sim_protocol`).
- **Image-coord-vs-PDE-grid disambiguation** (`_find_image_space_coord`
  matches the literal `image (X, Y)` pattern). The ch651 trap was
  reading `PRIMARY pacemaker at PDE grid (32, 32)` as an image
  prior; the disambiguation keeps PDE coords out.

Disclosed > inferred — when a brief surfaces a value, it overrides
the feature heuristic on the same key. The brief author chose to
surface it.

## Out of scope

`auto_recipe` is a dispatcher, not a workflow. It must not:

1. **Auto-tune sweep ranges.** Returns one starting point, not a hyperparameter grid.
2. **Run a recipe.** No `submit_with_showcase`, no `run_events`, no second snap.
3. **Dispatch hardware.** No `setProperty` / `setConfig` / `setState`. All probes are read-only (`getState`, `getProperty`, `available_channels`, `objective_labels`).
4. **Fall through silently to a fallback.** When the primary fails to import, it stays in the `recipe_module` slot — the caller decides whether to walk fallbacks.

## Worked examples

See `tests/test_auto_recipe.py` (26 tests). Each synthetic-image generator is paired with the expected primary recipe; the core-probe tests use a `SimpleNamespace` fake to demonstrate channel/SLM/modality refinement; the `brief_*` tests cover archetype override, kwarg pre-fill, dict / str / StructuredBrief acceptance, and the SIM submit-shape → mode dispatcher.

## See also

- [[Core/Strategies/Adaptive acquisition]] — what to do once a recipe is picked.
- [[Core/Strategies/Physical-unit thresholds]] — why `min_area_um2` mirrors `min_area_px` when a pixel size is known.
- `src/core/utils/sample_classifier.py` — upstream feature extractor.
- `src/recipes/README.md` — the canonical list of recipes the table maps to.
