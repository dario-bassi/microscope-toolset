# Sample-class auto-detection

The first thing to do on an unknown challenge is **identify what kind of sample is in front of you**. If you don't, you spend the snap budget tuning a recipe that was never going to work — phase-contrast tessellations get treated as plate-reader grids because both look "structured", FRAP samples get treated as sparse-cell counts because there happens to be one bright spot, etc. This note describes the strategy and points at where it sits on the literature axis.

## The strategy

Two parts, deliberately separated:

1. **Classifier.** Cheap features → discrete sample class. Runs in <200 ms on a 512×512 frame. Implemented in `src/core/utils/sample_classifier.py` (sprint #17). The current implementation extracts 8 hand-crafted features (entropy, FFT periodicity, edge density, CC count + size, sparsity) and uses ranked decision rules over 7 classes (`plate_reader_grid`, `voronoi_monolayer`, `phase_contrast_tessellation`, `sparse_cells`, `single_bright_spot`, `wide_dynamic_range_field`, `unknown`). Returns a confidence-ranked list — *not* a single label, because borderline samples should keep optionality.
2. **Dispatcher.** Class → recipe pick + parameter prefill + hardware-gated fallbacks. Implemented in `src/core/utils/auto_recipe.py` (sprint #18). Optional core probe (channel list, magnification, SLM device, objective state labels) refines the pick — see [[Core/Strategies/Auto recipe selection]] for the full mapping table.

The dispatcher is the *image-level* layer of first-contact. Scenario-level signals (dose budgets, per-cell state) used to come from a sibling `bridge_preflight` probe, but the bridge RPC surface was closed 2026-04-27 — those signals must now be re-derived from the brief text + camera frames. See [[Core/Approach/Transferability contract]].

The split is deliberate: classifier is a sample property; dispatcher is a recipe-catalogue property. Adding a new recipe touches the dispatcher only. Tightening a class boundary touches the classifier only. Don't merge them.

## When to call

Right after the first `core.snapImage()`, before any workflow decision:

```python
core.snapImage()
preview = core.getImage()
suggestion = auto_recipe(preview, core=core)
```

`suggestion.classifier_class` and `suggestion.classifier_confidence` are part of the public output — *log them* in the solve script. When a recipe gets a 4/10, the post-mortem is much faster if you can see whether the classifier was wrong, whether the dispatch was wrong, or whether the recipe itself misfired on a correctly-classified sample.

## Anti-patterns

- **Solving by pattern-match against the last challenge.** No-template-solving (NN rule 7) exists because two challenges with similar titles can be totally different sample classes. Run the classifier even when you "know what it is".
- **Trusting the top-1 above ~0.7 confidence and ignoring fallbacks.** The 7-class taxonomy doesn't carve every real sample cleanly. When confidence is in 0.4–0.7, walk the fallbacks before committing.
- **Hand-tuning classifier features for one challenge.** The features are deliberately cheap and generic. If a challenge needs a feature the current classifier doesn't have, consider whether it actually needs a *new recipe* (handled at the dispatch layer) rather than a new feature.

## Literature axis

This is the entry-level instantiation of an axis that the field is moving along:

```
hand-crafted features  →  per-task trained detector  →  foundation-model encoder
   (sprint #17 here)        (Conrad 2011, Shi 2024)        (Yu 2024 PLIP)
```

[[Papers/Yu 2024]] is the canonical end-state: PLIP (a pathology-specific vision-language foundation model, ~200k image-text pairs) replaces hand-crafted features with a pretrained semantic encoder, and a small task head turns the embedding into "zoom here / skip / continue" + "use this exposure / objective". The closed loop is the same shape as `sample_classifier → auto_recipe → recipe`; the difference is **what produces the score**.

[[Papers/Morgado 2024]] frames this as the "task-driven microscopy" axis: the score driving acquisition is conditioned on the research question, and a re-targetable encoder (rather than a hand-crafted phenotype classifier or a YOLO detector trained per-task) makes that conditioning portable. Yu 2024 demonstrates the foundation-model end. Our `sample_classifier` is the floor: cheap, transparent, no GPU. The progression isn't "throw away the cheap classifier", it's "swap features for embeddings when (a) a domain-specific foundation model exists for the modality and (b) per-class fine-tuning data is available".

The cheap classifier should remain even after a foundation-model swap — as a sanity check on the encoder, and as a fast-path when the foundation model is unavailable (offline, low-power, latency-bounded).

## Sketch of the foundation-model swap

Drop-in replacement preserves the `auto_recipe` interface:

```python
def auto_recipe_fm(image, core=None, *, encoder, task_head):
    feats = encoder(image)                        # PLIP / Cell-painting CLIP / ...
    decision = task_head(feats, recipes=...)      # MLP head returning class + confidence
    return _build_suggestion(...)                 # same RecipeSuggestion shape
```

The interface choice (image in, RecipeSuggestion out) is exactly right for the swap: the caller doesn't know or care whether the score came from a 5-line histogram heuristic or a 1B-parameter encoder. That's the protocol the `auto_recipe` sprint locked in.

## See also

- [[Core/Strategies/Auto recipe selection]] — the dispatch half of the pair.
- [[Core/Strategies/Adaptive acquisition]] — the survey→zoom shape this strategy feeds into.
- [[Core/Strategies/Multi-scale morphometry]] — cousin pattern: low-mag survey + high-mag follow-up; the classifier choice flips between scales.
- [[Papers/Yu 2024]] — the foundation-model end of the axis.
- [[Papers/Morgado 2024]] — taxonomy that places sample-class auto-detection on the data-driven axis.
- [[Papers/Shi 2024]] — the per-task YOLO scout pattern that PLIP supersedes.
- [[Papers/Conrad 2011]] — the hand-crafted-classifier pattern our sprint #17 instantiates.
