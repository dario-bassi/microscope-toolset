"""Auto-recipe selector — image (+ optional core) → recipe suggestion.

Bridges :mod:`self_learn.utils.sample_classifier` to the recipes catalogue
under ``src/recipes/``. Pure dispatch primitive: never snaps, never moves
hardware, never invokes a recipe. Returns a :class:`RecipeSuggestion`
the caller can act on.

Typical use, immediately after the first preview snap::

    core.snapImage()
    img = core.getImage()
    suggestion = auto_recipe(img, core=core)
    if suggestion.recipe_module:
        fn = suggestion.import_callable()
        result = fn(core, **suggestion.default_kwargs)

The "core never imports recipes" rule still holds — we reference recipes
by *string* paths (``"src.recipes.<module>"``) and the caller does the
final ``importlib`` dance via :meth:`RecipeSuggestion.import_callable`.

Sprint #18 (2026-04-26).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .sample_classifier import classify_sample, extract_features


# ---------------------------------------------------------------------------
# RecipeSuggestion dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecipeSuggestion:
    """A single recipe pick with rationale + pre-filled kwargs."""

    recipe_module: Optional[str]
    callable_name: Optional[str]
    default_kwargs: dict = field(default_factory=dict)
    rationale: str = ""
    classifier_class: str = "unknown"
    classifier_confidence: float = 0.0
    fallbacks: tuple = field(default_factory=tuple)
    requires_timelapse: bool = False
    requires_slm: bool = False
    requires_modality_device: bool = False
    requires_objective_states: bool = False
    requires_dm_device: bool = False
    features: dict = field(default_factory=dict)

    def import_callable(self):
        """Resolve ``recipe_module``/``callable_name`` to a real function.

        Raises ``ValueError`` if either field is empty (i.e. ``unknown``
        suggestions). Caller invokes the returned function.
        """
        if self.recipe_module is None or self.callable_name is None:
            raise ValueError(
                "RecipeSuggestion has no concrete recipe — handle the "
                "unknown / no-primary case before calling import_callable()."
            )
        mod = importlib.import_module(self.recipe_module)
        return getattr(mod, self.callable_name)


# ---------------------------------------------------------------------------
# Static class → recipe mapping
# ---------------------------------------------------------------------------

# Each entry: (recipe_module, callable_name, rationale_template, flags).
# Flags is a dict of {field: bool} pointing at RecipeSuggestion booleans.
_PRIMARY = {
    "plate_reader_grid": (
        "src.recipes.plate_reader_ic50",
        "fit_compound_pair",
        "8x12 well-block periodicity → plate-reader IC50 fit.",
        {},
    ),
    "voronoi_monolayer": (
        "src.recipes.two_population_stain",
        "classify_pipeline",
        "tessellated bright-nuclei monolayer → bimodal-stain classifier.",
        {},
    ),
    "phase_contrast_tessellation": (
        "src.recipes.two_population_stain",
        "classify_pipeline",
        "phase-contrast tessellation → cell-detection + bimodal classify.",
        {},
    ),
    "sparse_cells": (
        "src.recipes.hemocytometer",
        "hemocytometer_count",
        "sparse, isolated cells on dark bg → hemocytometer-style count.",
        {},
    ),
    "single_bright_spot": (
        "src.recipes.frap_background_correction",
        "analyze_frap",
        "single bright FOI on dark field → FRAP recovery curve.",
        {"requires_timelapse": True},
    ),
    "wide_dynamic_range_field": (
        "src.recipes.event_driven_modality_switch",
        "modality_switch_pipeline",
        "high-entropy speckle field → wave-source pacemaker scout.",
        {"requires_timelapse": True, "requires_objective_states": True},
    ),
    "lightsheet_align": (
        "src.recipes.lightsheet_align",
        "align_descent",
        "Archetype=lightsheet_align → AutoPilot 3-axis coordinate descent.",
        {},
    ),
    "lightsheet_drift": (
        "src.recipes.lightsheet_align",
        "drift_track_capture",
        "Archetype=lightsheet_drift → McDole 2018 periodic re-alignment under hidden drift.",
        {},
    ),
    "cytokinesis_kinetics": (
        "src.recipes.cytokinesis_kinetics",
        "two_phase_kinetics",
        "Archetype=cytokinesis → myosin-band PCA + 2-phase rate fit.",
        {"requires_timelapse": True},
    ),
}

# Hardware-gated recipes that only appear in fallbacks when a core probe
# confirms the device is present.
_HARDWARE_GATED = {
    "bacteria_trap": (
        "src.recipes.bacteria_trap",
        "run_bacteria_trap_mda",
        "SLM device present → phototaxis-trap fallback.",
        {"requires_slm": True, "requires_timelapse": True},
    ),
    "cybergenetic_per_cell_control": (
        "src.recipes.cybergenetic_per_cell_control",
        "run_per_cell_feedback",
        "SLM + monolayer → per-cell integral feedback fallback.",
        {"requires_slm": True, "requires_timelapse": True},
    ),
    "adaptive_sted_burst": (
        "src.recipes.adaptive_sted_burst",
        "scout_then_burst_pipeline",
        "Modality state device present → scout→burst→restore fallback.",
        {"requires_modality_device": True, "requires_timelapse": True},
    ),
    "blood_smear_wbc": (
        "src.recipes.blood_smear_wbc",
        "detect_and_classify_wbc_v2",
        "RGB Wright-Giemsa channel → WBC differential fallback.",
        {},
    ),
    "fibroblast_focal_adhesions": (
        "src.recipes.fibroblast_focal_adhesions",
        "count_focal_adhesions",
        "phalloidin / F-actin channel → focal-adhesion count fallback.",
        {},
    ),
    "neuron_puncta": (
        "src.recipes.neuron_puncta",
        "count_puncta_per_neuron",
        "synaptic-marker channel → soma-core + puncta count fallback.",
        {},
    ),
    "sensorless_ao": (
        "src.recipes.sensorless_ao",
        "solve_dm_sweep",
        "DeformableMirror state device present → quantised-action sensorless AO fallback.",
        {"requires_dm_device": True},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_suggestion(
    *,
    key: str,
    classifier_class: str,
    classifier_confidence: float,
    features: dict,
    default_kwargs: Optional[dict] = None,
    rationale_override: Optional[str] = None,
    fallbacks: tuple = (),
) -> RecipeSuggestion:
    """Materialise a RecipeSuggestion from an entry in the mapping tables."""
    entry = _PRIMARY.get(key) or _HARDWARE_GATED.get(key)
    if entry is None:
        raise KeyError(f"unknown recipe key: {key}")
    module, callable_name, rationale, flags = entry
    return RecipeSuggestion(
        recipe_module=module,
        callable_name=callable_name,
        default_kwargs=dict(default_kwargs or {}),
        rationale=rationale_override or rationale,
        classifier_class=classifier_class,
        classifier_confidence=classifier_confidence,
        fallbacks=fallbacks,
        requires_timelapse=flags.get("requires_timelapse", False),
        requires_slm=flags.get("requires_slm", False),
        requires_modality_device=flags.get("requires_modality_device", False),
        requires_objective_states=flags.get("requires_objective_states", False),
        requires_dm_device=flags.get("requires_dm_device", False),
        features=dict(features),
    )


def _kwargs_for(class_key: str, features: dict, cfg=None, brief=None) -> dict:
    """Build a recipe's ``default_kwargs``: feature-derived defaults
    overlaid with anything the brief disclosed explicitly.

    Disclosed > inferred — the brief author chose to surface a value,
    so it overrides the feature heuristic on the same key.
    """
    kw = _featurise_kwargs(class_key, features, cfg)
    if brief is not None:
        from .brief_parse import kwargs_from_brief
        kw.update(kwargs_from_brief(brief, class_key))
    return kw


def _featurise_kwargs(class_key: str, features: dict, cfg=None) -> dict:
    """Pre-fill recipe kwargs from classifier features (and optional cfg)."""
    kw: dict = {}
    median_area = float(features.get("median_area", 0.0) or 0.0)
    n_components = int(features.get("n_components", 0) or 0)

    if class_key in ("voronoi_monolayer", "phase_contrast_tessellation"):
        if median_area > 0:
            kw["min_area_px"] = max(20, int(0.5 * median_area))
        if n_components > 0:
            kw["n_cells_hint"] = n_components

    elif class_key == "sparse_cells":
        if median_area > 0:
            kw["min_area"] = max(3, int(0.3 * median_area))
            kw["max_area"] = max(10, int(5.0 * median_area))

    elif class_key == "single_bright_spot":
        if median_area > 0:
            roi = int(round(2.0 * float(np.sqrt(median_area))))
            kw["frap_roi_size"] = max(80, min(200, roi))

    # If a cfg is available, mirror px-area kwargs into µm² where present.
    if cfg is not None and getattr(cfg, "pixel_size_um", None):
        px = float(cfg.pixel_size_um)
        if "min_area_px" in kw:
            kw["min_area_um2"] = round(kw["min_area_px"] * px * px, 4)

    return kw


def _channel_hardware_keys(cfg) -> tuple:
    """Return hardware-gated recipe keys unlocked by core/channel probe."""
    keys: list[str] = []
    chans = tuple(c.lower() for c in (cfg.available_channels or ()) if isinstance(c, str))

    def _any_match(patterns):
        return any(p in c for c in chans for p in patterns)

    if _any_match(("phall", "actin", "f-actin")):
        keys.append("fibroblast_focal_adhesions")
    if _any_match(("map2", "synapt", "psd95", "vglut")):
        keys.append("neuron_puncta")
    if _any_match(("giemsa", "wright")):
        keys.append("blood_smear_wbc")

    if cfg.slm_device is not None:
        keys.extend(("bacteria_trap", "cybergenetic_per_cell_control"))

    if getattr(cfg, "dm_device", None) is not None:
        keys.append("sensorless_ao")

    # Modality device: scan objective_labels for known modality names.
    obj_labels_lower = " ".join(
        lbl.lower() for _, lbl in (cfg.objective_labels or ())
    )
    if any(tok in obj_labels_lower
           for tok in ("sted", "lattice", "confocal", "widefield_scout")):
        keys.append("adaptive_sted_burst")

    return tuple(dict.fromkeys(keys))  # dedupe preserving order


def _ranked_other_classes(class_scores: list, top_class: str) -> list:
    """Return non-top, non-unknown classes ranked by score (high → low)."""
    return [(name, score) for name, score in class_scores
            if name != top_class and name != "unknown" and score > 0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Archetype keywords from the brief → recipe-class hints. The brief
# author's archetype label is a strong dispatch signal that the image
# classifier alone may miss (e.g. modality_switch on cardio looks like
# wide_dynamic_range_field but the brief explicitly says so).
_ARCHETYPE_TO_CLASS = {
    "modality_switch": "wide_dynamic_range_field",
    "frap": "single_bright_spot",
    "voronoi": "voronoi_monolayer",
    "hemocytometer": "sparse_cells",
    "plate_reader": "plate_reader_grid",
    "lightsheet_align": "lightsheet_align",
    "lightsheet_drift": "lightsheet_drift",
    "cytokinesis": "cytokinesis_kinetics",
    "cytokinesis_kinetics": "cytokinesis_kinetics",
}


def auto_recipe(
    image: np.ndarray,
    core=None,
    *,
    channel: Optional[int] = None,
    brief_archetype: Optional[str] = None,
    brief=None,
) -> RecipeSuggestion:
    """Pick a recipe from a fresh first-contact image (+ optional core).

    Args:
        image: 2D or RGB image — typically the result of an initial
            ``core.snapImage(); core.getImage()``.
        core: Optional pymmcore-plus / -proxy core. When provided, the
            channel list, pixel size, magnification, SLM and objective
            states refine the suggestion (see module docstring for the
            full table). All probes are read-only.
        channel: Optional channel index passed through to
            :func:`extract_features` for multi-channel inputs.
        brief_archetype: Short-form archetype override — pass the
            archetype string directly (e.g. ``"frap"``). Takes
            precedence over ``brief["archetype"]`` when both are set.
        brief: Optional dict with an ``"archetype"`` key that overrides
            the image-feature classifier. Example:
            ``brief={"archetype": "frap"}``. When supplied, the
            archetype drives recipe selection above image-feature scores.

    Returns:
        :class:`RecipeSuggestion`. When the classifier returns
        ``unknown``, ``recipe_module`` is ``None`` and the top-3 weak
        candidates are placed in ``fallbacks``.
    """
    feats = extract_features(image, channel=channel)
    ranked = classify_sample(image, channel=channel)

    # Archetype override: explicit brief_archetype wins, then brief["archetype"].
    archetype_str = brief_archetype
    if archetype_str is None and isinstance(brief, dict):
        archetype_str = brief.get("archetype")

    if archetype_str:
        key = archetype_str.lower().split()[0]
        mapped_class = _ARCHETYPE_TO_CLASS.get(key)
        if mapped_class and mapped_class in _PRIMARY:
            # Prepend at conf=1.0 (above any image-feature score).
            ranked = [(mapped_class, 1.0)] + [r for r in ranked if r[0] != mapped_class]

    if not ranked:
        return RecipeSuggestion(
            recipe_module=None,
            callable_name=None,
            rationale="classifier returned empty ranking",
            classifier_class="unknown",
            classifier_confidence=0.0,
            features=feats,
        )

    top_class, top_conf = ranked[0]

    # Optional core probe.
    cfg = None
    cur_mag = None
    hardware_keys: tuple = ()
    if core is not None:
        from ..hardware.config import get_config
        cfg = get_config(core)
        try:
            cur_mag = cfg.current_magnification(core)
        except Exception:
            cur_mag = None
        hardware_keys = _channel_hardware_keys(cfg)

    # Unknown → no primary, top-3 weak candidates as fallbacks.
    if top_class == "unknown":
        weak_others = _ranked_other_classes(ranked, top_class)
        fb = []
        for name, score in weak_others[:3]:
            if name in _PRIMARY:
                fb.append(_build_suggestion(
                    key=name,
                    classifier_class=name,
                    classifier_confidence=score,
                    features=feats,
                    default_kwargs=_kwargs_for(name, feats, cfg, structured),
                    rationale_override=(
                        f"weak signal (conf={score:.2f}) — fallback because "
                        "classifier could not resolve unambiguously."
                    ),
                ))
        return RecipeSuggestion(
            recipe_module=None,
            callable_name=None,
            rationale=("classifier returned unknown — manually pick a recipe "
                       "or run a scout sweep"),
            classifier_class="unknown",
            classifier_confidence=top_conf,
            fallbacks=tuple(fb),
            features=feats,
        )

    # Channel-gated channel-substring overrides may swap the primary.
    chosen_class = top_class
    if cfg is not None:
        # Phase-contrast + phalloidin / F-actin channel → fibroblast.
        if (top_class == "phase_contrast_tessellation"
                and "fibroblast_focal_adhesions" in hardware_keys):
            return _build_suggestion(
                key="fibroblast_focal_adhesions",
                classifier_class=top_class,
                classifier_confidence=top_conf,
                features=feats,
                default_kwargs=_kwargs_for(top_class, feats, cfg, structured),
                rationale_override=(
                    "phase-contrast tessellation + phalloidin/F-actin "
                    "channel detected → fibroblast focal adhesions."
                ),
                fallbacks=_default_fallbacks(
                    top_class, ranked, feats, cfg, hardware_keys, structured),
            )

    # Build primary suggestion from the table.
    if chosen_class in _PRIMARY:
        suggestion = _build_suggestion(
            key=chosen_class,
            classifier_class=top_class,
            classifier_confidence=top_conf,
            features=feats,
            default_kwargs=_kwargs_for(chosen_class, feats, cfg, structured),
            fallbacks=_default_fallbacks(
                chosen_class, ranked, feats, cfg, hardware_keys, structured),
        )

        # Magnification refinement: low mag + wide_dynamic → keep primary;
        # high mag + wide_dynamic → demote.
        if (chosen_class == "wide_dynamic_range_field"
                and cur_mag is not None and cur_mag >= 40):
            # Demote to fallback, promote two_population_stain.
            return _build_suggestion(
                key="voronoi_monolayer",
                classifier_class=top_class,
                classifier_confidence=top_conf,
                features=feats,
                default_kwargs=_kwargs_for("voronoi_monolayer", feats, cfg, structured),
                rationale_override=(
                    f"high magnification ({cur_mag}x) demotes wide-dynamic "
                    "scout → bimodal-stain classifier."
                ),
                fallbacks=(suggestion,) + _default_fallbacks(
                    "voronoi_monolayer", ranked, feats, cfg, hardware_keys, structured),
            )

        return suggestion

    # Defensive fallthrough.
    return RecipeSuggestion(
        recipe_module=None,
        callable_name=None,
        rationale=f"no recipe mapped for class={top_class}",
        classifier_class=top_class,
        classifier_confidence=top_conf,
        features=feats,
    )


def _default_fallbacks(
    top_key: str,
    ranked: list,
    features: dict,
    cfg,
    hardware_keys: tuple,
    brief=None,
) -> tuple:
    """Build a ranked fallback tuple for the given primary key."""
    fb: list[RecipeSuggestion] = []

    # Other ranked classes (after the top) that have a primary mapping.
    for name, score in ranked[1:]:
        if name == "unknown" or name == top_key:
            continue
        if name not in _PRIMARY:
            continue
        fb.append(_build_suggestion(
            key=name,
            classifier_class=name,
            classifier_confidence=score,
            features=features,
            default_kwargs=_kwargs_for(name, features, cfg, brief),
            rationale_override=(
                f"alternate class (conf={score:.2f}): "
                + _PRIMARY[name][2]
            ),
        ))

    # Hardware-gated fallbacks — only if cfg unlocked them.
    for hk in hardware_keys:
        if hk == top_key:
            continue
        fb.append(_build_suggestion(
            key=hk,
            classifier_class=top_key,
            classifier_confidence=0.0,
            features=features,
            default_kwargs={},
        ))

    return tuple(fb)
