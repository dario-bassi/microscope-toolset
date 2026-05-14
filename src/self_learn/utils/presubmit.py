"""Presubmit guard orchestrator.

Composes the 2-stage guard pattern from
``knowledge/Core/Approach/Pre-submission checklist.md`` § 2 into a
single entry-point:

  2a. ``self_learn.utils.preflight.run_preflight``          (cheapest)
  2b. ``self_learn.utils.render_vs_submit.render_vs_submit_check``

Stages run in order with early exit on ``block``: a preflight error
stops the chain before the more expensive render_vs_submit check runs.

Aggregate severity = max rank across stages:
  ``error > block > flag > ok``

Stage 2b is gracefully skipped when any of ``image``,
``submitted_value``, or ``recipe_re_detect_fn`` is not provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .preflight import run_preflight
from .render_vs_submit import render_vs_submit_check


_UNSET = object()

_SEVERITY_RANK = {"ok": 0, "flag": 1, "block": 2, "error": 3}


@dataclass(frozen=True)
class PresubmitResult:
    """Aggregate outcome of the presubmit guard pipeline."""

    severity: str
    rationale: str
    stage_results: dict = field(default_factory=dict)
    stages_run: tuple = ()


def _aggregate(stage_severities: dict) -> str:
    if not stage_severities:
        return "ok"
    return max(stage_severities.values(), key=lambda s: _SEVERITY_RANK.get(s, 0))


def _preflight_severity(preflight_dict: dict, treat_warnings_as_flag: bool) -> str:
    if preflight_dict.get("errors"):
        return "block"
    if preflight_dict.get("warnings") and treat_warnings_as_flag:
        return "flag"
    return "ok"


def run_presubmit_guards(
    *,
    # Stage 2a — preflight
    answer: dict,
    pixel_size_um: Optional[float] = None,
    rules: Optional[dict] = None,
    required_keys: Optional[list] = None,
    sample_type: Optional[str] = None,
    n_visible_objects: Optional[int] = None,
    # Stage 2b — render_vs_submit
    image: Optional[np.ndarray] = None,
    submitted_value: Any = _UNSET,
    recipe_re_detect_fn: Optional[Callable] = None,
    rel_tol_flag: float = 0.05,
    rel_tol_block: float = 0.20,
    abs_tol: Optional[float] = None,
    position_match_tol_frac: float = 0.02,
    recall_floor: float = 0.8,
    # Behaviour
    treat_warnings_as_flag: bool = True,
) -> PresubmitResult:
    """Run preflight then render_vs_submit with early exit on block.

    Returns a ``PresubmitResult`` with the aggregate severity and per-stage
    results. Typical usage::

        result = run_presubmit_guards(
            answer=answer,
            pixel_size_um=ps,
            image=final_snap,
            submitted_value=answer["count"],
            recipe_re_detect_fn=lambda img: len(detect_cells(img, **kw)),
        )
        if result.severity == "block":
            raise SystemExit(result.rationale)
        if result.severity == "flag":
            method_description += f"  [{result.rationale}]"
    """
    stage_results: dict = {}
    stage_severities: dict = {}
    stages_run: list = []
    rationales: list = []

    # ---- Stage 2a — preflight ----
    stages_run.append("preflight")
    try:
        pre = run_preflight(
            answer,
            pixel_size_um=pixel_size_um,
            rules=rules,
            required_keys=required_keys,
            sample_type=sample_type,
            n_visible_objects=n_visible_objects,
        )
    except Exception as exc:
        return PresubmitResult(
            severity="error",
            rationale=f"preflight raised {type(exc).__name__}: {exc}",
            stage_results={"preflight": {"exception": str(exc)}},
            stages_run=tuple(stages_run),
        )

    stage_results["preflight"] = pre
    sev_a = _preflight_severity(pre, treat_warnings_as_flag)
    stage_severities["preflight"] = sev_a

    if sev_a == "block":
        first_err = pre["errors"][0] if pre.get("errors") else "?"
        return PresubmitResult(
            severity="block",
            rationale=f"preflight: {len(pre['errors'])} error(s); first: {first_err}",
            stage_results=stage_results,
            stages_run=tuple(stages_run),
        )
    if sev_a == "flag":
        rationales.append(f"preflight: {len(pre.get('warnings', []))} warning(s)")

    # ---- Stage 2b — render_vs_submit (graceful skip) ----
    can_run_2b = (
        image is not None
        and submitted_value is not _UNSET
        and recipe_re_detect_fn is not None
    )
    if can_run_2b:
        stages_run.append("render_vs_submit")
        check = render_vs_submit_check(
            image=image,
            submitted_value=submitted_value,
            recipe_re_detect_fn=recipe_re_detect_fn,
            rel_tol_flag=rel_tol_flag,
            rel_tol_block=rel_tol_block,
            abs_tol=abs_tol,
            position_match_tol_frac=position_match_tol_frac,
            recall_floor=recall_floor,
        )
        stage_results["render_vs_submit"] = check
        stage_severities["render_vs_submit"] = check.severity
        if check.severity in ("flag", "block", "error"):
            rationales.append(f"render_vs_submit: {check.rationale}")
        if check.severity in ("block", "error"):
            return PresubmitResult(
                severity=check.severity,
                rationale="; ".join(rationales),
                stage_results=stage_results,
                stages_run=tuple(stages_run),
            )

    final_severity = _aggregate(stage_severities)
    if not rationales and final_severity == "ok":
        rationales.append("all stages passed")
    return PresubmitResult(
        severity=final_severity,
        rationale="; ".join(rationales),
        stage_results=stage_results,
        stages_run=tuple(stages_run),
    )
