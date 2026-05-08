"""Presubmit guard orchestrator.

Composes the 3-tier guard pattern from
``knowledge/core/approach/Pre-submission checklist.md`` § 2 into a
single entry-point so recipes don't have to glue the three utilities
together by hand:

  2a. ``src.core.utils.preflight.run_preflight``         (cheapest)
  2b. ``src.core.utils.render_vs_submit.render_vs_submit_check``
  2c. ``agent/skills/pre-submit-review`` skill (LLM, opt-in via
       ``run_review=True``; this orchestrator only *prepares* its
       inputs — the caller dispatches the skill afterwards).

Stages run in fixed order with **early exit on `block`**: a preflight
error stops the chain (we don't burn an LLM call after a hard fail).

Aggregate severity = max-rank across stages, where
``error > block > flag > ok``.

What this module does NOT do:

- No LLM dispatch. Stage 2c only writes the skill's input directory.
- No overlay rendering / matplotlib (the subagent handles that).
- No ``submit_solution`` / ``submit_with_showcase`` call.
- No re-implementation of preflight or render_vs_submit logic.
- No new severity levels beyond ok / flag / block / error.

Sprint #22 (counter=57).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .preflight import run_preflight
from .render_vs_submit import render_vs_submit_check


_UNSET = object()
_TMP_ROOT = Path("/tmp")  # patched in tests


_SEVERITY_RANK = {"ok": 0, "flag": 1, "block": 2, "error": 3}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PresubmitResult:
    """Aggregate outcome of the presubmit guard pipeline."""

    severity: str
    rationale: str
    stage_results: dict = field(default_factory=dict)
    review_prep_path: Optional[Path] = None
    stages_run: tuple = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_jsonable(o):
    if isinstance(o, (np.integer, np.bool_)):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


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


def _prep_review_dir(
    *,
    challenge_id: int,
    answer: dict,
    solve_script_path: Optional[Path],
    challenge_json_path: Optional[Path],
    source_images: Optional[dict],
    intermediate_arrays: Optional[dict],
) -> Path:
    """Build /tmp/ch<N>_pre_submit/ inputs the LLM skill expects."""
    tmp_dir = _TMP_ROOT / f"ch{int(challenge_id)}_pre_submit"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with (tmp_dir / "answer.json").open("w") as fp:
        json.dump(answer, fp, indent=2, default=_to_jsonable)

    if solve_script_path is not None:
        shutil.copyfile(str(solve_script_path), tmp_dir / "solve.py")
    if challenge_json_path is not None:
        shutil.copyfile(str(challenge_json_path), tmp_dir / "challenge.json")

    if source_images:
        src_dir = tmp_dir / "source"
        src_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in source_images.items():
            np.save(src_dir / f"{name}.npy", np.asarray(arr))

    if intermediate_arrays:
        int_dir = tmp_dir / "intermediate"
        int_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in intermediate_arrays.items():
            np.save(int_dir / f"{name}.npy", np.asarray(arr))

    return tmp_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_presubmit_guards(
    *,
    # Stage 2a — preflight
    answer: dict,
    pixel_size_um: Optional[float] = None,
    challenge_notes: Optional[str] = None,
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
    # Stage 2c — review prep
    run_review: bool = False,
    challenge_id: Optional[int] = None,
    solve_script_path: Optional[Path] = None,
    challenge_json_path: Optional[Path] = None,
    source_images: Optional[dict] = None,
    intermediate_arrays: Optional[dict] = None,
    # Behaviour
    treat_warnings_as_flag: bool = True,
) -> PresubmitResult:
    """Run preflight → render_vs_submit (→ review prep) with early exit on block.

    See module docstring for stage order, severity ladder, and what
    each kwarg controls. Stage 2b is gracefully skipped when any of
    ``image`` / ``submitted_value`` / ``recipe_re_detect_fn`` is missing.
    Stage 2c only runs when ``run_review=True`` and no earlier blocker.
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
            challenge_notes=challenge_notes,
            rules=rules,
            required_keys=required_keys,
            sample_type=sample_type,
            n_visible_objects=n_visible_objects,
        )
    except Exception as e:  # noqa: BLE001
        return PresubmitResult(
            severity="error",
            rationale=f"preflight raised {type(e).__name__}: {e}",
            stage_results={"preflight": {"exception": str(e)}},
            stages_run=tuple(stages_run),
        )

    stage_results["preflight"] = pre
    sev_a = _preflight_severity(pre, treat_warnings_as_flag)
    stage_severities["preflight"] = sev_a
    if sev_a == "block":
        rationales.append(
            f"preflight: {len(pre['errors'])} error(s); first: "
            f"{pre['errors'][0] if pre.get('errors') else '?'}"
        )
        return PresubmitResult(
            severity="block",
            rationale="; ".join(rationales),
            stage_results=stage_results,
            stages_run=tuple(stages_run),
        )
    if sev_a == "flag":
        rationales.append(
            f"preflight: {len(pre.get('warnings', []))} warning(s)"
        )

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

    # ---- Stage 2c — review prep ----
    review_path: Optional[Path] = None
    if run_review:
        if challenge_id is None:
            raise ValueError(
                "run_review=True requires challenge_id (used to build "
                "/tmp/ch<N>_pre_submit/)"
            )
        try:
            review_path = _prep_review_dir(
                challenge_id=int(challenge_id),
                answer=answer,
                solve_script_path=solve_script_path,
                challenge_json_path=challenge_json_path,
                source_images=source_images,
                intermediate_arrays=intermediate_arrays,
            )
            stages_run.append("review_prep")
            if solve_script_path is None or challenge_json_path is None:
                rationales.append(
                    "review_prep: solve.py and/or challenge.json not "
                    "supplied — skill's code-review pass will be partial"
                )
        except Exception as e:  # noqa: BLE001
            return PresubmitResult(
                severity="error",
                rationale=(
                    f"review_prep raised {type(e).__name__}: {e}"
                ),
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
        review_prep_path=review_path,
        stages_run=tuple(stages_run),
    )
