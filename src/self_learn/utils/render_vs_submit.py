"""Re-derivation numerical guard.

Pre-commit check: re-run the workflow's own detector on the final
image, compare the re-derived value to the value about to be recorded,
and surface a structured ``RenderSubmitCheck`` so the caller can log,
abort and re-derive, or escalate.

Catches the pitfall described in
``knowledge/Core/Pitfalls/Sim-state vs rendered count asymmetry.md``:
a value derived from a source other than the image (config, upstream
state) may differ from what the actual pixels show.

What this does NOT do:

- No LLM, no subagent dispatch (deterministic numpy/scipy only).
- No overlay rendering / matplotlib import.
- No file I/O. Returns the dataclass and stops.
- No silent exceptions. Detector errors become severity="error".
- No re-import of the workflow module. Detector is passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


DEFAULT_REL_TOL_FLAG = 0.05
DEFAULT_REL_TOL_BLOCK = 0.20
DEFAULT_POS_MATCH_TOL_FRAC = 0.02
DEFAULT_RECALL_FLOOR = 0.8


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RenderSubmitCheck:
    """Outcome of a render-vs-submit numerical check."""

    re_derived_value: Any
    submitted_value: Any
    delta: float
    severity: str  # "ok" | "flag" | "block" | "error"
    rationale: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Delta computers
# ---------------------------------------------------------------------------

def _is_position_list(v) -> bool:
    """Detect (Nx2) coordinate list/array — N pairs of numbers."""
    if isinstance(v, np.ndarray):
        return v.ndim == 2 and v.shape[1] == 2
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        return False
    for item in v:
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                return False
            for x in item:
                if not isinstance(x, (int, float, np.integer, np.floating)):
                    return False
        elif isinstance(item, np.ndarray):
            if item.shape != (2,):
                return False
        else:
            return False
    return True


def _delta_scalar(a, b) -> float:
    """Relative error in [0, 1] (capped); 0 when both are 0."""
    af = float(a)
    bf = float(b)
    denom = max(abs(af), abs(bf), 1.0)
    return abs(af - bf) / denom


def _delta_list(a, b) -> tuple[float, dict]:
    """Element-wise relative error; max across elements."""
    a_list = list(a)
    b_list = list(b)
    details: dict = {"n_submitted": len(a_list), "n_re_derived": len(b_list)}
    if len(a_list) != len(b_list):
        details["per_element_delta"] = []
        return 1.0, details
    if len(a_list) == 0:
        details["per_element_delta"] = []
        return 0.0, details
    per = [_delta_scalar(x, y) for x, y in zip(a_list, b_list)]
    details["per_element_delta"] = per
    return float(max(per)), details


def _delta_positions(
    submitted, re_derived, image_shape, match_tol_px,
) -> tuple[float, dict]:
    """Hungarian-match positions; return diagonal-fraction delta + P/R/F1."""
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(submitted, dtype=float).reshape(-1, 2)
    b = np.asarray(re_derived, dtype=float).reshape(-1, 2)
    diag = float(np.hypot(image_shape[0], image_shape[1]))
    if diag <= 0:
        diag = 1.0

    if len(a) == 0 and len(b) == 0:
        return 0.0, {"recall": 1.0, "precision": 1.0, "f1": 1.0,
                     "mean_match_dist_px": 0.0, "n_submitted": 0,
                     "n_re_derived": 0, "n_matched": 0}
    if len(a) == 0 or len(b) == 0:
        return 1.0, {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                     "mean_match_dist_px": float("inf"),
                     "n_submitted": len(a), "n_re_derived": len(b),
                     "n_matched": 0}

    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    rows, cols = linear_sum_assignment(cost)
    pair_d = cost[rows, cols]
    matched_mask = pair_d <= float(match_tol_px)
    n_matched = int(matched_mask.sum())
    matched_d = pair_d[matched_mask]

    recall = n_matched / len(a) if len(a) > 0 else 0.0
    precision = n_matched / len(b) if len(b) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    mean_d = float(matched_d.mean()) if n_matched > 0 else float(pair_d.mean())

    return mean_d / diag, {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "mean_match_dist_px": float(mean_d),
        "n_submitted": int(len(a)),
        "n_re_derived": int(len(b)),
        "n_matched": int(n_matched),
    }


def _delta_dict(a, b) -> tuple[float, dict]:
    """Per-key delta; max across the intersection. Missing keys ⇒ flag."""
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    per_key: dict = {}
    deltas = []
    for k in common:
        per_key[k] = _delta_scalar(a[k], b[k])
        deltas.append(per_key[k])
    details = {"per_key": per_key, "missing_in_re_derived": list(only_a),
               "missing_in_submitted": list(only_b)}
    if only_a or only_b:
        return 1.0, details
    if not deltas:
        return 0.0, details
    return float(max(deltas)), details


def _classify_severity(
    delta: float, rel_tol_flag: float, rel_tol_block: float,
    abs_delta: Optional[float] = None, abs_tol: Optional[float] = None,
    extra_floor: bool = False,
) -> str:
    """Map a normalised delta to ok / flag / block."""
    if abs_tol is not None and abs_delta is not None and abs_delta <= abs_tol:
        return "ok"
    if delta <= rel_tol_flag and not extra_floor:
        return "ok"
    if delta > rel_tol_block:
        return "block"
    if delta > rel_tol_flag or extra_floor:
        return "flag"
    return "ok"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_vs_submit_check(
    *,
    image: np.ndarray,
    submitted_value: Any,
    recipe_re_detect_fn: Callable[[np.ndarray], Any],
    rel_tol_flag: float = DEFAULT_REL_TOL_FLAG,
    rel_tol_block: float = DEFAULT_REL_TOL_BLOCK,
    abs_tol: Optional[float] = None,
    position_match_tol_frac: float = DEFAULT_POS_MATCH_TOL_FRAC,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> RenderSubmitCheck:
    """Re-run the recipe's detector on the rendered image; compare numbers.

    Args:
        image: The image whose answer is about to be submitted (typically
            the final post-acquisition snap or showcase frame).
        submitted_value: The value (scalar / list / list of [x, y] /
            dict) the recipe is about to submit via ``submit_solution``.
        recipe_re_detect_fn: ``Callable[[np.ndarray], same-shape-as-
            submitted_value]`` — the recipe's own detector. Required.
        rel_tol_flag: Relative-error threshold above which severity is
            promoted to ``flag``.
        rel_tol_block: Relative-error threshold above which severity is
            promoted to ``block``.
        abs_tol: Optional absolute-error escape hatch for low-count
            integer cases (3 vs 4 is 33% relative but only 1 absolute).
            When supplied, ``|abs_delta| <= abs_tol`` forces ``ok``.
        position_match_tol_frac: For position lists, fraction of image
            diagonal within which a Hungarian-matched pair counts as
            ``matched``. Default 0.02 ≈ 10 px on a 512×512.
        recall_floor: For position lists, recall < floor forces severity
            ≥ ``flag`` even if mean-distance delta is small.

    Returns:
        ``RenderSubmitCheck`` with re_derived_value / submitted_value /
        delta / severity (ok|flag|block|error) / rationale / details.
    """
    if recipe_re_detect_fn is None:
        raise ValueError(
            "recipe_re_detect_fn is required — render_vs_submit_check "
            "is intentionally a re-derivation check using the recipe's "
            "own detector. There is no generic-detector fallback."
        )

    try:
        re_derived = recipe_re_detect_fn(image)
    except Exception as e:  # noqa: BLE001
        return RenderSubmitCheck(
            re_derived_value=None,
            submitted_value=submitted_value,
            delta=float("nan"),
            severity="error",
            rationale=f"re-detect raised {type(e).__name__}: {e}",
            details={"exception_type": type(e).__name__, "message": str(e)},
        )

    img_shape = image.shape[:2] if hasattr(image, "shape") else (1, 1)

    # Position-list dispatch (must come before list dispatch)
    if _is_position_list(submitted_value) and _is_position_list(re_derived):
        match_tol_px = position_match_tol_frac * float(np.hypot(*img_shape))
        delta, details = _delta_positions(
            submitted_value, re_derived, img_shape, match_tol_px,
        )
        recall_low = details.get("recall", 1.0) < recall_floor
        severity = _classify_severity(
            delta, rel_tol_flag, rel_tol_block, extra_floor=recall_low,
        )
        return RenderSubmitCheck(
            re_derived_value=re_derived,
            submitted_value=submitted_value,
            delta=float(delta),
            severity=severity,
            rationale=(
                f"positions: recall={details['recall']:.2f}, "
                f"precision={details['precision']:.2f}, "
                f"mean_dist={details['mean_match_dist_px']:.1f}px, "
                f"diag-frac delta={delta:.3f}"
            ),
            details=details,
        )

    if isinstance(submitted_value, dict) and isinstance(re_derived, dict):
        delta, details = _delta_dict(submitted_value, re_derived)
        severity = _classify_severity(delta, rel_tol_flag, rel_tol_block)
        return RenderSubmitCheck(
            re_derived_value=re_derived,
            submitted_value=submitted_value,
            delta=float(delta),
            severity=severity,
            rationale=f"dict: max per-key delta={delta:.3f}",
            details=details,
        )

    if isinstance(submitted_value, (list, tuple, np.ndarray)) and \
            isinstance(re_derived, (list, tuple, np.ndarray)):
        delta, details = _delta_list(submitted_value, re_derived)
        severity = _classify_severity(delta, rel_tol_flag, rel_tol_block)
        return RenderSubmitCheck(
            re_derived_value=re_derived,
            submitted_value=submitted_value,
            delta=float(delta),
            severity=severity,
            rationale=f"list: max element delta={delta:.3f}",
            details=details,
        )

    # Scalar path (numbers, bools, strings)
    try:
        delta = _delta_scalar(submitted_value, re_derived)
        abs_delta = abs(float(submitted_value) - float(re_derived))
    except (TypeError, ValueError):
        delta = 0.0 if submitted_value == re_derived else 1.0
        abs_delta = None
    severity = _classify_severity(
        delta, rel_tol_flag, rel_tol_block, abs_delta=abs_delta, abs_tol=abs_tol,
    )
    return RenderSubmitCheck(
        re_derived_value=re_derived,
        submitted_value=submitted_value,
        delta=float(delta),
        severity=severity,
        rationale=(
            f"scalar: submitted={submitted_value} re_derived={re_derived} "
            f"rel_delta={delta:.3f}"
        ),
        details={"abs_delta": abs_delta},
    )
