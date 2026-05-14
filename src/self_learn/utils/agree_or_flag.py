"""Two-estimator agreement check.

When two independent estimators give the same physical quantity,
agreement validates both; disagreement reveals a bug in the model
or the measurement. Don't naively combine — compare.

The pattern landed ch609 r3 = 10/10 (Pinkard 2019 single-shot AO):
the Brenner pre-scan estimate of `tissue_z` (`z_focus_approx`) and
the off-axis-LED estimate (`-dx_probe / slope`) agreed within 0.5 µm,
which validated both. r2 (6/10) tried to *combine* them and
double-counted; r3 used one as the answer and the other as the
cross-check. Captured durably in
`feedback_two_estimator_cross_check.md`.

This module exposes a tiny primitive that recipes can call:

    from self_learn.utils.agree_or_flag import agree_or_flag

    agreement = agree_or_flag(
        name="tissue_z",
        a=z_focus_approx,    a_label="Brenner pre-scan",
        b=-dx_probe / slope, b_label="off-axis LED slope",
        rel_tol_flag=0.10,  abs_tol_flag=0.5,
        rel_tol_block=0.30, abs_tol_block=2.0,
    )
    if agreement.severity == "block":
        raise SystemExit(agreement.rationale)

Severity ladder (matches `RenderSubmitCheck`):
    ok    — relative OR absolute flag-tolerance passes.
    flag  — flag-tolerance fails on at least one axis but block
            tolerance passes on both.
    block — both block-level tolerances violated, OR a non-finite
            input was supplied.

What this module does NOT do:
    - No array / list handling. For per-element comparisons (per-well
      IC50, per-cell counts) use `render_vs_submit_check`, which
      already handles list and position-list shapes.
    - No N-way generalisation. Two estimators only — pairwise
      composition reveals *which* pair disagrees.
    - No auto-pick. `mean` is a convenience; the caller chooses
      which estimator to submit.
    - No orchestrator integration. `presubmit.py` works on the
      answer; this works on intermediate values, one tier earlier.

Sprint #24 (counter=84).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


_EPS = 1e-12


@dataclass(frozen=True)
class EstimatorAgreement:
    """Outcome of a two-estimator cross-check."""

    name: str
    a: float
    b: float
    a_label: str
    b_label: str
    mean: float
    delta: float
    rel_delta: float
    severity: str               # "ok" | "flag" | "block"
    rationale: str
    details: dict = field(default_factory=dict)


def agree_or_flag(
    *,
    name: str,
    a: float,
    b: float,
    a_label: Optional[str] = None,
    b_label: Optional[str] = None,
    rel_tol_flag: float = 0.05,
    abs_tol_flag: Optional[float] = None,
    rel_tol_block: float = 0.20,
    abs_tol_block: Optional[float] = None,
) -> EstimatorAgreement:
    """Compare two scalar estimators of the same quantity.

    Args:
        name: what is being estimated (e.g. ``"tissue_z"``); used in
            ``rationale`` and details.
        a, b: the two scalar estimates. Coerced to ``float``.
        a_label, b_label: human-readable labels for the rationale; default
            to ``"estimator A"`` / ``"estimator B"``.
        rel_tol_flag: relative-error threshold below which severity stays
            ok (default 5%).
        abs_tol_flag: optional absolute-error threshold; passing *either*
            the relative or the absolute flag-level test keeps severity
            ok. Default ``None`` disables the absolute axis.
        rel_tol_block: relative-error threshold above which severity is
            block.
        abs_tol_block: optional absolute-error threshold for block.

    Returns:
        :class:`EstimatorAgreement` with ``severity`` ∈ ok/flag/block.

    Severity rules (OR-for-ok, see module docstring):
      - ok      iff rel_pass_flag  OR  abs_pass_flag
      - block   iff (not rel_pass_block) AND (not abs_pass_block)
      - flag    otherwise
      - block   if either input is non-finite (NaN / ±inf), with
        ``delta=NaN``.
    """
    a_f = float(a)
    b_f = float(b)
    al = a_label or f"{name} estimator A"
    bl = b_label or f"{name} estimator B"

    # Non-finite guard.
    if not (math.isfinite(a_f) and math.isfinite(b_f)):
        return EstimatorAgreement(
            name=str(name),
            a=a_f, b=b_f, a_label=al, b_label=bl,
            mean=float("nan"),
            delta=float("nan"),
            rel_delta=float("nan"),
            severity="block",
            rationale=(
                f"non-finite input — {al}={a_f!r}, {bl}={b_f!r}"
            ),
            details={"reason": "non_finite_input"},
        )

    delta = abs(a_f - b_f)
    denom = max(abs(a_f), abs(b_f), _EPS)
    rel_delta = delta / denom

    rel_pass_flag = rel_delta <= rel_tol_flag
    abs_pass_flag = (abs_tol_flag is not None) and (delta <= abs_tol_flag)
    rel_pass_block = rel_delta <= rel_tol_block
    abs_pass_block = (abs_tol_block is not None) and (delta <= abs_tol_block)

    if rel_pass_flag or abs_pass_flag:
        severity = "ok"
    elif (not rel_pass_block) and (not abs_pass_block):
        severity = "block"
    else:
        severity = "flag"

    rationale = (
        f"{name}: {al}={a_f:.4g}, {bl}={b_f:.4g}, "
        f"|Δ|={delta:.4g} ({rel_delta:.1%}) → {severity}"
    )

    return EstimatorAgreement(
        name=str(name),
        a=a_f, b=b_f, a_label=al, b_label=bl,
        mean=0.5 * (a_f + b_f),
        delta=delta,
        rel_delta=rel_delta,
        severity=severity,
        rationale=rationale,
        details={
            "rel_pass_flag": rel_pass_flag,
            "abs_pass_flag": abs_pass_flag,
            "rel_pass_block": rel_pass_block,
            "abs_pass_block": abs_pass_block,
            "rel_tol_flag": rel_tol_flag,
            "abs_tol_flag": abs_tol_flag,
            "rel_tol_block": rel_tol_block,
            "abs_tol_block": abs_tol_block,
        },
    )
