"""Pre-submit lint for ``code_used`` against the NON_NEGOTIABLES.

Catches the failure modes that ``session_audit`` only surfaces *after*
a submission is graded:

  - NN rule 1: never read ``truth.json``.
  - NN rule 4: solutions must import from ``src.`` (>30 inline lines
    without any ``from src.`` import is a violation).
  - NN rule 7: no template-solving — every challenge approached fresh.

Returns a structured ``LintResult`` with ``errors`` (hard blocks,
should fail submission) and ``warnings`` (soft hints, should warn).
``submit_with_showcase`` invokes this before forwarding to
``submit_solution``; callers can also invoke directly to validate
a candidate solve script before running it.

Sprint #45 (2026-04-28). Counterfactual: ch640 self-lapse
(2026-04-27) where ``cat *.json`` in a challenge dir read truth.json
— a code-side lint would have flagged ``truth.json`` in the script's
own text and aborted before the read happened, even if the agent's
shell command was already off the rails.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ── Patterns ────────────────────────────────────────────────────────────

# NN rule 1: any textual hint that the script reads or names truth.json
# / ground_truth.json. We match liberally (even comments) because
# touching the path is *itself* a contract violation worth catching.
_TRUTH_PATTERNS = (
    re.compile(r"\btruth\.json\b", flags=re.IGNORECASE),
    re.compile(r"\bground[_-]?truth(?:\.json)?\b", flags=re.IGNORECASE),
    re.compile(r"['\"]truth['\"]"),  # ``challenge["truth"]`` lookup
)

_FROM_SRC_RE = re.compile(r"^\s*from\s+src\.", flags=re.MULTILINE)
_IMPORT_SRC_RE = re.compile(r"^\s*import\s+src\.", flags=re.MULTILINE)

# A "code line" is a non-blank, non-comment line. Mirrors the spirit of
# ``session_audit._count_code_lines`` (which uses ``code.count("\n")``)
# but is stricter: blank lines and pure-comment lines don't count.
def _count_code_lines(code: str) -> int:
    n = 0
    for raw in code.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        n += 1
    return n


def _has_src_import(code: str) -> bool:
    return bool(_FROM_SRC_RE.search(code) or _IMPORT_SRC_RE.search(code))


# ── Result type ─────────────────────────────────────────────────────────


@dataclass
class LintResult:
    """Outcome of a pre-submit lint pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    code_lines: int = 0
    has_src_imports: bool = False
    matched_truth_patterns: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ── Top-level lint ──────────────────────────────────────────────────────


def lint_submission(
    code_used: str,
    *,
    prior_code_samples: Iterable[str] = (),
    inline_threshold: int = 30,
    duplicate_similarity: float = 0.95,
    skip_duplicate_check: bool = False,
) -> LintResult:
    """Lint a candidate ``code_used`` string against NON_NEGOTIABLES.

    Args:
        code_used: the full source of the solve script (the same
            string that will be passed to ``submit_solution(code_used=)``).
        prior_code_samples: previously-shipped solve scripts. Each is a
            full source string. Used for the NN rule 7 duplicate check
            — if any prior sample is ≥ ``duplicate_similarity`` close to
            ``code_used`` it's a template-solve violation. Pass an empty
            iterable to skip.
        inline_threshold: line count above which a script with no
            ``from src.`` import is flagged. Mirrors session_audit's
            >30-lines threshold.
        duplicate_similarity: Ratcliff/Obershelp similarity threshold.
            ``difflib.SequenceMatcher.quick_ratio`` ≥ this counts as
            a duplicate.
        skip_duplicate_check: bypass the prior-samples diff (fast
            path for one-shot lint without a history).

    Returns:
        :class:`LintResult` with ``errors`` (hard blocks) and
        ``warnings`` (soft).

    Hard rules (errors):
        - NN rule 1: any textual reference to ``truth.json`` /
          ``ground_truth`` / ``challenge["truth"]``.

    Soft rules (warnings):
        - NN rule 4: ``code_lines > inline_threshold`` AND no
          ``from src.`` / ``import src.`` import line.
        - NN rule 7: ``code_used`` is ≥ ``duplicate_similarity``
          identical to a prior sample.
    """
    result = LintResult()
    result.code_lines = _count_code_lines(code_used)
    result.has_src_imports = _has_src_import(code_used)

    # NN #1
    for pat in _TRUTH_PATTERNS:
        m = pat.search(code_used)
        if m:
            result.matched_truth_patterns.append(m.group(0))
    if result.matched_truth_patterns:
        result.errors.append(
            f"NON_NEGOTIABLES rule 1: code_used references "
            f"{result.matched_truth_patterns[:3]} — never read truth.json. "
            "Remove all references to ground-truth files and resubmit."
        )

    # NN #4
    if (
        result.code_lines > inline_threshold
        and not result.has_src_imports
    ):
        result.warnings.append(
            f"NON_NEGOTIABLES rule 4: {result.code_lines} code lines "
            f"with zero `from src.` imports — solutions must import "
            f"from src/ (no inline analysis > {inline_threshold} lines). "
            "Extract reusable logic to src/core/utils/ or src/recipes/."
        )

    # NN #7
    if not skip_duplicate_check:
        for i, prior in enumerate(prior_code_samples):
            if not prior:
                continue
            ratio = difflib.SequenceMatcher(
                None, code_used, prior, autojunk=False,
            ).quick_ratio()
            if ratio >= duplicate_similarity:
                result.warnings.append(
                    f"NON_NEGOTIABLES rule 7: code_used is "
                    f"{ratio:.0%} similar to prior sample #{i} — "
                    "every challenge must be approached fresh, not "
                    "copy-pasted from a previous solve."
                )
                break  # one warning is enough

    return result


def lint_submission_or_raise(
    code_used: str,
    *,
    prior_code_samples: Iterable[str] = (),
    inline_threshold: int = 30,
    duplicate_similarity: float = 0.95,
) -> LintResult:
    """Same as :func:`lint_submission` but raises ``ValueError`` on errors.

    Convenience for callers that want to abort on hard violations.
    """
    result = lint_submission(
        code_used,
        prior_code_samples=prior_code_samples,
        inline_threshold=inline_threshold,
        duplicate_similarity=duplicate_similarity,
    )
    if result.errors:
        raise ValueError(
            "submission lint failed (hard errors):\n  - "
            + "\n  - ".join(result.errors)
        )
    return result
