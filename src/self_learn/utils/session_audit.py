"""Session audit — score distributions across challenge history.

Mines ``logs/challenges/<id>_<slug>/{submission.json, grade.json}``
to surface where recipes / strategies fail and where they reliably
ship. Useful before a knowledge note pass to confirm hunches with
data, or as a /loop idle-mode check to spot drift.

API:
    iter_history(logs_dir=None) → iterate `(challenge_id, sub, grade)` tuples.
    score_distribution(rows) → dict of {label: stats}.
    audit_session() → top-level convenience that returns the structured
        report.

What this is NOT:
    - Not a recipe-quality grading model — score variance is one
      observation among many.
    - Not a replacement for the grader's per-challenge feedback.
    - Not auto-rerun: it reads, it doesn't dispatch.

Sprint #23 (counter=65).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


_DEFAULT_LOGS = Path.home() / "sync/phd/code/self-learning/logs/challenges"
_AGENT_ROOT = Path(__file__).resolve().parents[3]


def _existing_utils() -> set:
    """Return current src/core/utils/<name>.py module names (no .py)."""
    utils_dir = _AGENT_ROOT / "src" / "core" / "utils"
    if not utils_dir.is_dir():
        return set()
    return {p.stem for p in utils_dir.glob("*.py") if p.stem != "__init__"}


def _existing_recipes() -> set:
    recipes_dir = _AGENT_ROOT / "src" / "recipes"
    if not recipes_dir.is_dir():
        return set()
    return {p.stem for p in recipes_dir.glob("*.py") if p.stem != "__init__"}
_RECIPE_RE = re.compile(r"src\.recipes\.(\w+)")
_RECIPE_FROM_RE = re.compile(r"from\s+src\.recipes\.(\w+)\s+import")
_CORE_MOD_RE = re.compile(r"src\.core\.(?:detection|analysis|workflows|utils)\.(\w+)")
_CORE_FROM_RE = re.compile(
    r"from\s+src\.core\.(?:detection|analysis|workflows|utils)\.(\w+)\s+import"
)
_UTILS_RE = re.compile(r"src\.core\.utils\.(\w+)")
_UTILS_FROM_RE = re.compile(r"from\s+src\.core\.utils\.(\w+)\s+import")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryRow:
    """One scored submission attempt."""

    challenge_id: int
    slug: str
    score: int
    max_score: int
    passed: bool
    round: int
    timestamp: str
    method_description: str
    recipes_mentioned: tuple
    core_modules_mentioned: tuple
    utils_mentioned: tuple = ()
    code_lines: int = 0
    has_src_imports: bool = False

    @property
    def score_frac(self) -> float:
        return float(self.score) / max(float(self.max_score), 1.0)

    @property
    def is_inline_violation(self) -> bool:
        """``True`` when the submission has >30 code lines AND zero
        ``from src.`` imports — a NON_NEGOTIABLES rule 4 violation.
        """
        return self.code_lines > 30 and not self.has_src_imports


@dataclass
class ScoreStats:
    """Score distribution for one label (recipe / module / overall)."""

    label: str
    n: int
    mean: float
    median: float
    stdev: float
    min: int
    max: int
    pass_rate: float          # fraction of submissions where passed=True
    examples: list = field(default_factory=list)   # (challenge_id, score)

    @classmethod
    def from_rows(cls, label: str, rows: list) -> "ScoreStats":
        if not rows:
            return cls(label=label, n=0, mean=0.0, median=0.0, stdev=0.0,
                       min=0, max=0, pass_rate=0.0, examples=[])
        scores = [r.score for r in rows]
        return cls(
            label=label,
            n=len(rows),
            mean=float(statistics.mean(scores)),
            median=float(statistics.median(scores)),
            stdev=float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0,
            min=int(min(scores)),
            max=int(max(scores)),
            pass_rate=sum(1 for r in rows if r.passed) / len(rows),
            examples=[(r.challenge_id, r.score) for r in rows[-5:]],
        )


@dataclass(frozen=True)
class Trend:
    """Recent-vs-historical score comparison for one recipe / module."""

    label: str
    n_recent: int
    n_historical: int
    recent_mean: float
    historical_mean: float
    delta: float                    # recent_mean - historical_mean
    direction: str                  # "improving" | "stable" | "degrading" | "insufficient_data"


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

def iter_history(logs_dir: Optional[Path] = None) -> Iterator[HistoryRow]:
    """Yield one ``HistoryRow`` per scored challenge.

    Pulls from ``submission.json`` (latest round) + ``grade.json``.
    Skips challenges that have neither.
    """
    base = Path(logs_dir) if logs_dir else _DEFAULT_LOGS
    if not base.exists():
        return
    for ch_dir in sorted(base.iterdir()):
        if not ch_dir.is_dir():
            continue
        sub_path = ch_dir / "submission.json"
        grade_path = ch_dir / "grade.json"
        if not sub_path.exists() or not grade_path.exists():
            continue
        try:
            sub = json.loads(sub_path.read_text())
            grade = json.loads(grade_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ch_id = grade.get("challenge_id") or sub.get("challenge_id")
        if ch_id is None:
            continue
        method = sub.get("method_description") or ""
        code = sub.get("code_used") or ""
        # Mine both the method prose and the actual solve script.
        recipes = set(_RECIPE_RE.findall(method))
        recipes.update(_RECIPE_FROM_RE.findall(code))
        core_mods = set(_CORE_MOD_RE.findall(method))
        core_mods.update(_CORE_FROM_RE.findall(code))
        utils = set(_UTILS_RE.findall(method))
        utils.update(_UTILS_FROM_RE.findall(code))
        # Platform-usage signal for NON_NEGOTIABLES rule 4 enforcement.
        # An "inline violation" is >30 lines of code without any
        # ``from src.`` imports — meaning the agent reimplemented work
        # the platform already provides.
        code_lines = code.count("\n") if code else 0
        has_src_imports = bool(re.search(r"\bfrom\s+src\.", code))
        yield HistoryRow(
            challenge_id=int(ch_id),
            slug=ch_dir.name,
            score=int(grade.get("score", 0)),
            max_score=int(grade.get("max_score", 10)),
            passed=bool(grade.get("passed", False)),
            round=int(grade.get("round", 1)),
            timestamp=str(grade.get("timestamp", "")),
            method_description=method,
            recipes_mentioned=tuple(sorted(recipes)),
            core_modules_mentioned=tuple(sorted(core_mods)),
            utils_mentioned=tuple(sorted(utils)),
            code_lines=code_lines,
            has_src_imports=has_src_imports,
        )


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def score_distribution(
    rows: Iterable[HistoryRow], *, group_by: str = "recipe",
) -> dict:
    """Group rows by recipe / core_module / overall and return stats per label."""
    rows = list(rows)
    buckets: dict = {}

    if group_by == "overall":
        return {"overall": ScoreStats.from_rows("overall", rows)}

    if group_by == "recipe":
        for r in rows:
            for recipe in r.recipes_mentioned or ("__inline_solve__",):
                buckets.setdefault(recipe, []).append(r)
    elif group_by == "core_module":
        for r in rows:
            for mod in r.core_modules_mentioned or ("__no_core_mod__",):
                buckets.setdefault(mod, []).append(r)
    elif group_by == "utils":
        for r in rows:
            for util in r.utils_mentioned or ("__inline_solve__",):
                buckets.setdefault(util, []).append(r)
    else:
        raise ValueError(f"unknown group_by: {group_by!r}")

    return {label: ScoreStats.from_rows(label, bucket_rows)
            for label, bucket_rows in buckets.items()}


def brittle_recipes(
    distribution: dict, *, min_n: int = 2, stdev_threshold: float = 1.5,
) -> list:
    """Recipes/modules with high variance — candidates for hardening."""
    return sorted(
        [s for s in distribution.values() if s.n >= min_n and s.stdev >= stdev_threshold],
        key=lambda s: -s.stdev,
    )


def trend_per_recipe(
    rows: Iterable[HistoryRow],
    *,
    window: int = 5,
    delta_threshold: float = 1.0,
    group_by: str = "recipe",
) -> dict:
    """Compare recent N submissions to all earlier ones, per label.

    For each recipe (or core module if ``group_by="core_module"``),
    splits its rows into ``recent = last window`` vs
    ``historical = the rest``, and returns a :class:`Trend` capturing
    the mean delta + a categorical direction:

    - ``improving``    if delta ≥ ``delta_threshold``
    - ``degrading``    if delta ≤ ``-delta_threshold``
    - ``stable``       if both halves have ≥ 2 entries and |delta| < threshold
    - ``insufficient_data``  if either half has < 2 entries

    Order: rows are sorted by ``timestamp`` so the most recent
    submissions land in the recent bucket.
    """
    if group_by not in ("recipe", "core_module", "utils"):
        raise ValueError(f"unknown group_by: {group_by!r}")

    rows = sorted(rows, key=lambda r: r.timestamp)
    buckets: dict = {}
    for r in rows:
        if group_by == "recipe":
            labels = r.recipes_mentioned
        elif group_by == "core_module":
            labels = r.core_modules_mentioned
        else:  # "utils"
            labels = r.utils_mentioned
        for lbl in (labels or ()):
            buckets.setdefault(lbl, []).append(r)

    trends: dict = {}
    for label, bucket_rows in buckets.items():
        n = len(bucket_rows)
        if n < 2:
            trends[label] = Trend(
                label=label, n_recent=n, n_historical=0,
                recent_mean=float(bucket_rows[0].score) if bucket_rows else 0.0,
                historical_mean=0.0, delta=0.0,
                direction="insufficient_data",
            )
            continue
        # Take the last `window` rows as recent; the rest are historical.
        cutoff = max(0, n - int(window))
        historical = bucket_rows[:cutoff]
        recent = bucket_rows[cutoff:]
        if len(recent) < 2 or len(historical) < 2:
            trends[label] = Trend(
                label=label, n_recent=len(recent), n_historical=len(historical),
                recent_mean=float(statistics.mean([r.score for r in recent])) if recent else 0.0,
                historical_mean=float(statistics.mean([r.score for r in historical])) if historical else 0.0,
                delta=0.0,
                direction="insufficient_data",
            )
            continue
        recent_mean = float(statistics.mean(r.score for r in recent))
        historical_mean = float(statistics.mean(r.score for r in historical))
        delta = recent_mean - historical_mean
        if delta >= delta_threshold:
            direction = "improving"
        elif delta <= -delta_threshold:
            direction = "degrading"
        else:
            direction = "stable"
        trends[label] = Trend(
            label=label,
            n_recent=len(recent),
            n_historical=len(historical),
            recent_mean=recent_mean,
            historical_mean=historical_mean,
            delta=delta,
            direction=direction,
        )
    return trends


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def inline_violation_rate(rows: list, *, window: int = 20) -> dict:
    """Drift signal: fraction of last-``window`` submissions that
    violate NON_NEGOTIABLES rule 4 (>30 inline lines, no ``from src.``).

    Returns a dict with ``window``, ``n_violations``, ``violation_rate``,
    and ``violations`` (list of (challenge_id, code_lines) for each).
    A row's ``is_inline_violation`` property is the boolean source.
    """
    if not rows:
        return {"window": window, "n_violations": 0, "violation_rate": 0.0,
                "violations": []}
    recent = sorted(rows, key=lambda r: r.challenge_id)[-window:]
    violations = [(r.challenge_id, r.code_lines)
                  for r in recent if r.is_inline_violation]
    return {
        "window": len(recent),
        "n_violations": len(violations),
        "violation_rate": len(violations) / len(recent),
        "violations": violations,
    }


def audit_session(logs_dir: Optional[Path] = None, *, trend_window: int = 5,
                   inline_window: int = 20) -> dict:
    """One-call audit. Returns a dict with:

    - ``n_graded``: total number of graded submissions found.
    - ``overall``: :class:`ScoreStats` across everything.
    - ``by_recipe``: ``{recipe: ScoreStats}``.
    - ``by_core_module``: ``{core_module: ScoreStats}``.
    - ``brittle_recipes``: list of :class:`ScoreStats` with stdev ≥ 1.5.
    - ``trends``: ``{recipe: Trend}`` — recent vs historical mean.
    - ``inline_violations``: NON_NEGOTIABLES rule 4 drift over the last
      ``inline_window`` submissions.
    """
    rows = list(iter_history(logs_dir))
    by_utils = score_distribution(rows, group_by="utils")
    return {
        "n_graded": len(rows),
        "overall": score_distribution(rows, group_by="overall")["overall"],
        "by_recipe": score_distribution(rows, group_by="recipe"),
        "by_core_module": score_distribution(rows, group_by="core_module"),
        "by_utils": by_utils,
        "brittle_recipes": brittle_recipes(
            score_distribution(rows, group_by="recipe")
        ),
        "brittle_utils": brittle_recipes(by_utils),
        "trends": trend_per_recipe(rows, window=trend_window),
        "utils_trends": trend_per_recipe(rows, window=trend_window, group_by="utils"),
        "inline_violations": inline_violation_rate(rows, window=inline_window),
    }


def format_report(report: dict) -> str:
    """Pretty-print an audit report for human / log consumption."""
    lines = [f"=== session audit (n_graded={report['n_graded']}) ==="]
    o = report["overall"]
    lines.append(
        f"overall: mean={o.mean:.2f}/{10}  median={o.median:.0f}  "
        f"stdev={o.stdev:.2f}  pass_rate={o.pass_rate:.1%}  range=[{o.min}, {o.max}]"
    )
    lines.append("")
    existing_recipes = _existing_recipes()
    existing_utils = _existing_utils()

    def _label_with_status(name: str, existing: set) -> str:
        if name.startswith("__") or name in existing:
            return name
        return f"{name} [deleted]"

    lines.append("Top 10 recipes by submission count:")
    by_recipe = sorted(report["by_recipe"].values(), key=lambda s: -s.n)[:10]
    for s in by_recipe:
        label = _label_with_status(s.label, existing_recipes)
        lines.append(
            f"  {label:45s}  n={s.n:3d}  mean={s.mean:5.2f}  "
            f"stdev={s.stdev:4.2f}  range=[{s.min}, {s.max}]"
        )
    if report.get("by_utils"):
        lines.append("")
        lines.append("Top 10 utils by submission count:")
        by_utils = sorted(report["by_utils"].values(), key=lambda s: -s.n)[:10]
        for s in by_utils:
            label = _label_with_status(s.label, existing_utils)
            lines.append(
                f"  {label:45s}  n={s.n:3d}  mean={s.mean:5.2f}  "
                f"stdev={s.stdev:4.2f}  range=[{s.min}, {s.max}]"
            )
    if report["brittle_recipes"]:
        lines.append("")
        lines.append("Brittle recipes (stdev ≥ 1.5):")
        for s in report["brittle_recipes"]:
            lines.append(
                f"  ⚠ {s.label:33s}  n={s.n:3d}  stdev={s.stdev:.2f}  "
                f"min={s.min} (recent: {s.examples[-3:]})"
            )
    iv = report.get("inline_violations") or {}
    if iv.get("window"):
        rate = iv["violation_rate"]
        marker = "⚠ " if rate >= 0.20 else "  "
        lines.append("")
        lines.append(
            f"{marker}NON_NEG #4 inline-rate (last {iv['window']}): "
            f"{iv['n_violations']}/{iv['window']} = {rate:.0%}"
        )
        if iv["violations"]:
            shown = ", ".join(
                f"ch{cid}({n}L)" for cid, n in iv["violations"][-6:]
            )
            lines.append(f"    recent violations: {shown}")
    if report.get("trends"):
        improving = [t for t in report["trends"].values() if t.direction == "improving"]
        degrading = [t for t in report["trends"].values() if t.direction == "degrading"]
        if improving or degrading:
            lines.append("")
            lines.append("Recipe trends (recent vs historical):")
            for t in degrading:
                lines.append(
                    f"  ↓ {t.label:33s}  recent={t.recent_mean:.2f} ({t.n_recent}) "
                    f"vs historical={t.historical_mean:.2f} ({t.n_historical}) "
                    f"Δ={t.delta:+.2f}"
                )
            for t in improving:
                lines.append(
                    f"  ↑ {t.label:33s}  recent={t.recent_mean:.2f} ({t.n_recent}) "
                    f"vs historical={t.historical_mean:.2f} ({t.n_historical}) "
                    f"Δ={t.delta:+.2f}"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point — `python -m src.core.utils.session_audit`
# ---------------------------------------------------------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Mine logs/challenges for score distributions."
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=None,
        help=f"override logs/challenges dir (default: {_DEFAULT_LOGS})",
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="recent-vs-historical window for trend detection",
    )
    parser.add_argument(
        "--inline-window", type=int, default=20,
        help="window for NON_NEG #4 inline-violation rate",
    )
    args = parser.parse_args()
    report = audit_session(
        args.logs_dir,
        trend_window=args.window,
        inline_window=args.inline_window,
    )
    print(format_report(report))


if __name__ == "__main__":
    _main()
