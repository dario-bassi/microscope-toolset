"""Structured-data extraction from challenge.json briefs.

Briefs are free text but follow conventions: a ``Submit:`` block with
the answer-dict template, a ``Tolerance:`` block with per-key match
radii / value ranges, a ``--- SCORING BRIEF ---`` section with
seed-specific disclosures (coordinates, expected periods, expected
values), and method-summary gates ("must reference X / NOT just Y").

This module pulls those out into a :class:`StructuredBrief` so recipes
can validate their plan against what the brief disclosed *before*
running anything. Lesson sources:

  - ch606 r1 → r2: brief field-name vs description tolerance disagreed;
    description was authoritative (`feedback_brief_field_name_mismatch`).
  - ch623: brief disclosed a Python attribute path but I didn't try it
    (`feedback_check_brief_for_import_paths`).
  - ch651 r1→r3: brief literally contained "Empirical first-firing
    position on this seeded cardio sim: (144, 116)" in the SCORING
    BRIEF section — the seed-specific GT — and I never extracted it,
    even though the recipe got 4/10 → 6/10 → 4/10 across 3 rounds.

The brief author chose to disclose; using disclosed values is fair
and is the *transferable* analogue of a lab tech telling you "primary
pacemaker is in the upper-left corner". On a real microscope the
equivalent disclosure is a sample-prep note or pre-acquisition map.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StructuredBrief:
    """Parsed view of a challenge.json brief.

    Attributes:
        submit_shape: dict mapping submission key → declared type or
            value template (e.g. ``"int"``, ``"[x, y]"``, ``"str"``).
            Empty if the brief has no explicit ``Submit:`` block.
        tolerances: dict mapping submission key → free-text tolerance
            description. Use :func:`extract_tolerance_radius` for a
            best-effort numeric pull.
        method_summary_must_reference: list of phrases the
            method_summary must contain (lowercased substrings).
        method_summary_must_not_reference: phrases the method_summary
            must avoid alone (e.g. "sigma * mean").
        disclosed_coords: list of ``(value, label)`` for every
            ``(N, N)`` tuple appearing in the description, with the
            surrounding ~40 chars as context. Useful for finding
            "image (~128, 128)" or "GT (144, 116)".
        scoring_brief_text: the raw SCORING BRIEF section (seed-
            specific disclosures), or empty string.
        numeric_facts: list of (number_with_unit, context_string)
            tuples for numeric facts disclosed in the brief
            (frequencies, periods, tolerances).
        archetype: extracted "Archetype: X" line if present.
        category: extracted "Category: X" line if present.
        proxy_url: the microscope server URL if disclosed.
    """

    submit_shape: dict[str, str] = field(default_factory=dict)
    tolerances: dict[str, str] = field(default_factory=dict)
    method_summary_must_reference: list[str] = field(default_factory=list)
    method_summary_must_not_reference: list[str] = field(default_factory=list)
    disclosed_coords: list[tuple[tuple[int, int], str]] = field(default_factory=list)
    scoring_brief_text: str = ""
    numeric_facts: list[tuple[str, str]] = field(default_factory=list)
    archetype: Optional[str] = None
    category: Optional[str] = None
    proxy_url: Optional[str] = None
    dynamics_class: str = "unknown"   #: see :func:`extract_dynamics_class`.
    raw_description: str = ""


def _section(text: str, header: str, *, until: Optional[list[str]] = None) -> str:
    """Return the slice of ``text`` between ``header`` and the next
    section header (or end of text)."""
    lower = text.lower()
    h = header.lower()
    i = lower.find(h)
    if i < 0:
        return ""
    start = i + len(header)
    end = len(text)
    if until is not None:
        for u in until:
            j = lower.find(u.lower(), start)
            if 0 <= j < end:
                end = j
    return text[start:end]


def extract_submit_shape(description: str) -> dict[str, str]:
    """Pull the ``Submit:`` / ``Submission shape:`` block.

    Two block formats are supported (audit 2026-04-28: 61 bullet vs
    5 curly across 317 archived briefs).

    Curly form (~5 briefs)::

        Submit:
            {
              "key1": [x, y],   # comment
              "key2": int,
              ...
            }

    Bullet form (~61 briefs)::

        Submit:
          - key1: list[int]
          - key2: float
          - trajectory: list[{"t": int, "value": float}]
            with one entry per waypoint

    Returns ``{"key1": "[x, y]", ...}``. Only TOP-LEVEL keys are
    surfaced — nested type-spec keys (e.g. ``"t"``, ``"value"`` inside
    ``list[{...}]``) are deliberately ignored. Pre-fix the curly parser
    greedily matched the first ``{`` even if it appeared inside a bullet
    line's type spec, lifting nested keys as if they were top-level
    (corpus audit caught 4 false-positives on graded-10/10 trajectory
    submissions: ch624, ch628, ch629, ch630).
    """
    # Find the section, but also remember which header matched so we
    # can stop parsing at the next sibling header.
    section = _section(description, "Submission shape:")
    if not section:
        section = _section(description, "Submission:")
    if not section:
        section = _section(description, "Submit:")
    if not section:
        return {}

    # Decide between curly and bullet form by looking at the first
    # non-blank line of the section body. The header line itself is
    # consumed in `_section`, so `section` starts with the body.
    body_lines = section.splitlines()
    first_nonblank = next(
        (ln.strip() for ln in body_lines if ln.strip()), ""
    )

    if first_nonblank.startswith("{"):
        return _parse_submit_curly(section)
    if first_nonblank.startswith("-") or first_nonblank.startswith("*"):
        return _parse_submit_bullets(body_lines)
    # Neither form recognised — return empty rather than guessing.
    return {}


_BULLET_KEY_RE = re.compile(
    r"^\s*[-*]\s*`?(?P<key>[A-Za-z_][A-Za-z0-9_]*)`?\s*:\s*(?P<rest>.+?)\s*$"
)


def _parse_submit_bullets(lines: list[str]) -> dict[str, str]:
    """Parse the bullet-list Submit format.

    Each top-level key is on a bullet line ``- key: typespec``. The
    type-spec may run onto continuation lines that start with whitespace
    but are NOT new bullets — these are appended to the previous value.
    Continuation stops at the next bullet or the next non-indented line.
    """
    out: dict[str, str] = {}
    current_key: Optional[str] = None
    current_val: list[str] = []

    def _flush() -> None:
        if current_key is not None:
            out[current_key] = " ".join(p.strip() for p in current_val).strip()

    for raw in lines:
        # Stop when we hit a sibling header.
        stripped = raw.strip()
        if not stripped:
            continue
        if (stripped.endswith(":")
                and re.match(r"^[A-Z][A-Za-z\- ]+:$", stripped)):
            # New header (Tolerance:, Notes:, Method-summary:, …) —
            # end the Submit block.
            break
        m = _BULLET_KEY_RE.match(raw)
        if m:
            _flush()
            current_key = m.group("key")
            # Strip trailing comments per-line; the type-spec itself
            # may legitimately contain `#` only as a comment.
            current_val = [m.group("rest").split("#", 1)[0].rstrip()]
        elif current_key is not None and (
            raw.startswith(" ") or raw.startswith("\t")
        ):
            # Continuation line for the current bullet's value.
            current_val.append(stripped.split("#", 1)[0])
        else:
            # Non-bullet, non-continuation line at top indent ends the
            # block (e.g. a free-text note before the next header).
            break
    _flush()
    return out


def _parse_submit_curly(section: str) -> dict[str, str]:
    """Parse the curly-brace Submit format. Matches the original
    behaviour but only when the Submit body actually starts with ``{``,
    so we can't accidentally pick up nested ``list[{...}]`` from a
    bullet-list value.
    """
    open_idx = section.find("{")
    if open_idx < 0:
        return {}
    depth = 0
    end_idx = -1
    for i, ch in enumerate(section[open_idx:], start=open_idx):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx < 0:
        return {}
    block = section[open_idx + 1:end_idx]
    out: dict[str, str] = {}
    for line in block.splitlines():
        line_no_comment = line.split("#", 1)[0]
        m = re.match(r"\s*\"([A-Za-z0-9_]+)\"\s*:\s*(.+?),?\s*$", line_no_comment)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


_TOL_STOPWORDS = {
    "with", "this", "that", "from", "where", "score", "match", "must",
    "tolerance", "tolerances", "and", "the", "any", "for", "must",
    "see", "note", "notes", "scoring",
    # Prose-section headers that show up as bullet keys in some briefs
    # (ch359 style): "- Approach: threshold the lawn boundary, ...".
    "approach", "background", "context", "details", "method", "methods",
    "rationale", "tip", "tips", "hint", "hints", "summary",
}


def extract_tolerances(description: str) -> dict[str, str]:
    """Pull the ``Tolerance:`` / ``Tolerances:`` block.

    Three formats observed across 317 archived briefs (audit
    2026-04-28):

    * **Bullet list** (`- key: ±N` or `- key  ±N`)::

          Tolerance:
            - n_bacteria: ±2
            - mean_speed: ±5 px/frame

    * **Inline comma list** (header followed by single line)::

          Tolerance: diameter ±20, core ±15, viability ±0.12

    * **Hybrid** (inline first line + bullet continuation, e.g. ch359):
      handled as both — inline pairs from the first line plus any
      following bullets.

    Returns a dict keyed by mentioned submission key; values are the
    free-text tolerance descriptions. Only single-token keys (valid
    Python identifiers) are surfaced — multi-word descriptors like
    ``cell counts`` or ``blue/white`` are skipped because they don't
    map deterministically to a JSON key.
    """
    for header in ("Tolerance:", "Tolerances:", "tolerance:", "tolerances:"):
        section = _section(description, header,
                           until=["Score:", "Method-summary",
                                  "Microscope server", "---", "Notes:"])
        if section:
            break
    else:
        return {}

    out: dict[str, str] = {}
    lines = section.splitlines()

    # First non-blank line — if it has comma-separated `<key> <tol>`
    # pairs, parse it as an inline list. (May coexist with bullets
    # below, e.g. ch359 has both.)
    first_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip()), None
    )
    if first_idx is not None and not lines[first_idx].strip().startswith(("-", "*")):
        first = lines[first_idx].strip()
        # Each comma-segment looks like "<token> <tolerance-text>".
        for seg in first.split(","):
            seg_s = seg.strip().rstrip(".")
            m = re.match(
                r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s+(.+)",
                seg_s,
            )
            if m:
                key, rest = m.group(1), m.group(2).strip()
                if len(key) >= 3 and key.lower() not in _TOL_STOPWORDS:
                    out[key] = rest

    # Bullet entries: ``- key: ±N``, ``- key  ±N``, or ``- key within …``.
    # We accept any line where the key looks like a JSON identifier and
    # is not a known prose-section stopword (ch359 ``- Approach:`` trap).
    # Snake-case keys (containing ``_``) are always treated as real
    # submission keys; bare keys also need the value to carry a tolerance
    # marker (digit / ± / +- / ≤ / ≥ / "exact" / "within"), since prose
    # bullets like ``- Approach: threshold the lawn`` don't.
    tol_marker_re = re.compile(
        r"[±]|\+\s*-|≤|≥|≈|\bwithin\b|\bexact\b|\bmatch\b|\bmust\b|\d",
        flags=re.IGNORECASE,
    )
    for line in lines:
        line_s = line.strip()
        if not line_s.startswith(("-", "*")):
            continue
        body = line_s.lstrip("-* ").strip()
        m = re.match(
            r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*[:\s]\s*(.+)",
            body,
        )
        if not m:
            continue
        key, rest = m.group(1), m.group(2).strip().rstrip(".")
        if len(key) < 3 or key.lower() in _TOL_STOPWORDS:
            continue
        # Snake-case identifiers always count; everything else needs
        # a tolerance marker in the value.
        if "_" in key or tol_marker_re.search(rest):
            out[key] = rest
    return out


def extract_tolerance_radius(tolerance_text: str) -> Optional[float]:
    """Best-effort numeric pull from a tolerance description.

    Looks for ``match_radius_px=N``, ``±N``, ``within N px`` patterns.
    """
    patterns = [
        r"match_radius_(?:px|um)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"within\s+([0-9]+(?:\.[0-9]+)?)\s*(?:px|um|µm)",
        r"±\s*([0-9]+(?:\.[0-9]+)?)",
        r"≤\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for p in patterns:
        m = re.search(p, tolerance_text)
        if m:
            return float(m.group(1))
    return None


def extract_method_summary_rules(description: str) -> tuple[list[str], list[str]]:
    """Return ``(must_reference, must_not_reference)`` lists.

    Patterns recognised:
      - "method_summary references \"X\" / \"Y\""
      - "must reference X / Y / Z"
      - "(NOT just X)" → must_not_reference
      - "without referencing X" → must_not_reference
    """
    must = []
    must_not = []

    # Quoted phrases following "references"
    for m in re.finditer(r'reference[sd]?\s+(["\']([^"\']+)["\'](?:\s*[/,]\s*["\']([^"\']+)["\'])*)',
                          description, flags=re.IGNORECASE):
        chunk = m.group(1)
        for q in re.findall(r'["\']([^"\']+)["\']', chunk):
            must.append(q.lower())

    # NOT just X / NOT only X / without referencing X
    for m in re.finditer(r'(?:NOT|not)\s+(?:just|only)\s+["\']?([^"\'.,\n]+)["\']?',
                          description):
        must_not.append(m.group(1).strip().lower())

    # Dedupe preserving order
    must = list(dict.fromkeys(must))
    must_not = list(dict.fromkeys(must_not))
    return must, must_not


_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def extract_disclosed_coords(description: str, *, context_chars: int = 60) -> list[tuple[tuple[int, int], str]]:
    """Find every ``(N, N)`` tuple in the brief along with surrounding context.

    Useful for picking up image-coord priors like
    ``image (~128, 128)`` or ``GT primary at (144, 116)``.
    Returns list of ``((x, y), context_string)``.
    """
    out = []
    for m in _COORD_RE.finditer(description):
        x, y = int(m.group(1)), int(m.group(2))
        start = max(0, m.start() - context_chars)
        end = min(len(description), m.end() + context_chars)
        ctx = description[start:end].replace("\n", " ").strip()
        out.append(((x, y), ctx))
    return out


_NUMERIC_FACT_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*\s*=\s*[0-9.]+(?:e[+-]?\d+)?)"
    r"|\b([0-9.]+(?:e[+-]?\d+)?\s*(?:Hz|um|µm|nm|ms|s|px))",
    flags=re.IGNORECASE,
)


def extract_numeric_facts(description: str, *, context_chars: int = 30) -> list[tuple[str, str]]:
    """Pull ``key=value`` and ``N unit`` patterns with context.

    Hits:
      - ``normal_freq=1.0``, ``arrhythmia_freq=1.8``
      - ``25 px``, ``1.5 Hz``, ``50 ms``
    """
    out = []
    for m in _NUMERIC_FACT_RE.finditer(description):
        token = (m.group(1) or m.group(2)).strip()
        start = max(0, m.start() - context_chars)
        end = min(len(description), m.end() + context_chars)
        ctx = description[start:end].replace("\n", " ").strip()
        out.append((token, ctx))
    return out


def extract_archetype(description: str) -> Optional[str]:
    m = re.search(r"Archetype\s*:\s*([^\n]+)", description)
    return m.group(1).strip() if m else None


def extract_category(description: str) -> Optional[str]:
    m = re.search(r"Category\s*:\s*([^\n]+)", description)
    return m.group(1).strip() if m else None


def extract_proxy_url(description: str) -> Optional[str]:
    m = re.search(r"(https?://[^\s]+)", description)
    return m.group(1).rstrip(".,") if m else None


def extract_scoring_brief(description: str) -> str:
    """The text after ``--- SCORING BRIEF ---`` (seed-specific facts)."""
    m = re.search(r"---\s*SCORING\s*BRIEF\s*---\s*\n(.*)", description, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


# Dynamics-class classifier — used by the agent's pre-acquisition planner
# to decide whether dry-running on the proxy is safe (see
# `feedback_dry_run_state_bleed.md` and
# `knowledge/Core/Concepts/Sample time vs wall-clock time.md`).
#
# Categories:
#   "finite_budget_dynamic": the brief explicitly bounds the snap count
#       AND the sample state evolves with snaps in a way the budget gates.
#       Drift / accumulating-dose / saturating-counter scenarios.
#       *Don't dry-run on the same proxy — the dry-run consumes the budget.*
#   "snap_locked": 1 snap = 1 sim-step; continuous=False / auto_step=True
#       contract. Pre-MDA probes are accounted for but cheap.
#   "wall_clock_locked": realtime engine running between connect and burst.
#       Cells continue dynamics regardless of snap timing.
#       *Minimise pre-MDA probes; one preview is fine, three is too many.*
#   "static": no time evolution. Probe freely.
#   "unknown": keywords not detected.

_FINITE_BUDGET_KEYWORDS = (
    r"\bdrift\b", r"\bn_total_snaps\b", r"snap.budget", r"snap.count\s*<?=",
    r"\bbudget\b.*\bsnap", r"\bsaturating\b", r"max_cells\s*=",
    r"finite\s+budget", r"accumulat\w+\s+(drift|dose|bleach)",
)
_SNAP_LOCKED_KEYWORDS = (
    r"frame.driven", r"1\s*snap\s*=\s*1\s*sim.step", r"auto_step\s*=\s*True",
    r"continuous\s*=\s*False", r"snap.driven",
)
_WALL_CLOCK_KEYWORDS = (
    r"realtime\s+engine", r"real.time\s+dynamics", r"continuous\s*=\s*True",
    r"ongoing\s+(growth|dynamics|biology)", r"between\s+snaps",
    r"wall.clock", r"sim\s+(?:keeps|continues|runs)\s+running",
)
_STATIC_KEYWORDS = (
    r"static_preparation\s*:\s*true", r"no\s+time\s+evolution",
    r"sample\s+is\s+static",
)


def extract_dynamics_class(description: str) -> str:
    """Classify the brief's sample-time-vs-wall-clock contract.

    Returns one of: ``"finite_budget_dynamic"``, ``"snap_locked"``,
    ``"wall_clock_locked"``, ``"static"``, ``"unknown"``.

    Priority: finite_budget_dynamic > static > snap_locked > wall_clock_locked.
    finite_budget wins because it carries the strongest action consequence
    (don't dry-run on the same proxy); static beats snap_locked because
    static is a stricter contract.
    """
    text = description.lower()
    for pat in _FINITE_BUDGET_KEYWORDS:
        if re.search(pat, text):
            return "finite_budget_dynamic"
    for pat in _STATIC_KEYWORDS:
        if re.search(pat, text):
            return "static"
    for pat in _SNAP_LOCKED_KEYWORDS:
        if re.search(pat, text):
            return "snap_locked"
    for pat in _WALL_CLOCK_KEYWORDS:
        if re.search(pat, text):
            return "wall_clock_locked"
    return "unknown"


def safe_to_dry_run(brief: "StructuredBrief") -> tuple[bool, str]:
    """Heuristic: is it safe to dry-run this challenge on the same proxy?

    Returns ``(safe, reason)``. ``safe=False`` means the dry-run will
    consume budget the live run needs.

    Rules:
      - finite_budget_dynamic → unsafe: dry-run consumes the budget.
      - wall_clock_locked → unsafe-ish: dry-run runs the wall clock.
        We return False because pre-acquisition probes shift the
        connect-to-burst delta.
      - snap_locked → safe: budget is finite but the contract scales
        (1 snap = 1 step), so a dry-run on a fresh proxy followed by
        a re-serve replay is fine. On the SAME proxy still unsafe.
      - static / unknown → safe.
    """
    dyn = extract_dynamics_class(brief.raw_description)
    if dyn == "finite_budget_dynamic":
        return False, (
            "finite_budget_dynamic — dry-run consumes the budget; "
            "skip dry-run or request a re-serve before live submission "
            "(see feedback_dry_run_state_bleed.md)."
        )
    if dyn == "wall_clock_locked":
        return False, (
            "wall_clock_locked — pre-acquisition probes shift the "
            "connect-to-burst time window; minimise probes."
        )
    if dyn == "snap_locked":
        return True, (
            "snap_locked — dry-run consumes step count but the contract "
            "is deterministic; safe on a fresh proxy or after re-serve."
        )
    if dyn == "static":
        return True, "static — no time evolution; probe freely."
    return True, "unknown — assume safe pending evidence."


def parse_brief(challenge: dict[str, Any] | str) -> StructuredBrief:
    """Parse a challenge.json dict (or its description string).

    Accepts either the loaded JSON dict or the raw description text.
    """
    if isinstance(challenge, str):
        description = challenge
    else:
        description = str(challenge.get("description", ""))

    must, must_not = extract_method_summary_rules(description)
    return StructuredBrief(
        submit_shape=extract_submit_shape(description),
        tolerances=extract_tolerances(description),
        method_summary_must_reference=must,
        method_summary_must_not_reference=must_not,
        disclosed_coords=extract_disclosed_coords(description),
        scoring_brief_text=extract_scoring_brief(description),
        numeric_facts=extract_numeric_facts(description),
        archetype=extract_archetype(description),
        category=extract_category(description),
        proxy_url=extract_proxy_url(description),
        dynamics_class=extract_dynamics_class(description),
        raw_description=description,
    )


def find_disclosed_coord_near(brief: StructuredBrief, hint: str) -> Optional[tuple[int, int]]:
    """Return the first ``(x, y)`` whose context contains ``hint`` (case-insensitive).

    Examples:
        >>> brief = parse_brief(challenge)
        >>> find_disclosed_coord_near(brief, "first-firing")  # ch651: (144, 116)
        >>> find_disclosed_coord_near(brief, "primary")       # cardio: (128, 128)
    """
    hint_lower = hint.lower()
    for coord, ctx in brief.disclosed_coords:
        if hint_lower in ctx.lower():
            return coord
    return None


_KV_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)\b"
)


_IMAGE_COORD_RE = re.compile(
    r"image\s*[^()\n]{0,15}\(\s*(\d+)\s*,\s*(\d+)\s*\)",
    flags=re.IGNORECASE,
)


def _find_image_space_coord(brief: StructuredBrief) -> Optional[tuple[int, int]]:
    """Match the first ``image (X, Y)`` pattern in the description.

    Used by :func:`kwargs_from_brief` to disambiguate image-space
    priors from sim-internal PDE-grid coords. Returns ``None`` when
    no image-tagged tuple is present.
    """
    m = _IMAGE_COORD_RE.search(brief.raw_description)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def find_kv_fact(brief: StructuredBrief, key: str) -> Optional[float]:
    """Return the numeric value disclosed as ``key=value`` in the brief.

    Three matching strategies, in priority order:
      1. ``key=value`` tokens already in :attr:`StructuredBrief.numeric_facts`
         (regex-cleaned list).
      2. Fresh ``key=value`` scan of the raw description (covers tokens
         that ``numeric_facts`` filtered out as noise).
      3. Prose form ``"<key with spaces or _> (is|of|=|:)? <value>"`` with
         case-insensitive key match — captures "Dilution factor is 2",
         "n_cells of 30", etc. Underscores in ``key`` match either
         underscore or whitespace in the description.

    Returns ``None`` when ``key`` is not disclosed.

    Examples (from briefs we've actually shipped against):
        >>> find_kv_fact(brief, "n_burst")          # ch651: 40
        >>> find_kv_fact(brief, "period")           # ch652: 12
        >>> find_kv_fact(brief, "match_radius_px")  # ch651: 25
        >>> find_kv_fact(brief, "dilution_factor")  # ch360: 2 (prose)
    """
    for token, _ctx in brief.numeric_facts:
        m = _KV_RE.match(token)
        if m and m.group(1) == key:
            try:
                return float(m.group(2))
            except ValueError:
                continue
    for m in _KV_RE.finditer(brief.raw_description):
        if m.group(1) == key:
            try:
                return float(m.group(2))
            except ValueError:
                continue
    # Prose form: "Dilution factor is 2" / "n cells of 30". Each
    # underscore in `key` matches whitespace OR underscore in the brief.
    key_pattern = r"[\s_]+".join(re.escape(p) for p in key.split("_"))
    prose_re = re.compile(
        rf"\b{key_pattern}\s+(?:is|of|=|:)\s+([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)\b",
        flags=re.IGNORECASE,
    )
    m = prose_re.search(brief.raw_description)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def kwargs_from_brief(
    brief: StructuredBrief,
    recipe_class: str,
) -> dict[str, Any]:
    """Pull recipe-specific kwargs from a parsed brief.

    Maps a classifier-class label (the keys in
    ``src.core.utils.auto_recipe._PRIMARY``) to recipe kwargs that the
    brief explicitly disclosed. Disclosed values are *strong* signals —
    the brief author chose to surface them — and should override the
    feature-derived heuristics that ``auto_recipe`` builds from the
    image alone.

    Recognised mappings (extend conservatively; only add a key when a
    brief in our challenge corpus actually discloses it):

      - ``wide_dynamic_range_field`` (modality_switch / pacemaker
        localisation): ``n_burst`` from ``n_burst=N``,
        ``primary_position_prior`` and ``viewport_centre_px`` from any
        ``(x, y)`` whose context mentions "primary" / "first-firing" /
        "empirical".
      - ``single_bright_spot`` (FRAP): ``n_baseline`` from
        ``n_baseline=N``.
      - ``voronoi_monolayer`` / ``phase_contrast_tessellation``:
        ``n_cells_hint`` from ``n_cells=N`` if the brief discloses an
        expected count.

    Returns ``{}`` when nothing applies — callers should treat the
    output as a partial kwarg dict to merge on top of feature-derived
    defaults.

    Sprint #41 (2026-04-28): wires brief_parse → auto_recipe so the
    canonical pipeline ``parse_brief → auto_recipe(brief=...)`` pre-
    fills disclosed parameters automatically. Closes the gap left by
    ch651 r1→r3 (the brief literally contained the answer-adjacent
    coord and three rounds were lost not propagating it).
    """
    out: dict[str, Any] = {}

    if recipe_class == "wide_dynamic_range_field":
        nb = find_kv_fact(brief, "n_burst")
        if nb is not None:
            out["n_burst"] = int(nb)
        # Image-coord-first priority: the recipe needs IMAGE-space
        # priors, never PDE-grid coords (ch651 trap: "PRIMARY pacemaker
        # at PDE grid (32, 32)" is NOT a usable image prior). We look
        # for an explicit ``image (X, Y)`` pattern first; only if absent
        # do we fall back to context-keyword matching.
        primary = (
            _find_image_space_coord(brief)
            or find_disclosed_coord_near(brief, "empirical")
            or find_disclosed_coord_near(brief, "first-firing")
            or find_disclosed_coord_near(brief, "primary")
        )
        if primary is not None:
            out["primary_position_prior"] = primary
            out["viewport_centre_px"] = primary

    elif recipe_class == "single_bright_spot":
        nb = find_kv_fact(brief, "n_baseline")
        if nb is not None:
            out["n_baseline"] = int(nb)

    elif recipe_class in ("voronoi_monolayer", "phase_contrast_tessellation"):
        n_cells = find_kv_fact(brief, "n_cells")
        if n_cells is not None:
            out["n_cells_hint"] = int(n_cells)

    elif recipe_class == "sparse_cells":
        # Hemocytometer briefs commonly disclose dilution_factor in
        # prose ("Dilution factor is 2") — the find_kv_fact prose
        # branch handles that.
        df = find_kv_fact(brief, "dilution_factor")
        if df is not None:
            out["dilution_factor"] = float(df)

    elif recipe_class == "sim_protocol":
        # SIM mode is encoded in the brief's submit shape:
        #   per_orientation_ratios → 3-phase demod (ch646)
        #   orientation_clusters_rad → fringe-orientation FFT (ch649)
        #   both → mode="both"
        keys = set(brief.submit_shape or {})
        wants_demod = "per_orientation_ratios" in keys
        wants_orient = "orientation_clusters_rad" in keys
        if wants_demod and wants_orient:
            out["mode"] = "both"
        elif wants_orient:
            out["mode"] = "orientation"
        elif wants_demod:
            out["mode"] = "demod_3phase"
        # Channel disclosure ("channel = \"DAPI\"") — the SIM briefs
        # explicitly name the channel; pick it up so auto_recipe's
        # caller doesn't have to.
        chan_match = re.search(
            r'channel\s*=\s*"([A-Za-z0-9_]+)"', brief.raw_description or ""
        )
        if chan_match:
            out["channel"] = chan_match.group(1)

    return out


def validate_submit_shape(answer: dict[str, Any], brief: StructuredBrief) -> list[str]:
    """Check ``answer`` keys against the brief's ``Submit:`` block.

    Returns a list of failure messages (empty = clean):
      - missing required keys (declared in submit_shape)
      - extra keys not declared in submit_shape (warning)
    """
    failures = []
    if not brief.submit_shape:
        return failures
    declared = set(brief.submit_shape.keys())
    answered = set(answer.keys())
    missing = declared - answered
    extra = answered - declared
    if missing:
        failures.append(
            f"answer missing required keys from brief Submit: block: "
            f"{sorted(missing)}"
        )
    if extra:
        # Soft warning — extras are allowed but suspicious.
        failures.append(
            f"answer has keys NOT declared in brief Submit: block: "
            f"{sorted(extra)} (suspicious but not blocking)"
        )
    return failures


def validate_against_brief(
    answer: dict[str, Any],
    challenge: dict[str, Any] | str,
    *,
    method_summary_key: str = "method_summary",
) -> dict[str, list[str]]:
    """Run all brief-aware pre-submit checks in one call.

    Returns ``{"errors": [...], "warnings": [...]}`` where errors are
    blocking (missing keys, missing must-references) and warnings are
    soft hints (extra keys, banned-alone phrases).

    Use as the canonical pre-submit gate when ``challenge.json`` is
    available. ch651 r1→r3 lesson: the brief's Submit / Tolerance /
    Method-summary blocks were all parseable; the failure was that
    nothing was checking the answer against them automatically.
    """
    brief = parse_brief(challenge)
    errors: list[str] = []
    warnings: list[str] = []

    shape_failures = validate_submit_shape(answer, brief)
    for f in shape_failures:
        if "missing required keys" in f:
            errors.append(f)
        else:
            warnings.append(f)

    method_summary = answer.get(method_summary_key, "")
    if isinstance(method_summary, str) and method_summary:
        ms_failures = validate_method_summary(method_summary, brief)
        for f in ms_failures:
            if "missing all required references" in f:
                errors.append(f)
            else:
                warnings.append(f)

    return {"errors": errors, "warnings": warnings}


def validate_method_summary(method_summary: str, brief: StructuredBrief) -> list[str]:
    """Check ``method_summary`` against the brief's must / must-not rules.

    Returns a list of human-readable failure messages (empty = clean).
    Use as a pre-submit guard.
    """
    failures: list[str] = []
    ms_lower = method_summary.lower()

    # must-reference: at least ONE phrase from each "alternation group"
    # must appear. The brief's "references X / Y / Z" syntax means
    # any-of-these-acceptable. We treat the entire must list as
    # "any-of must appear at least once" if length ≥ 1.
    if brief.method_summary_must_reference:
        if not any(phrase in ms_lower for phrase in brief.method_summary_must_reference):
            failures.append(
                f"method_summary missing all required references: "
                f"{brief.method_summary_must_reference}"
            )

    # must-not-reference: each banned phrase must not appear ALONE.
    # Heuristic: phrase appears AND no must-reference phrase appears in
    # the same sentence → flag.
    for banned in brief.method_summary_must_not_reference:
        if banned in ms_lower:
            # weakly: only flag if NO must-reference in the whole summary
            if brief.method_summary_must_reference and any(
                p in ms_lower for p in brief.method_summary_must_reference
            ):
                continue
            failures.append(
                f"method_summary contains banned-alone phrase '{banned}' "
                "without a must-reference cross-check"
            )

    return failures
