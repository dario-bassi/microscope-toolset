"""Showcase-aware submit wrapper.

NN rule 10 requires a showcase per submission. The orchestrator's
recurring "no showcase" nudge motivated this wrapper: build the PNG
*first*, then call ``comms.messaging.submit_solution``. Recipes can
no longer forget the showcase.

Sprint #16 (2026-04-26).

Functions:
    submit_with_showcase   -- main wrapper around submit_solution.
    count_panels           -- standard panel set for count submissions.
    kinetics_panels        -- standard panel set for kinetics / dose-response.
    closed_loop_panels     -- standard panel set for event-driven loops.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Optional


SHOWCASE_DIR = Path.home() / "sync/phd/code/self-learning/logs/showcase"


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not s:
        s = "submission"
    return s[:max_len]


def _default_save_to(challenge_id: int, method_description: str) -> Path:
    slug = _slugify(method_description)
    return SHOWCASE_DIR / f"agent_ch{int(challenge_id)}_{slug}.png"


def submit_with_showcase(
    *,
    challenge_id: int,
    answer: dict,
    method_description: str,
    code_used: str = "",
    panels: Optional[list] = None,
    title: Optional[str] = None,
    save_to: Optional[Path] = None,
    results_text: Optional[list[str]] = None,
    grid: Optional[tuple[int, int]] = None,
    skip_lint: bool = False,
    prior_code_samples: Optional[list[str]] = None,
) -> dict:
    """Build the showcase PNG, then forward to ``submit_solution``.

    Args:
        challenge_id, answer, method_description, code_used: forwarded
            verbatim to ``comms.messaging.submit_solution``.
        panels: list of ``src.core.utils.showcase.Panel`` to render
            into the showcase. If ``None``, a UserWarning is emitted
            and the submission proceeds without a showcase (back-compat).
        title: title for the showcase figure (defaults to a brief summary).
        save_to: optional override for the PNG path. Default:
            ``logs/showcase/agent_ch{N}_{slug}.png`` where slug is
            derived from ``method_description``.
        results_text: optional one-line summary lines printed under
            the figure (passed to ``make_showcase``).
        grid: optional explicit ``(rows, cols)`` for >6-panel figures.

    Returns:
        Whatever ``submit_solution`` returns (typically the path to
        the submission JSON).

    NN-rule lint: ``code_used`` is checked against
    :func:`src.core.utils.submission_lint.lint_submission` before the
    submission goes out. Hard errors (NN rule 1, truth.json reads)
    raise ``ValueError``. Soft violations (NN rule 4 inline-only,
    NN rule 7 duplicate solve) emit ``UserWarning``. Set
    ``skip_lint=True`` to bypass (pre-merge sanity tests, replays).
    Sprint #45 (2026-04-28).
    """
    from comms.messaging import submit_solution

    if not skip_lint:
        from .submission_lint import lint_submission_or_raise
        lint = lint_submission_or_raise(
            code_used,
            prior_code_samples=prior_code_samples or (),
        )
        for w in lint.warnings:
            warnings.warn(w, UserWarning, stacklevel=2)

    save_path: Optional[Path] = None
    if panels:
        from .showcase import make_showcase

        save_path = (
            Path(save_to) if save_to is not None
            else _default_save_to(challenge_id, method_description)
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        make_showcase(
            panels,
            title=title or f"ch{challenge_id} — {method_description[:80]}",
            save_to=save_path,
            grid=grid,
            results_text=results_text,
        )
    else:
        warnings.warn(
            "submit_with_showcase called without panels — submission "
            "will proceed without an NN-rule-10 showcase image. Pass "
            "panels=count_panels(...) / kinetics_panels(...) / "
            "closed_loop_panels(...) or a custom Panel list.",
            UserWarning,
            stacklevel=2,
        )

    result = submit_solution(
        challenge_id=challenge_id,
        answer=answer,
        method_description=method_description,
        code_used=code_used,
    )
    return {"submission_result": result, "showcase_path": save_path}


# ── Standard panel-set helpers ──────────────────────────────────────────


def count_panels(
    image: Any,
    centroids: list[tuple[float, float]],
    count: int,
    *,
    areas: Optional[list[float]] = None,
    title_prefix: str = "DAPI snap",
) -> list:
    """Standard 3-panel set for count-style submissions.

    Image + overlay, histogram of detection areas (if supplied) or a
    summary text trace, and a count summary.
    """
    from .showcase import Panel

    panels: list = [
        Panel(
            "image",
            title=f"{title_prefix} — {count} detections",
            data=image,
            cmap="gray",
            overlay_points=list(centroids),
            colorbar=False,
        ),
    ]
    if areas:
        panels.append(
            Panel(
                "histogram",
                title="Detection area distribution (px²)",
                data=list(areas),
                xlabel="area (px²)",
                ylabel="count",
                kwargs={"bins": 20},
            )
        )
    return panels


def kinetics_panels(
    t: Any,
    signal: Any,
    *,
    fit_t: Optional[Any] = None,
    fit_y: Optional[Any] = None,
    metric_value: Optional[float] = None,
    label: str = "signal",
    xlabel: str = "time (frames)",
    ylabel: str = "intensity",
) -> list:
    """Standard 1-2-panel set for kinetics / FRAP / dose-response."""
    from .showcase import Panel

    series_list = [{"x": list(t), "y": list(signal), "label": label,
                    "color": "tab:blue"}]
    if fit_t is not None and fit_y is not None:
        series_list.append({"x": list(fit_t), "y": list(fit_y), "label": "fit",
                             "color": "tab:red"})
    panel: dict = {"series": series_list}
    return [
        Panel(
            "trace",
            title=label,
            data=panel,
            xlabel=xlabel,
            ylabel=ylabel,
            vlines=[metric_value] if metric_value is not None else None,
        )
    ]


def closed_loop_panels(
    score_map: Any,
    primary_xy: tuple[float, float],
    sharpness_trace: list[float],
    *,
    sharpness_floor: Optional[float] = None,
) -> list:
    """Standard 2-panel set for event-driven modality-switch submissions."""
    from .showcase import Panel

    panels: list = [
        Panel(
            "heatmap",
            title="σ × mean score map (10x scout)",
            data=score_map,
            cmap="magma",
            colorbar=True,
            highlight=[(int(primary_xy[1]), int(primary_xy[0]))],
            xlabel="camera px",
            ylabel="camera px",
        ),
        Panel(
            "trace",
            title="100x detail sharpness over time",
            data={"x": list(range(len(sharpness_trace))),
                  "y": list(sharpness_trace),
                  "label": "Laplacian variance",
                  "color": "tab:blue"},
            xlabel="100x detail frame",
            ylabel="Laplacian variance",
            vlines=[sharpness_floor] if sharpness_floor is not None else None,
        ),
    ]
    return panels
