"""Matplotlib-native showcase-figure auto-generator.

Sprint #12 (2026-04-26 → 2026-04-27 rename). Replaces the per-recipe
ad-hoc matplotlib boilerplate with a single
``make_showcase(panels, title, save_to)`` call. Predecessor was an
OpenCV ``showcase.py`` (now retired); this matplotlib version was
shipped as ``showcase_mpl.py`` then promoted to the canonical name
in sprint #12 (rename + reconciliation).

API
---

::

    from src.core.utils.showcase import make_showcase, Panel

    panels = [
        Panel("heatmap", title="plate", data=plate_array,
              cmap="viridis", highlight=[(0, 0), (7, 11)], colorbar=True),
        Panel("trace", title="per-row controls",
              data={"x": rows, "y": pos_ctrl, "label": "pos"},
              ylabel="kRLU"),
        Panel("fit_curve", title="4-PL fit", xscale="log",
              data={"series": [{"x": doses, "y": viab, "yerr": sd,
                                 "xfit": xfit, "yfit": yfit,
                                 "label": "Compound 1"}]}),
    ]
    make_showcase(panels, "ch593 r4 — plate-reader IC50",
                  Path("../logs/showcase/agent_ch593_r4.png"))

The dispatcher resolves grid layout (1×1 ··· 2×3) automatically and
saves at 140 dpi. Power-users mutate the returned ``Figure`` before
or after save.

Built-in panel kinds:

- ``"image"``     — ``imshow`` + optional scatter overlay
- ``"heatmap"``   — ``imshow`` + ``Rectangle`` patches for highlights
                  + colorbar
- ``"trace"``     — ``plot`` time series with optional fit + events
- ``"histogram"`` — ``hist`` with optional vertical lines
- ``"fit_curve"`` — error-bar data + smooth fit + IC50/EC50 vline
- ``"rgb_overlay"`` — channel-stack RGB display

New kinds register via the ``@register("kind")`` decorator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


_GRIDS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
_RENDERERS: dict[str, Callable[[Any, "Panel"], None]] = {}


@dataclass
class Panel:
    """A single panel in a showcase figure.

    ``kind`` selects the renderer (``image``, ``heatmap``, ``trace``,
    ``histogram``, ``fit_curve``, ``rgb_overlay``, or anything
    registered via ``@register``). ``data`` is panel-kind-specific:
    a 2-D array for image/heatmap, a dict ``{"x", "y", ...}`` for
    trace/fit_curve, an array of values for histogram, etc.

    Well-known optional kwargs: ``title``, ``xlabel``, ``ylabel``,
    ``cmap``, ``vmin``, ``vmax``, ``xscale``, ``yscale``,
    ``colorbar``, ``highlight``, ``overlay_points``, ``vlines``.
    Anything else goes in ``kwargs`` and is forwarded to the
    renderer's primary matplotlib call.
    """
    kind: str
    title: str = ""
    data: Any = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xscale: Optional[str] = None
    yscale: Optional[str] = None
    colorbar: bool = False
    highlight: Optional[list[tuple[int, int]]] = None
    overlay_points: Optional[list[tuple[float, float]]] = None
    vlines: Optional[list[float]] = None
    xticks: Optional[list[Any]] = None
    yticks: Optional[list[Any]] = None
    legend: bool = True
    kwargs: dict = field(default_factory=dict)


def register(kind: str):
    """Decorator: register a renderer under ``kind``."""

    def _deco(fn: Callable[[Any, Panel], None]):
        _RENDERERS[kind] = fn
        return fn

    return _deco


def render_panel(ax, panel: Panel) -> None:
    """Dispatch to the panel's renderer."""
    if panel.kind not in _RENDERERS:
        raise KeyError(
            f"unknown panel kind {panel.kind!r}; "
            f"registered: {sorted(_RENDERERS)}"
        )
    _RENDERERS[panel.kind](ax, panel)


def _resolve_grid(n_panels: int, grid: Optional[tuple[int, int]]) -> tuple[int, int]:
    if grid is not None:
        return grid
    if n_panels not in _GRIDS:
        raise ValueError(
            f"{n_panels} panels has no default grid; "
            f"pass grid=(rows, cols) explicitly"
        )
    return _GRIDS[n_panels]


def make_showcase(
    panels: list[Panel],
    title: str,
    save_to: Path,
    *,
    grid: Optional[tuple[int, int]] = None,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 140,
    results_text: Optional[list[str]] = None,
):
    """Build a multi-panel showcase figure and save to ``save_to``.

    Returns the matplotlib ``Figure`` so callers can mutate it
    further before/after the save.
    """
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("panels list is empty")

    rows, cols = _resolve_grid(len(panels), grid)
    if figsize is None:
        figsize = (5.0 * cols, 4.0 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows * cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat) if hasattr(axes, "flat") else list(axes)

    for ax, panel in zip(axes_list, panels):
        render_panel(ax, panel)

    # Hide unused axes (when grid > n_panels, e.g. 5 panels in 2×3).
    for ax in axes_list[len(panels):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)
    if results_text:
        fig.text(
            0.5, 0.005, "  |  ".join(results_text),
            ha="center", fontsize=9, color="dimgray",
        )

    fig.tight_layout()
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=int(dpi), bbox_inches="tight")
    return fig


# ── Built-in renderers ─────────────────────────────────────────────────


@register("image")
def _render_image(ax, panel: Panel) -> None:
    arr = panel.data
    cmap = panel.cmap or "gray"
    im = ax.imshow(arr, cmap=cmap, vmin=panel.vmin, vmax=panel.vmax,
                   **panel.kwargs)
    if panel.overlay_points:
        xs = [p[0] for p in panel.overlay_points]
        ys = [p[1] for p in panel.overlay_points]
        ax.scatter(xs, ys, s=80, facecolor="none",
                   edgecolor="cyan", linewidth=1.5)
    if panel.colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(panel.title)
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)


@register("heatmap")
def _render_heatmap(ax, panel: Panel) -> None:
    from matplotlib.patches import Rectangle

    arr = panel.data
    cmap = panel.cmap or "viridis"
    im = ax.imshow(arr, cmap=cmap, vmin=panel.vmin, vmax=panel.vmax,
                   aspect=panel.kwargs.pop("aspect", "auto"),
                   **panel.kwargs)
    if panel.colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.04)
    if panel.highlight:
        for r, c in panel.highlight:
            ax.add_patch(Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                edgecolor="red", facecolor="none", linewidth=2.0,
            ))
    ax.set_title(panel.title)
    if panel.xticks is not None:
        ax.set_xticks(range(len(panel.xticks)))
        ax.set_xticklabels([str(x) for x in panel.xticks])
    if panel.yticks is not None:
        ax.set_yticks(range(len(panel.yticks)))
        ax.set_yticklabels([str(y) for y in panel.yticks])
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)


@register("trace")
def _render_trace(ax, panel: Panel) -> None:
    d = panel.data
    if "series" in d:
        for series in d["series"]:
            ax.plot(series["x"], series["y"],
                    label=series.get("label"),
                    color=series.get("color"))
    else:
        ax.plot(d["x"], d["y"], label=d.get("label"),
                color=d.get("color"))
        if "y2" in d:
            label2 = d.get("labels", [None, None])[1] if "labels" in d else d.get("label2")
            ax.plot(d["x"], d["y2"], label=label2)
        if "fit" in d:
            ax.plot(d["x"], d["fit"], "--", color="tab:red", label="fit")
    if panel.vlines:
        for v in panel.vlines:
            ax.axvline(v, color="gray", linestyle=":", alpha=0.6)
    if panel.xscale:
        ax.set_xscale(panel.xscale)
    if panel.yscale:
        ax.set_yscale(panel.yscale)
    ax.set_title(panel.title)
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    if panel.legend:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


@register("histogram")
def _render_histogram(ax, panel: Panel) -> None:
    bins = panel.kwargs.pop("bins", 30)
    ax.hist(panel.data, bins=bins, color="steelblue", **panel.kwargs)
    if panel.vlines:
        for v in panel.vlines:
            ax.axvline(v, color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_title(panel.title)
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel or "count")
    ax.grid(alpha=0.3)


@register("fit_curve")
def _render_fit_curve(ax, panel: Panel) -> None:
    d = panel.data
    series = d.get("series") or [d]
    for s in series:
        if "yerr" in s:
            ax.errorbar(s["x"], s["y"], yerr=s["yerr"], fmt="o",
                        markersize=6, label=s.get("label"),
                        color=s.get("color"))
        else:
            ax.plot(s["x"], s["y"], "o", label=s.get("label"),
                    color=s.get("color"))
        if "xfit" in s and "yfit" in s:
            ax.plot(s["xfit"], s["yfit"], lw=1.5, color=s.get("color"))
        if "vline" in s:
            ax.axvline(s["vline"], color=s.get("color"), ls=":", alpha=0.6)
    if panel.xscale:
        ax.set_xscale(panel.xscale)
    if panel.yscale:
        ax.set_yscale(panel.yscale)
    ax.set_title(panel.title)
    if panel.xlabel:
        ax.set_xlabel(panel.xlabel)
    if panel.ylabel:
        ax.set_ylabel(panel.ylabel)
    if panel.legend:
        ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


@register("rgb_overlay")
def _render_rgb_overlay(ax, panel: Panel) -> None:
    import numpy as np

    d = panel.data
    if isinstance(d, (list, tuple)) and len(d) == 3:
        r, g, b = d
        rgb = np.stack([
            r / max(r.max(), 1e-9),
            g / max(g.max(), 1e-9),
            b / max(b.max(), 1e-9),
        ], axis=-1)
    else:
        rgb = np.asarray(d)
    rgb = np.clip(rgb * panel.kwargs.pop("gain", 1.5), 0, 1)
    ax.imshow(rgb)
    if panel.overlay_points:
        xs = [p[0] for p in panel.overlay_points]
        ys = [p[1] for p in panel.overlay_points]
        ax.scatter(xs, ys, s=80, facecolor="none",
                   edgecolor="white", linewidth=1.5)
    ax.set_title(panel.title)
    ax.axis("off")
