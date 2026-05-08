"""Pluggable segmentation backends for `src.core.detection`.

The platform currently ships sigma-tuned thresholding under
`src.core.detection.cells`. The role doc is explicit that on a real
microscope you'd use Cellpose / StarDist instead — and the
`Sim-state vs rendered count asymmetry` pitfall (knowledge/core/
pitfalls/) showed exactly why: shared-feature segmentation can't be
recovered by threshold tuning when the rendering fuses object
boundaries.

This module provides:

  - ``Backend`` protocol — three methods: ``available()``, ``segment(image, ...)``, ``name``.
  - ``LabeledMask`` dataclass — narrow protocol that all backends return.
  - Built-in backends: ``SigmaBackend`` (wraps the existing thresholding),
    ``CellposeBackend`` (lazy import, no-op when cellpose isn't
    installed), ``StarDistBackend`` (placeholder stub).
  - ``segment(image, backend="auto", ...)`` dispatcher with a
    four-tier fallback ladder (auto → cellpose → roi-crop → sigma)
    and budget-aware skipping.
  - ``register_backend(name, backend)`` for new backends.

The default ``backend="auto"`` is **conservative**: it falls through
to ``"sigma"`` whenever Cellpose isn't available or the budget is
tight. Recipes can opt-in explicitly with ``backend="cellpose"``.

Sprint #10 (2026-04-26) plan-pass deliverable. Pilot rollout is on
``src.recipes.two_population_stain.detect_all_cells``; the existing
``detect_cells`` and ``count_*`` helpers remain on the sigma path
until validated.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np


logger = logging.getLogger(__name__)


# ── Protocols ───────────────────────────────────────────────────────────


@dataclass
class LabeledMask:
    """Backend-agnostic segmentation result.

    ``labels[r, c]`` is 0 for background and 1..N for instance IDs.
    Optional fields ``flow``, ``prob``, ``diameter_px`` carry
    backend-specific extras when available; consumers should treat
    them as optional.
    """
    labels: np.ndarray
    flow: Optional[np.ndarray] = None
    prob: Optional[np.ndarray] = None
    diameter_px: Optional[float] = None
    backend_used: str = ""
    inference_ms: float = 0.0
    was_cached: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_instances(self) -> int:
        return int(self.labels.max()) if self.labels.size else 0


class Backend(Protocol):
    """Segmentation-backend protocol."""

    name: str

    def available(self) -> bool:
        """Return True if the backend can run in this environment."""

    def segment(self, image: np.ndarray, **hints: Any) -> LabeledMask:
        """Segment ``image`` into a ``LabeledMask``."""


# ── Registry ────────────────────────────────────────────────────────────


_REGISTRY: dict[str, Backend] = {}


def register_backend(name: str, backend: Backend) -> None:
    """Register a backend under ``name`` (or replace if already present)."""
    _REGISTRY[name] = backend


def get_backend(name: str) -> Backend:
    """Look up a backend by name. ``KeyError`` if not registered."""
    if name not in _REGISTRY:
        raise KeyError(
            f"backend {name!r} not registered; "
            f"known backends: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def available_backends() -> list[str]:
    """Names of registered backends whose ``available()`` returns True."""
    return [n for n, b in _REGISTRY.items() if b.available()]


# ── Built-in backends ───────────────────────────────────────────────────


class SigmaBackend:
    """Wraps the existing sigma-thresholding pipeline as a Backend.

    Used as the universal fallback when Cellpose / StarDist aren't
    installed or the budget is tight. Computes labels via:

      mask = (image > image.mean() + sigma * image.std())
      labels = scipy.ndimage.label(mask)
    """

    name = "sigma"

    def __init__(self, sigma: float = 2.5, min_area: int = 10):
        self.sigma = float(sigma)
        self.min_area = int(min_area)

    def available(self) -> bool:
        return True

    def segment(self, image: np.ndarray, **hints: Any) -> LabeledMask:
        from scipy import ndimage

        t0 = time.time()
        img = np.asarray(image, dtype=float)
        sigma = float(hints.get("sigma", self.sigma))
        min_area = int(hints.get("min_area", self.min_area))
        thresh = img.mean() + sigma * img.std()
        mask = img > thresh
        labels, _ = ndimage.label(mask)
        if min_area > 1:
            sizes = np.bincount(labels.ravel())
            keep = np.where(sizes >= min_area)[0]
            keep_set = set(keep.tolist())
            keep_set.discard(0)
            relabel = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
            for new_id, old_id in enumerate(sorted(keep_set), start=1):
                relabel[old_id] = new_id
            labels = relabel[labels]
        return LabeledMask(
            labels=labels,
            backend_used=self.name,
            inference_ms=(time.time() - t0) * 1000.0,
        )


class CellposeBackend:
    """Lazy-loaded Cellpose backend.

    Imports `cellpose` only when ``available()`` or ``segment()`` is
    called for the first time, so the platform stays installable
    without the heavy dependency. ``available()`` returns False if
    the import fails.

    Uses `models.CellposeModel(model_type='cyto', gpu=False)` by
    default; override via the ``model_type`` / ``gpu`` constructor
    args.
    """

    name = "cellpose"

    def __init__(
        self,
        model_type: str = "cyto",
        gpu: bool = False,
        diameter_px: Optional[float] = None,
    ):
        self.model_type = model_type
        self.gpu = bool(gpu)
        self.diameter_px = diameter_px
        self._model = None
        self._import_failed: Optional[Exception] = None

    def _try_import(self):
        if self._model is not None or self._import_failed is not None:
            return
        try:
            cellpose_models = importlib.import_module("cellpose.models")
            self._model = cellpose_models.CellposeModel(
                model_type=self.model_type, gpu=self.gpu
            )
        except Exception as e:  # pragma: no cover — env-specific
            self._import_failed = e
            logger.info("CellposeBackend unavailable: %s", e)

    def available(self) -> bool:
        self._try_import()
        return self._model is not None

    def segment(self, image: np.ndarray, **hints: Any) -> LabeledMask:
        self._try_import()
        if self._model is None:
            raise RuntimeError(
                f"CellposeBackend not available: {self._import_failed}"
            )
        t0 = time.time()
        diameter = hints.get("diameter_px", self.diameter_px)
        masks, flows, _styles = self._model.eval(  # type: ignore[attr-defined]
            np.asarray(image), diameter=diameter, channels=[0, 0],
        )
        return LabeledMask(
            labels=np.asarray(masks, dtype=np.int32),
            flow=flows[0] if flows else None,
            diameter_px=float(diameter) if diameter else None,
            backend_used=self.name,
            inference_ms=(time.time() - t0) * 1000.0,
        )


class StarDistBackend:
    """Placeholder StarDist backend. Lazy-imports `stardist.models`."""

    name = "stardist"

    def __init__(self, model_name: str = "2D_versatile_fluo"):
        self.model_name = model_name
        self._model = None
        self._import_failed: Optional[Exception] = None

    def _try_import(self):
        if self._model is not None or self._import_failed is not None:
            return
        try:
            stardist_models = importlib.import_module("stardist.models")
            self._model = stardist_models.StarDist2D.from_pretrained(self.model_name)
        except Exception as e:  # pragma: no cover
            self._import_failed = e
            logger.info("StarDistBackend unavailable: %s", e)

    def available(self) -> bool:
        self._try_import()
        return self._model is not None

    def segment(self, image: np.ndarray, **hints: Any) -> LabeledMask:
        self._try_import()
        if self._model is None:
            raise RuntimeError(
                f"StarDistBackend not available: {self._import_failed}"
            )
        t0 = time.time()
        labels, _details = self._model.predict_instances(np.asarray(image))  # type: ignore[attr-defined]
        return LabeledMask(
            labels=np.asarray(labels, dtype=np.int32),
            backend_used=self.name,
            inference_ms=(time.time() - t0) * 1000.0,
        )


# Auto-register the built-ins.
register_backend("sigma", SigmaBackend())
register_backend("cellpose", CellposeBackend())
register_backend("stardist", StarDistBackend())


# ── Top-level dispatcher ────────────────────────────────────────────────


def segment(
    image: np.ndarray,
    *,
    backend: str = "auto",
    budget_s: Optional[float] = None,
    **hints: Any,
) -> LabeledMask:
    """Segment ``image`` with a backend selected by name or auto-fallback.

    Args:
        image: 2D numpy array.
        backend: ``"auto"`` (default) → preferred-order fallback;
            ``"sigma"``, ``"cellpose"``, ``"stardist"``, or any
            backend registered via ``register_backend``.
        budget_s: optional wall-clock budget. If set and a heavy
            backend's expected inference exceeds it, falls back to
            ``"sigma"`` and logs the decision via ``backend_used``.
        **hints: passed through to the backend's ``segment``
            (``sigma``, ``min_area``, ``diameter_px``, ...).

    Returns:
        ``LabeledMask`` with ``backend_used`` recording the actual
        backend that ran.
    """
    if backend == "auto":
        # Preference order: stardist (if available) → cellpose → sigma.
        # Budget gate: if budget_s is set and < 2 s, skip heavy backends.
        if budget_s is not None and budget_s < 2.0:
            chosen = "sigma"
        else:
            for name in ("stardist", "cellpose", "sigma"):
                if name in _REGISTRY and _REGISTRY[name].available():
                    chosen = name
                    break
            else:  # pragma: no cover — sigma is always available
                chosen = "sigma"
    else:
        chosen = backend
    return get_backend(chosen).segment(image, **hints)


# ── Utility: labels → centroid dicts (recipe adapter) ───────────────────


def labels_to_centroid_dicts(
    mask: LabeledMask,
    image: Optional[np.ndarray] = None,
) -> list[dict]:
    """Convert a ``LabeledMask`` to ``detect_cells``-style dicts.

    Each dict has ``centroid_px`` (cx, cy), ``area_px``, ``label_id``,
    plus ``mean_intensity`` / ``max_intensity`` when ``image`` is
    provided.
    """
    from scipy import ndimage

    labels = mask.labels
    n = int(labels.max())
    if n == 0:
        return []
    out = []
    centroids = ndimage.center_of_mass(np.ones_like(labels), labels, range(1, n + 1))
    for lid, (cy, cx) in enumerate(centroids, start=1):
        area = int((labels == lid).sum())
        d: dict = {
            "centroid_px": (float(cx), float(cy)),
            "area_px": area,
            "label_id": lid,
        }
        if image is not None:
            roi = image[labels == lid]
            d["mean_intensity"] = float(roi.mean()) if roi.size else 0.0
            d["max_intensity"] = float(roi.max()) if roi.size else 0.0
        out.append(d)
    return out
