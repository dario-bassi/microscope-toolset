"""Pre-submit GT-calibration preflight checks.

Detects three failure modes that have repeatedly cost the agent points
in challenges where the simulator's published GT measures a *different
quantity* than what the rendered pixels can recover:

  - **Shared-feature segmentation fusion** (ch599 Kaede / cadherin
    membranes): connected-component labels span multiple expected
    cells. No threshold tuning closes the gap; the GT records
    per-object sim state but the render fuses adjacent objects.

  - **Sub-pixel artefact fragmentation** (ch599 Voronoi vertex
    pixels): a sample is rendered with thousands of small bright
    artefacts that pass an under-tuned segmentation, inflating the
    count well past the density-implied population.

  - **Noise-floor degeneracy** (ch593 free-bottom IC50, ch348 FRAP
    half-life): a metric is dominated by render-stochasticity rather
    than the underlying physical quantity, so re-running on a fresh
    server gives wildly different values.

Each detector returns a structured ``{flag, message, details}`` dict
that the agent can paste into ``method_summary``. Recipes with a
flagged result should escalate to v-env for an apparent-GT
republication (the ch593-style fix) rather than submitting silently-
wrong numbers.

Designed to compose cleanly with ``skills/pre-submit-review.md`` —
add a step 5b that imports this module and runs the relevant
detectors against the agent's intermediate artefacts.

Sprint #13 (2026-04-26).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# ── Decision-rule defaults (tuned on ch593/ch599/ch348) ─────────────────
DEFAULT_MAX_PER_CELL_AREA_RATIO = 3.0      # blob > 3× expected → fusion
DEFAULT_MIN_PER_CELL_AREA_RATIO = 0.3      # blob < 0.3× expected → fragmentation
DEFAULT_FRAGMENT_COUNT_RATIO = 3.0         # CC count > 3× density-implied → fragmentation
DEFAULT_NOISE_CV_THRESHOLD = 0.30          # repeat-measurement CV → noise-floor flag
DEFAULT_DENSITY_AREA_PARITY_RANGE = (0.33, 3.0)


def detect_shared_feature_segmentation(
    labeled_mask: np.ndarray,
    *,
    expected_cell_area_px: float,
    max_per_cell_area_ratio: float = DEFAULT_MAX_PER_CELL_AREA_RATIO,
) -> dict:
    """Flag when CC labels carry > 1 expected cell each.

    Args:
        labeled_mask: 2-D int array. 0 = background; 1..N = instance
            IDs.
        expected_cell_area_px: per-cell area you'd see if cells were
            *not* fused (e.g. cell diameter² · π / 4).
        max_per_cell_area_ratio: a blob whose area exceeds
            ``ratio · expected_cell_area_px`` is flagged as a fusion.

    Returns dict with ``flag``, ``n_merged``, ``merged_label_ids``,
    ``message``, ``details``.
    """
    labels = np.asarray(labeled_mask, dtype=np.int64)
    if labels.size == 0 or labels.max() == 0:
        return {
            "flag": None,
            "n_merged": 0,
            "merged_label_ids": [],
            "message": "no labels",
            "details": {"max_label": 0},
        }

    n_max = int(labels.max())
    sizes = np.bincount(labels.ravel(), minlength=n_max + 1)
    threshold = float(max_per_cell_area_ratio) * float(expected_cell_area_px)
    merged_ids = [i for i in range(1, n_max + 1) if sizes[i] > threshold]

    if not merged_ids:
        return {
            "flag": None,
            "n_merged": 0,
            "merged_label_ids": [],
            "message": (
                f"no fusion detected — all blobs ≤ "
                f"{threshold:.0f} px ({max_per_cell_area_ratio}×"
                f"{expected_cell_area_px:.0f} expected)"
            ),
            "details": {"largest_blob_area": int(sizes[1:].max())},
        }
    return {
        "flag": "shared_membrane_fusion",
        "n_merged": len(merged_ids),
        "merged_label_ids": merged_ids,
        "message": (
            f"{len(merged_ids)} blob(s) larger than {threshold:.0f} px "
            f"({max_per_cell_area_ratio}× expected = {expected_cell_area_px:.0f} px). "
            f"GT may count per-object sim state while render fuses cells; "
            f"escalate to apparent-count grading or document the asymmetry."
        ),
        "details": {
            "expected_cell_area_px": float(expected_cell_area_px),
            "threshold_px": float(threshold),
            "merged_areas": [int(sizes[i]) for i in merged_ids],
            "largest_blob_area": int(sizes[1:].max()),
        },
    }


def detect_fragment_artifacts(
    labeled_mask: np.ndarray,
    *,
    expected_cell_area_px: float,
    expected_cell_count: Optional[int] = None,
    min_per_cell_area_ratio: float = DEFAULT_MIN_PER_CELL_AREA_RATIO,
    fragment_count_ratio: float = DEFAULT_FRAGMENT_COUNT_RATIO,
) -> dict:
    """Flag when CC labels are sub-pixel artefacts inflating the count.

    Two-criterion test: (a) CC count exceeds
    ``fragment_count_ratio · expected_cell_count`` (when caller supplies
    the density-implied count) AND (b) median label area is below
    ``min_per_cell_area_ratio · expected_cell_area_px``. Both must hold
    so a sparse sample with naturally large cells doesn't false-positive.
    """
    labels = np.asarray(labeled_mask, dtype=np.int64)
    n_max = int(labels.max()) if labels.size else 0
    if n_max == 0:
        return {"flag": None, "n_fragments": 0, "message": "no labels", "details": {}}

    sizes = np.bincount(labels.ravel(), minlength=n_max + 1)[1:]  # drop bg
    median_area = float(np.median(sizes))
    area_threshold = float(min_per_cell_area_ratio) * float(expected_cell_area_px)
    too_small = median_area < area_threshold

    too_many = False
    if expected_cell_count is not None:
        too_many = n_max > fragment_count_ratio * expected_cell_count

    if not (too_small and (too_many or expected_cell_count is None and n_max > 100)):
        return {
            "flag": None,
            "n_fragments": 0,
            "message": (
                f"no fragment artefacts — median blob area "
                f"{median_area:.0f} px (≥ {area_threshold:.0f} threshold)"
            ),
            "details": {"n_labels": n_max, "median_area_px": median_area},
        }

    return {
        "flag": "subpixel_artefact_fragmentation",
        "n_fragments": int(n_max),
        "message": (
            f"{n_max} labels with median area {median_area:.0f} px (< "
            f"{area_threshold:.0f} = {min_per_cell_area_ratio}× expected). "
            f"Likely sub-pixel rendering artefacts (Voronoi vertices, "
            f"speckle, vertex brightening); raise min_area or use blob-"
            f"detection with size-based filtering."
        ),
        "details": {
            "n_labels": int(n_max),
            "median_area_px": median_area,
            "expected_cell_area_px": float(expected_cell_area_px),
            "expected_cell_count": expected_cell_count,
        },
    }


def detect_apparent_gt_mismatch(
    metric_fn: Callable[[], float],
    *,
    n_samples: int = 4,
    cv_threshold: float = DEFAULT_NOISE_CV_THRESHOLD,
) -> dict:
    """Run ``metric_fn`` ``n_samples`` times; flag when the CV is high.

    A high coefficient of variation across re-renders of the same
    nominal sample means the metric is dominated by render
    stochasticity, not the underlying physical quantity. This is the
    free-bottom-IC50 / FRAP-degenerate-half-life class of failure.

    Args:
        metric_fn: zero-arg callable that re-acquires + re-measures.
            Caller is responsible for budget — typical use is 4
            quick re-snaps via existing scout machinery.
        n_samples: how many re-runs.
        cv_threshold: stddev / mean above this triggers the flag.

    Returns ``{flag, mean, std, cv, message, details}``.
    """
    samples = [float(metric_fn()) for _ in range(int(n_samples))]
    arr = np.asarray(samples, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = std / abs(mean) if mean != 0 else float("inf")
    flag = "noise_floor_degeneracy" if cv > cv_threshold else None
    msg = (
        f"metric stable across {n_samples} samples (cv={cv:.3f} ≤ {cv_threshold:.2f})"
        if flag is None
        else f"metric unstable (cv={cv:.3f} > {cv_threshold:.2f}) — render stochasticity dominates"
    )
    return {
        "flag": flag,
        "mean": mean,
        "std": std,
        "cv": cv,
        "samples": samples,
        "message": msg,
        "details": {"n_samples": int(n_samples), "cv_threshold": cv_threshold},
    }


def reconcile_count_with_density(
    n_detected: int,
    expected_n_from_density: int,
    *,
    parity_range: tuple[float, float] = DEFAULT_DENSITY_AREA_PARITY_RANGE,
) -> dict:
    """Compare a CC count against a density-implied count.

    ``parity_range`` defines the acceptable range of
    ``n_detected / expected``. Outside that range, returns a
    structured reason.
    """
    expected = max(int(expected_n_from_density), 1)
    ratio = float(n_detected) / float(expected)
    lo, hi = parity_range
    if lo <= ratio <= hi:
        return {
            "flag": None,
            "ratio": ratio,
            "message": f"count parity OK (n_detected/expected = {ratio:.2f} in [{lo}, {hi}])",
        }
    direction = "high" if ratio > hi else "low"
    return {
        "flag": "density_count_disagreement",
        "ratio": ratio,
        "message": (
            f"n_detected/expected = {ratio:.2f} outside [{lo}, {hi}] — "
            f"detected count is {direction} vs density estimate; "
            f"either segmentation is fused/fragmented or density estimate is wrong."
        ),
        "details": {
            "n_detected": int(n_detected),
            "expected": int(expected_n_from_density),
            "parity_range": parity_range,
        },
    }
