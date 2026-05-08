"""Two-pass scanning workflow for multi-scale cell analysis.

Implements the MDA-native two-pass pattern commonly used for:
  - Lysosomal puncta counting (10x survey → 40x per-cell puncta)
  - WBC differential counting (10x survey → 40x per-cell classification)
  - Stress fiber subtype analysis (10x overview → 40x per-cell Frangi)
  - Any workflow requiring a low-mag survey followed by high-mag analysis

Pattern:
  Pass 1 (Survey):  Wide FOV, detect object positions
  Pass 2 (Analysis): High-mag FOV, analyze each object in detail

Functions:
    build_position_events    -- Create MDA events from position list
    two_pass_mda             -- Full two-pass MDA acquisition pipeline
    survey_nuclei_pass       -- Standard 10x nucleus survey helper
    analysis_pass            -- Standard 40x per-cell analysis helper
"""

from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.feature import blob_log
from useq import MDAEvent

from ..hardware.core import run_events, set_objective, get_pixel_size


def build_position_events(
    positions: Sequence[Tuple[float, float]],
    channel_config: str,
    channel_group: str,
    exposure: float = 50.0,
    metadata: Optional[dict] = None,
) -> List[MDAEvent]:
    """Create MDAEvents for a list of stage positions.

    Args:
        positions: List of (x, y) stage coordinates in world units.
        channel_config: Channel configuration name.
        channel_group: Configuration group name (e.g., 'Fake').
        exposure: Exposure time in ms.
        metadata: Optional dict added to each event's metadata.

    Returns:
        List of MDAEvents, one per position.

    Example::

        events = build_position_events(
            [(100, 200), (300, 400)],
            channel_config='lyso-channel',
            channel_group='Fake',
        )
        run_events(core, events, on_frame=analyze_fn)
    """
    meta = metadata or {}
    return [
        MDAEvent(
            channel={'config': channel_config, 'group': channel_group},
            x_pos=float(x),
            y_pos=float(y),
            exposure=exposure,
            metadata={'position_idx': i, 'position_xy': (x, y), **meta},
        )
        for i, (x, y) in enumerate(positions)
    ]


def two_pass_mda(
    core,
    survey_channel: str,
    analysis_channel: str,
    group: str,
    survey_positions: Sequence[Tuple[float, float]],
    analysis_positions: Sequence[Tuple[float, float]],
    survey_callback: Optional[Callable] = None,
    analysis_callback: Optional[Callable] = None,
    survey_mag: int = 10,
    analysis_mag: int = 40,
    survey_exposure: float = 50.0,
    analysis_exposure: float = 100.0,
) -> Tuple[List, List]:
    """Execute a two-pass MDA acquisition: survey then per-object analysis.

    Pass 1 (Survey):
        Set survey magnification, acquire images at survey_positions,
        call survey_callback for each frame.

    Pass 2 (Analysis):
        Set analysis magnification, acquire images at analysis_positions,
        call analysis_callback for each frame.

    Both passes use run_events() — no snap() loops.

    Args:
        core: pymmcore-proxy or pymmcore-plus core instance.
        survey_channel: Channel config name for survey pass.
        analysis_channel: Channel config name for analysis pass.
        group: Config group name (e.g., 'Fake').
        survey_positions: List of (x, y) positions for survey.
        analysis_positions: List of (x, y) positions for analysis.
        survey_callback: fn(img, event) called for each survey frame.
        analysis_callback: fn(img, event) called for each analysis frame.
        survey_mag: Objective magnification for survey (default 10).
        analysis_mag: Objective magnification for analysis (default 40).
        survey_exposure: Exposure for survey pass (default 50 ms).
        analysis_exposure: Exposure for analysis pass (default 100 ms).

    Returns:
        Tuple of (survey_results, analysis_results).
        Each is a list of whatever the callbacks return (None if no callback).

    Example::

        survey_results = []
        analysis_results = []

        def on_survey(img, event):
            nuclei = detect_nuclei(img)
            survey_results.append(nuclei)

        def on_analysis(img, event):
            n_puncta = count_puncta(img)
            analysis_results.append(n_puncta)

        survey_r, analysis_r = two_pass_mda(
            core,
            survey_channel='nucleus-channel',
            analysis_channel='lyso-channel',
            group='Fake',
            survey_positions=[(256, 256)],
            analysis_positions=world_positions,
            survey_callback=on_survey,
            analysis_callback=on_analysis,
        )
    """
    # ── Pass 1: Survey ────────────────────────────────────────────────────────
    set_objective(core, survey_mag)
    survey_events = build_position_events(
        survey_positions, survey_channel, group, survey_exposure,
        metadata={'pass': 'survey', 'mag': survey_mag},
    )

    survey_frames = []

    def _survey_cb(img, event):
        result = survey_callback(img, event) if survey_callback else None
        survey_frames.append(result)

    run_events(core, survey_events, on_frame=_survey_cb)

    # ── Pass 2: Analysis ──────────────────────────────────────────────────────
    set_objective(core, analysis_mag)
    analysis_events = build_position_events(
        analysis_positions, analysis_channel, group, analysis_exposure,
        metadata={'pass': 'analysis', 'mag': analysis_mag},
    )

    analysis_frames = []

    def _analysis_cb(img, event):
        result = analysis_callback(img, event) if analysis_callback else None
        analysis_frames.append(result)

    run_events(core, analysis_events, on_frame=_analysis_cb)

    return survey_frames, analysis_frames


def detect_nuclei_log(
    gray: np.ndarray,
    stage_x: float,
    stage_y: float,
    pixel_size: float,
    min_sigma: float = 1.0,
    max_sigma: float = 8.0,
    log_threshold: float = 0.05,
    min_area: int = 30,
    max_area: int = 8000,
    edge_margin_px: int = 15,
) -> List[Tuple[float, float]]:
    """Detect nuclei using multi-scale Laplacian-of-Gaussian blob detection.

    More robust than Otsu threshold for varying-intensity nuclei. Uses LoG
    to find bright blobs at the expected nuclear scale range, then converts
    centroids to world coordinates.

    Args:
        gray: 2D float array. Nucleus channel image (brighter = more signal).
        stage_x: Stage X position (world units) for center of image.
        stage_y: Stage Y position (world units) for center of image.
        pixel_size: µm per pixel.
        min_sigma: Minimum blob sigma (min nucleus radius in pixels).
        max_sigma: Maximum blob sigma (max nucleus radius in pixels).
        log_threshold: LoG response threshold. Lower = more sensitive.
        min_area: Minimum nucleus area in pixels (reject small noise).
        max_area: Maximum nucleus area in pixels (reject large artifacts).
        edge_margin_px: Pixels from edge to exclude centroids.

    Returns:
        List of (world_x, world_y) for each detected nucleus.

    Notes:
        Use sigma values based on expected nucleus radius:
        - At 10x (1 µm/px): nuclei ~10-20 µm → sigma 5–10 px
        - At 40x (0.25 µm/px): nuclei ~10-20 µm → sigma 20–40 px
        For 10x typical nuclei: min_sigma=3, max_sigma=12 works well.
        For mixed sizes, use min_sigma=1, max_sigma=8 (default).
    """
    H, W = gray.shape

    # Normalize to [0, 1]
    g_min, g_max = gray.min(), gray.max()
    if g_max == g_min:
        return []
    gray_norm = (gray - g_min) / (g_max - g_min)

    blobs = blob_log(gray_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=8, threshold=log_threshold, overlap=0.3)

    positions = []
    for blob in blobs:
        cy, cx, sigma = blob
        cy, cx = float(cy), float(cx)
        area = int(np.pi * sigma ** 2)

        if area < min_area or area > max_area:
            continue
        if (cy < edge_margin_px or cy > H - edge_margin_px or
                cx < edge_margin_px or cx > W - edge_margin_px):
            continue

        world_x = stage_x + (cx - W / 2) * pixel_size
        world_y = stage_y + (cy - H / 2) * pixel_size
        positions.append((world_x, world_y))

    return positions


def survey_nuclei_pass(
    core,
    channel: str,
    group: str,
    survey_positions: Sequence[Tuple[float, float]],
    objective_mag: int = 10,
    min_nucleus_area: int = 80,
    max_nucleus_area: int = 8000,
    edge_margin_px: int = 15,
    use_log: bool = False,
    log_min_sigma: float = 1.0,
    log_max_sigma: float = 8.0,
    log_threshold: float = 0.05,
) -> List[Tuple[float, float]]:
    """Standard 10x nucleus survey: returns list of nucleus world positions.

    Scans survey_positions, segments DAPI/nucleus channel, converts
    pixel coordinates to world coordinates for the analysis pass.

    Args:
        core: pymmcore-proxy core instance.
        channel: Channel config name (DAPI/nucleus channel).
        group: Config group name.
        survey_positions: List of (x, y) stage positions to survey.
        objective_mag: Survey objective magnification.
        min_nucleus_area: Minimum nucleus area in pixels.
        max_nucleus_area: Maximum nucleus area in pixels.
        edge_margin_px: Pixels from edge to exclude nuclei. Default 15
            (smaller than 20 to avoid missing edge-adjacent cells).
        use_log: If True, use multi-scale LoG blob detection instead of
            Otsu threshold. More robust for varying intensities.
        log_min_sigma: Minimum LoG sigma (used when use_log=True).
        log_max_sigma: Maximum LoG sigma (used when use_log=True).
        log_threshold: LoG response threshold (used when use_log=True).

    Returns:
        List of (world_x, world_y) positions for each detected nucleus.
    """
    set_objective(core, objective_mag)
    ps = get_pixel_size(core)

    world_positions = []
    survey_events = build_position_events(survey_positions, channel, group)

    def on_survey(img, event):
        gray = img[:, :, 0].astype(float) if img.ndim == 3 else img.astype(float)
        H, W = gray.shape
        sx = event.x_pos if event.x_pos is not None else 0.0
        sy = event.y_pos if event.y_pos is not None else 0.0

        if use_log:
            positions = detect_nuclei_log(
                gray, sx, sy, ps,
                min_sigma=log_min_sigma, max_sigma=log_max_sigma,
                log_threshold=log_threshold,
                min_area=min_nucleus_area, max_area=max_nucleus_area,
                edge_margin_px=edge_margin_px,
            )
            world_positions.extend(positions)
        else:
            try:
                thresh = filters.threshold_otsu(gray)
            except Exception:
                thresh = gray.mean()

            nuc_mask = gray > thresh
            nuc_mask = morphology.remove_small_objects(nuc_mask, max_size=min_nucleus_area)
            nuc_mask = ndimage.binary_fill_holes(nuc_mask)

            labeled, _ = ndimage.label(nuc_mask)
            props = measure.regionprops(labeled)

            for p in props:
                cy, cx = p.centroid
                if p.area < min_nucleus_area or p.area > max_nucleus_area:
                    continue
                if (cy < edge_margin_px or cy > H - edge_margin_px or
                        cx < edge_margin_px or cx > W - edge_margin_px):
                    continue
                world_x = sx + (cx - W / 2) * ps
                world_y = sy + (cy - H / 2) * ps
                world_positions.append((world_x, world_y))

    run_events(core, survey_events, on_frame=on_survey)
    return world_positions


def analysis_pass(
    core,
    channel: str,
    group: str,
    analysis_positions: Sequence[Tuple[float, float]],
    analysis_fn: Callable,
    objective_mag: int = 40,
    exposure: float = 100.0,
) -> List[Any]:
    """Execute MDA analysis pass at each position using a custom callback.

    Args:
        core: pymmcore-proxy core instance.
        channel: Channel config for analysis.
        group: Config group name.
        analysis_positions: List of (x, y) world positions to image.
        analysis_fn: Callable(img, event) → result. Called per position.
        objective_mag: Objective magnification for analysis.
        exposure: Exposure time in ms.

    Returns:
        List of analysis_fn() return values, one per position.

    Example::

        from src.core.analysis.puncta import detect_puncta_log

        def count_puncta(img, event):
            from src.core.analysis.puncta import detect_puncta_log
            peaks, _ = detect_puncta_log(img)
            return len(peaks)

        puncta_counts = analysis_pass(
            core, 'lyso-channel', 'Fake', world_positions, count_puncta
        )
        mean_puncta = np.mean(puncta_counts)
    """
    set_objective(core, objective_mag)
    events = build_position_events(
        analysis_positions, channel, group, exposure,
        metadata={'pass': 'analysis', 'mag': objective_mag},
    )

    results = []

    def on_analysis(img, event):
        result = analysis_fn(img, event)
        results.append(result)

    run_events(core, events, on_frame=on_analysis)
    return results
