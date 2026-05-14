"""Adaptive multi-position monitoring with event-triggered zoom.

Orchestrates the patrol -> detect -> zoom -> capture -> return loop:

1. Patrol: Cycle through N stage positions at survey magnification,
   snapping and running event detection at each position.
2. Detect: User-provided scorer evaluates each frame for
   "interestingness" (morphology change, fluorescence spike, etc.).
3. Zoom: When score exceeds threshold, switch to high magnification
   and run rapid timelapse at that position.
4. Return: After capture completes, switch back to survey
   magnification and resume patrol.
5. Report: All captured events with positions and measurements.

This is the core smart microscopy workflow: efficient allocation of
imaging time across space, driven by real-time analysis.

Functions:
    adaptive_monitor        -- Full patrol-detect-zoom-capture loop
    adaptive_monitor_events -- MDA-native generator version
    make_position_states    -- Initialize per-position tracking
    default_event_scorer    -- Simple intensity-change scorer
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from useq import MDAEvent

from ..hardware.core import run_events, set_objective


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Per-position tracking state across patrol rounds."""
    index: int
    xy: Tuple[float, float]
    scores: List[float] = field(default_factory=list)
    recent_means: List[float] = field(default_factory=list)
    n_visits: int = 0
    triggered: bool = False


@dataclass
class CapturedEvent:
    """Record of a captured event during monitoring."""
    position_index: int
    position_xy: Tuple[float, float]
    trigger_score: float
    trigger_round: int
    capture_frames: List[np.ndarray] = field(default_factory=list, repr=False)
    measurements: List[Any] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            'position_index': self.position_index,
            'position_xy': self.position_xy,
            'trigger_score': round(self.trigger_score, 3),
            'trigger_round': self.trigger_round,
            'n_capture_frames': len(self.capture_frames),
            'measurements': self.measurements,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_position_states(
    positions: Sequence[Tuple[float, float]],
) -> List[PositionState]:
    """Initialize per-position tracking state.

    Args:
        positions: List of (x, y) stage positions.

    Returns:
        List of PositionState, one per position.
    """
    return [PositionState(index=i, xy=pos) for i, pos in enumerate(positions)]


def default_event_scorer(
    image: np.ndarray,
    pos_state: PositionState,
) -> float:
    """Score based on fractional change in mean intensity.

    Returns absolute fractional change vs running baseline.
    Score > 0.2 means 20% intensity change. First visit returns 0.

    Useful for detecting calcium transients, fluorescence spikes,
    cell arrival/departure, or bleaching events.

    Args:
        image: Current frame at this position.
        pos_state: Position state with history from prior visits.

    Returns:
        Event score (0 = no change, higher = more interesting).
    """
    current_mean = float(image.mean())

    if len(pos_state.recent_means) == 0:
        return 0.0

    baseline = float(np.mean(pos_state.recent_means[-5:]))
    if baseline < 1.0:
        return 0.0

    return abs(current_mean - baseline) / baseline


def make_bright_region_scorer(
    threshold_percentile: float = 95.0,
    min_bright_fraction: float = 0.01,
) -> Callable[[np.ndarray, PositionState], float]:
    """Create a scorer that detects appearance of bright objects.

    Computes the fraction of pixels above a high percentile threshold.
    When bright objects (e.g., fluorescent cells, M-phase nuclei)
    appear, this fraction increases. Returns the change in bright
    fraction vs baseline.

    Args:
        threshold_percentile: Percentile to define "bright" (default 95).
        min_bright_fraction: Minimum bright fraction to report
            non-zero score (filters noise at empty positions).

    Returns:
        Scorer callable(image, PositionState) -> float.
    """
    bright_history: dict[int, list] = {}

    def scorer(image: np.ndarray, pos_state: PositionState) -> float:
        threshold = float(np.percentile(image, threshold_percentile))
        bright_frac = float(np.mean(image > threshold))

        idx = pos_state.index
        if idx not in bright_history:
            bright_history[idx] = []

        history = bright_history[idx]

        if len(history) == 0 or bright_frac < min_bright_fraction:
            history.append(bright_frac)
            if len(history) > 10:
                history.pop(0)
            return 0.0

        baseline = float(np.mean(history[-5:]))
        history.append(bright_frac)
        if len(history) > 10:
            history.pop(0)

        if baseline < 1e-6:
            return bright_frac * 100  # new signal from nothing

        return abs(bright_frac - baseline) / baseline

    return scorer


def make_count_change_scorer(
    detect_fn: Callable[[np.ndarray], int],
    min_change: int = 2,
) -> Callable[[np.ndarray, PositionState], float]:
    """Create a scorer that detects changes in object count.

    Useful for detecting cell division (count increases), cell death
    (count decreases), or cell migration (count changes).

    Args:
        detect_fn: Callable(image) -> int count of objects.
        min_change: Minimum absolute count change to report non-zero score.

    Returns:
        Scorer callable(image, PositionState) -> float.
    """
    count_history: dict[int, list] = {}

    def scorer(image: np.ndarray, pos_state: PositionState) -> float:
        count = detect_fn(image)

        idx = pos_state.index
        if idx not in count_history:
            count_history[idx] = []

        history = count_history[idx]

        if len(history) == 0:
            history.append(count)
            return 0.0

        baseline = float(np.mean(history[-5:]))
        change = abs(count - baseline)
        history.append(count)
        if len(history) > 10:
            history.pop(0)

        if change < min_change:
            return 0.0

        # Normalize by baseline count (avoid div by zero)
        if baseline < 1:
            return float(change)
        return float(change / baseline)

    return scorer


# ---------------------------------------------------------------------------
# Main imperative API
# ---------------------------------------------------------------------------

def adaptive_monitor(
    core,
    positions: Sequence[Tuple[float, float]],
    event_scorer: Callable[[np.ndarray, PositionState], float],
    score_threshold: float = 0.3,
    n_patrol_rounds: int = 10,
    capture_frames: int = 20,
    capture_interval: float = 1.0,
    survey_mag: int = 10,
    capture_mag: int = 40,
    survey_channel: Optional[str] = None,
    capture_channel: Optional[str] = None,
    channel_group: Optional[str] = None,
    survey_exposure: float = 50.0,
    capture_exposure: float = 50.0,
    max_captures: int = 5,
    on_capture_frame: Optional[Callable] = None,
    cooldown_rounds: int = 2,
) -> dict:
    """Run adaptive multi-position monitoring with event-triggered zoom.

    Patrols N positions at survey magnification. At each position,
    runs event_scorer to assess interestingness. When score exceeds
    threshold, switches to capture magnification and runs a rapid
    timelapse. After capture, returns to patrol.

    Args:
        core: Microscope core (CMMCorePlus or proxy).
        positions: List of (x, y) stage positions to patrol.
        event_scorer: Callable(image, PositionState) -> float.
            Higher score = more interesting. Called each patrol visit.
        score_threshold: Score above which to trigger capture.
        n_patrol_rounds: Number of complete patrol cycles.
        capture_frames: Frames per capture timelapse.
        capture_interval: Seconds between capture frames.
        survey_mag: Magnification for patrol (default 10).
        capture_mag: Magnification for event capture (default 40).
        survey_channel: Channel config for patrol.
        capture_channel: Channel for capture. Defaults to survey_channel.
        channel_group: Config group name (e.g. 'Fake').
        survey_exposure: Exposure for patrol (ms).
        capture_exposure: Exposure for capture (ms).
        max_captures: Stop after this many captures.
        on_capture_frame: Optional callback(image, event, captured_event)
            for real-time analysis during capture timelapse.
        cooldown_rounds: Rounds to skip a position after it triggers,
            preventing re-triggering on the same event.

    Returns:
        dict with:
            captured_events: List of CapturedEvent.
            position_states: List of PositionState.
            n_patrol_rounds_completed: Full rounds completed.
            n_captures: Number of events captured.
    """
    if capture_channel is None:
        capture_channel = survey_channel

    pos_states = make_position_states(positions)
    captured_events: List[CapturedEvent] = []
    rounds_completed = 0

    # Build channel kwargs for MDAEvent
    survey_ch = {}
    if survey_channel:
        ch = {'config': survey_channel}
        if channel_group:
            ch['group'] = channel_group
        survey_ch['channel'] = ch

    capture_ch = {}
    if capture_channel:
        ch = {'config': capture_channel}
        if channel_group:
            ch['group'] = channel_group
        capture_ch['channel'] = ch

    set_objective(core, survey_mag)

    for round_idx in range(n_patrol_rounds):
        if len(captured_events) >= max_captures:
            break

        for ps in pos_states:
            if len(captured_events) >= max_captures:
                break

            # Patrol: acquire one frame at this position
            patrol_event = MDAEvent(
                x_pos=ps.xy[0],
                y_pos=ps.xy[1],
                exposure=survey_exposure,
                **survey_ch,
            )
            results = run_events(core, [patrol_event])
            if not results:
                continue

            image, _ = results[0]
            current_mean = float(image.mean())

            # Score BEFORE updating state (scorer sees previous history)
            score = event_scorer(image, ps)

            # Update state
            ps.n_visits += 1
            ps.scores.append(score)
            ps.recent_means.append(current_mean)
            if len(ps.recent_means) > 10:
                ps.recent_means.pop(0)

            # Trigger?
            if score >= score_threshold and not ps.triggered:
                ps.triggered = True

                event_record = CapturedEvent(
                    position_index=ps.index,
                    position_xy=ps.xy,
                    trigger_score=score,
                    trigger_round=round_idx,
                )

                # Switch to capture magnification
                set_objective(core, capture_mag)

                # Rapid timelapse at this position
                capture_events = [
                    MDAEvent(
                        x_pos=ps.xy[0],
                        y_pos=ps.xy[1],
                        exposure=capture_exposure,
                        min_start_time=i * capture_interval,
                        **capture_ch,
                    )
                    for i in range(capture_frames)
                ]

                def _on_cap(img, evt, _rec=event_record):
                    _rec.capture_frames.append(img)
                    if on_capture_frame:
                        on_capture_frame(img, evt, _rec)

                run_events(core, capture_events, on_frame=_on_cap)
                captured_events.append(event_record)

                # Return to survey
                set_objective(core, survey_mag)

        rounds_completed = round_idx + 1

    return {
        'captured_events': captured_events,
        'position_states': pos_states,
        'n_patrol_rounds_completed': rounds_completed,
        'n_captures': len(captured_events),
    }


# ---------------------------------------------------------------------------
# MDA-native generator API
# ---------------------------------------------------------------------------

def adaptive_monitor_events(
    positions: Sequence[Tuple[float, float]],
    score_threshold: float = 0.3,
    n_patrol_rounds: int = 10,
    capture_frames: int = 20,
    capture_interval: float = 1.0,
    survey_mag: int = 10,
    capture_mag: int = 40,
    survey_channel: Optional[str] = None,
    capture_channel: Optional[str] = None,
    channel_group: Optional[str] = None,
    survey_exposure: float = 50.0,
    capture_exposure: float = 50.0,
    max_captures: int = 5,
):
    """MDA-native adaptive monitoring via generator + on_frame + state.

    Returns a (generator_fn, on_frame_callback, shared_state) tuple
    following the same pattern as adaptive_survey_mda and scan_and_detect_mda.

    The generator yields patrol events, checks scores via shared state
    between yields, and emits capture events when triggered. Because
    MDA engines call next(generator) after processing each frame,
    the on_frame callback has already updated state by the time the
    generator decides whether to trigger.

    Usage::

        gen, on_frame, state = adaptive_monitor_events(
            positions=[(100, 200), (300, 400), (500, 500)],
            score_threshold=0.3,
        )
        results = run_events(core, gen(), on_frame=on_frame)
        print(state['captured_events'])

    Args:
        positions: List of (x, y) stage positions to patrol.
        score_threshold: Score above which to trigger capture.
        n_patrol_rounds: Number of complete patrol cycles.
        capture_frames: Frames per capture timelapse.
        capture_interval: Seconds between capture frames.
        survey_mag: Magnification for patrol.
        capture_mag: Magnification for capture.
        survey_channel: Channel config for patrol.
        capture_channel: Channel for capture (defaults to survey_channel).
        channel_group: Config group name.
        survey_exposure: Exposure for patrol (ms).
        capture_exposure: Exposure for capture (ms).
        max_captures: Stop after this many captures.

    Returns:
        Tuple of (event_generator_fn, on_frame_callback, shared_state).
    """
    from useq import CustomAction

    if capture_channel is None:
        capture_channel = survey_channel

    pos_states = make_position_states(positions)

    state: dict[str, Any] = {
        'phase': 'patrol',
        'position_states': pos_states,
        'captured_events': [],
        'current_position_idx': -1,
        'last_score': 0.0,
        'last_image_mean': 0.0,
        'n_captures': 0,
    }

    survey_ch = {}
    if survey_channel:
        ch = {'config': survey_channel}
        if channel_group:
            ch['group'] = channel_group
        survey_ch['channel'] = ch

    capture_ch = {}
    cap_channel = capture_channel or survey_channel
    if cap_channel:
        ch = {'config': cap_channel}
        if channel_group:
            ch['group'] = channel_group
        capture_ch['channel'] = ch

    def event_generator():
        for round_idx in range(n_patrol_rounds):
            if state['n_captures'] >= max_captures:
                return

            for ps in pos_states:
                if state['n_captures'] >= max_captures:
                    return

                # Set current position for on_frame to know which state to update
                state['current_position_idx'] = ps.index
                state['phase'] = 'patrol'

                # Yield patrol event
                yield MDAEvent(
                    x_pos=ps.xy[0],
                    y_pos=ps.xy[1],
                    exposure=survey_exposure,
                    metadata={'phase': 'patrol', 'round': round_idx,
                              'position_idx': ps.index},
                    **survey_ch,
                )

                # After yield returns, on_frame has processed the image
                # and updated state['last_score']
                if (state['last_score'] >= score_threshold
                        and not ps.triggered):
                    ps.triggered = True

                    event_record = CapturedEvent(
                        position_index=ps.index,
                        position_xy=ps.xy,
                        trigger_score=state['last_score'],
                        trigger_round=round_idx,
                    )
                    state['captured_events'].append(event_record)
                    state['n_captures'] += 1

                    # Switch to capture mag
                    state['phase'] = 'capture'
                    yield MDAEvent(
                        action=CustomAction(
                            name='switch_objective',
                            data={'mag': capture_mag},
                        ),
                        metadata={'action': 'switch_objective',
                                  'mag': capture_mag},
                    )

                    # Rapid timelapse
                    for i in range(capture_frames):
                        yield MDAEvent(
                            x_pos=ps.xy[0],
                            y_pos=ps.xy[1],
                            exposure=capture_exposure,
                            min_start_time=i * capture_interval,
                            metadata={'phase': 'capture',
                                      'capture_frame': i,
                                      'position_idx': ps.index},
                            **capture_ch,
                        )

                    # Return to survey mag
                    state['phase'] = 'patrol'
                    yield MDAEvent(
                        action=CustomAction(
                            name='switch_objective',
                            data={'mag': survey_mag},
                        ),
                        metadata={'action': 'switch_objective',
                                  'mag': survey_mag},
                    )

    def on_frame(image, event):
        idx = state['current_position_idx']
        if idx < 0:
            return

        ps = pos_states[idx]
        current_mean = float(image.mean())

        if state['phase'] == 'patrol':
            score = default_event_scorer(image, ps)
            state['last_score'] = score
            state['last_image_mean'] = current_mean

            ps.n_visits += 1
            ps.scores.append(score)
            ps.recent_means.append(current_mean)
            if len(ps.recent_means) > 10:
                ps.recent_means.pop(0)

        elif state['phase'] == 'capture':
            # Store capture frame in the most recent event record
            if state['captured_events']:
                state['captured_events'][-1].capture_frames.append(image)

    return event_generator, on_frame, state
