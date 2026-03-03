"""Closed-loop phototaxis steering via SLM.

Steer a phototactic organism to a target position using adaptive
SLM light placement. The light is placed AHEAD of the organism
in the direction of the target, causing it to turn and swim toward
the goal.

Usage:
    from src.workflows.phototaxis_steering import steering_generator, SteeringState

    state = SteeringState(target=(370, 170), lead_distance=50)

    def on_frame(img, event):
        cx, cy = detect_colony(img)
        state.update_position(cx, cy)

    events = steering_generator(state, channel='brightfield', group='Fake',
                                max_steps=150, threshold_px=30)
    results = run_events(core, events, on_frame=on_frame)

Real microscope note:
    This controller assumes INSTANT behavioral response to SLM light placement.
    Real organisms (C. elegans, Drosophila larvae, algae, etc.) have sensorimotor
    lag (100–300ms typical) between light stimulus and behavioral turn initiation.
    On real hardware:
    - Add response_lag parameter to predict organism position N frames ahead
    - Implement lag compensation: place light at predicted future position
    - Reduce lead_distance for organisms with slower phototaxis response
    - Use feedback from velocity history to auto-tune lead distance
    - Test on single organism before running population-level experiments
"""

import numpy as np
from useq import MDAEvent, SLMImage


class SteeringState:
    """Tracks phototaxis steering state."""

    def __init__(self, target, lead_distance=50, light_radius=40,
                 slm_size=512, slm_device='SLM'):
        """Initialize steering state.

        Parameters
        ----------
        target : tuple of (x, y)
            Target position in pixel coordinates (col, row).
        lead_distance : float
            How far ahead of organism to place light (pixels).
        light_radius : float
            Radius of light spot on SLM.
        slm_size : int
            SLM image size (square).
        slm_device : str
            SLM device name.
        """
        self.target_x, self.target_y = target
        self.lead_distance = lead_distance
        self.light_radius = light_radius
        self.slm_size = slm_size
        self.slm_device = slm_device

        self.positions = []
        self.distances = []
        self.reached = False
        self.last_img = None

    def update_position(self, x, y):
        """Update organism position and check if target reached."""
        self.positions.append((x, y))
        dist = np.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
        self.distances.append(dist)
        return dist

    @property
    def current_position(self):
        """Current organism position (x, y)."""
        if self.positions:
            return self.positions[-1]
        return None

    @property
    def current_distance(self):
        """Current distance to target."""
        if self.distances:
            return self.distances[-1]
        return float('inf')

    @property
    def velocity(self):
        """Estimated velocity vector from recent positions."""
        if len(self.positions) < 3:
            return (0, 0)
        recent = np.array(self.positions[-5:])
        if len(recent) < 2:
            return (0, 0)
        vx = float(np.mean(np.diff(recent[:, 0])))
        vy = float(np.mean(np.diff(recent[:, 1])))
        return (vx, vy)

    def compute_light_position(self):
        """Compute optimal SLM light position.

        Places light AHEAD of organism in the direction of the target.
        This is more efficient than placing light directly at the target
        because the angular correction is maximized when the light is
        close and in the right direction.

        Returns
        -------
        tuple of (light_x, light_y)
        """
        if not self.positions:
            return (self.target_x, self.target_y)

        cx, cy = self.positions[-1]
        dx = self.target_x - cx
        dy = self.target_y - cy
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1:
            return (self.target_x, self.target_y)

        # Normalize direction to target
        nx, ny = dx / dist, dy / dist

        # Lead distance: place light ahead of organism
        # As distance decreases, reduce lead to avoid overshoot
        effective_lead = min(self.lead_distance, dist * 0.8)

        light_x = cx + nx * effective_lead
        light_y = cy + ny * effective_lead

        # Clamp to SLM bounds
        light_x = max(0, min(self.slm_size - 1, light_x))
        light_y = max(0, min(self.slm_size - 1, light_y))

        return (light_x, light_y)

    def make_slm_mask(self):
        """Create SLM mask with light spot at computed position.

        Returns
        -------
        np.ndarray
            uint8 array of shape (slm_size, slm_size).
        """
        light_x, light_y = self.compute_light_position()
        mask = np.zeros((self.slm_size, self.slm_size), dtype=np.uint8)
        yy, xx = np.ogrid[:self.slm_size, :self.slm_size]
        circle = (yy - light_y)**2 + (xx - light_x)**2 <= self.light_radius**2
        mask[circle] = 255
        return mask

    @property
    def n_steps(self):
        """Number of steering steps completed."""
        return len(self.positions)


def steering_generator(state, channel='brightfield', group='Fake',
                       max_steps=150, threshold_px=30):
    """Generate MDA events for phototaxis steering.

    Yields MDAEvents with SLM masks that steer the organism toward
    the target. The generator stops when the target is reached or
    max_steps is exceeded.

    Parameters
    ----------
    state : SteeringState
        Shared steering state (updated by on_frame callback).
    channel : str
        Channel config name for BF imaging.
    group : str
        Config group name.
    max_steps : int
        Maximum number of steering steps.
    threshold_px : float
        Distance threshold to consider target reached.

    Yields
    ------
    MDAEvent
        Events with SLM images for phototaxis.
    """
    for step in range(max_steps):
        if state.reached:
            return

        # Check distance
        if state.current_distance < threshold_px:
            state.reached = True
            return

        slm_mask = state.make_slm_mask()
        yield MDAEvent(
            channel={"config": channel, "group": group},
            slm_image=SLMImage(data=slm_mask, device=state.slm_device),
        )


def make_tracking_callback(state, detector_fn, log_interval=10,
                           threshold_px=30):
    """Create an on_frame callback for steering.

    Parameters
    ----------
    state : SteeringState
        Shared steering state.
    detector_fn : callable
        Function(img) -> (x, y) that detects organism position.
    log_interval : int
        Print progress every N steps.
    threshold_px : float
        Distance threshold for target reached.

    Returns
    -------
    callable
        on_frame callback for run_events.
    """
    def on_frame(img, event):
        cx, cy = detector_fn(img)
        dist = state.update_position(cx, cy)
        state.last_img = img.copy()

        step = state.n_steps
        if step % log_interval == 0 or dist < threshold_px + 10:
            print(f'  Step {step}: pos=({cx:.0f},{cy:.0f}), '
                  f'dist={dist:.1f}px')

        if dist < threshold_px:
            state.reached = True
            print(f'  *** TARGET REACHED at step {step}! '
                  f'dist={dist:.1f}px ***')

    return on_frame


def clear_slm_event(channel='brightfield', group='Fake',
                    slm_device='SLM', size=512):
    """Create an event that clears the SLM.

    Returns
    -------
    MDAEvent
        Event with blank SLM mask.
    """
    return MDAEvent(
        channel={"config": channel, "group": group},
        slm_image=SLMImage(
            data=np.zeros((size, size), dtype=np.uint8),
            device=slm_device,
        ),
    )
