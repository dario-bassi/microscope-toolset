"""Test 19: stage drift simulation — fluorescence XY drift correction.

The agent must detect and compensate for continuous XY stage drift.  The
stage drifts at 3.5 µm/min at 30° from the X axis; cells are static
(brownian_d=0.0).

The drift is invisible to the position readout — get_stage_position() returns
the commanded stage position, not the drifted position.  The agent must
infer drift by detecting displacement of cells across sequential frames.

Suggested agent strategy:
  1. Acquire a reference frame at t=0 and note cell positions
  2. Acquire frames at regular intervals (e.g. every 30 s)
  3. Register each frame against the reference (cross-correlation or
     centroid matching) to measure displacement
  4. Compute drift rate (µm/min) and direction from accumulated offsets
  5. Compensate by commanding the stage to counteract the measured drift

Drift parameters:
  Rate:  3.5 µm/min  (≈ 0.058 µm/s)
  Angle: 30° from X axis  (dx = rate·cos 30°, dy = rate·sin 30°)

After 10 minutes the total accumulated drift is ~35 µm — visible as ~7%
of the 512 µm FOV at 10× objective.

Channels:
  DAPI    — nucleus  — filter SCFP2(434/474)    + LED UV
  mCherry — membrane — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 2000×2000 µm, 80 static cells.
"""

import math
import time

import numpy as np


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim

    _rate_um_per_sec = TEST_CONFIG["drift_rate_um_per_min"] / 60.0
    _angle_rad = math.radians(TEST_CONFIG["drift_angle_deg"])

    class DriftingCellSim(ScatteredCellSim):
        """ScatteredCellSim with time-based XY stage drift applied per snap."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._drift_last_time = time.monotonic()

        def _crop_fov(self, full):
            now = time.monotonic()
            dt = now - self._drift_last_time
            self._drift_last_time = now
            self._drift_accumulator[0] += _rate_um_per_sec * dt * math.cos(_angle_rad)
            self._drift_accumulator[1] += _rate_um_per_sec * dt * math.sin(_angle_rad)
            return super()._crop_fov(full)

        def get_stage_drift(self) -> tuple:
            """Return accumulated drift (dx, dy) in µm."""
            return (round(float(self._drift_accumulator[0]), 3),
                    round(float(self._drift_accumulator[1]), 3))

        def reset_stage_drift(self) -> None:
            self._drift_accumulator[:] = 0.0

    sim = DriftingCellSim(
        width=TEST_CONFIG["width"],
        height=TEST_CONFIG["height"],
        n_cells=TEST_CONFIG["n_cells"],
        cell_type=TEST_CONFIG["cell_type"],
        seed=TEST_CONFIG["seed"],
    )

    for cell in sim._cells:
        cell.brownian_d = TEST_CONFIG["brownian_d"]

    return sim


TEST_CONFIG = {
    "title": "Stage drift correction — 3.5 µm/min XY drift, 80 static cells, 2000×2000 µm",
    "backend": "particle",
    "cell_type": "normal",
    "n_cells": 80,
    "seed": 42,
    "width": 2000,
    "height": 2000,
    "brownian_d": 0.0,
    "drift_rate_um_per_min": 3.5,
    "drift_angle_deg": 30.0,
    "focal_plane": 0.0,
    "channels": [
        {
            "name": "DAPI",
            "filter": "SCFP2(434/474)",
            "led": "UV",
        },
        {
            "name": "mCherry",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
