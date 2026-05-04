# TEST_CONFIG is injected from test.yaml by test_runner.py

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
