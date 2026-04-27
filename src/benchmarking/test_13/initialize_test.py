"""Test 13: optogenetic cell path-following — steer a cell along a defined shape.

The agent's goal is to guide one cell along a pre-determined geometric path
(e.g. spiral, square, circle) using SLM illumination. With only 15 cells in a
large 2000x2000 µm arena the target cell is easy to isolate and there is ample
space to trace multi-step trajectories without interference from neighbours.

Channels:
  phase-contrast  — brightfield / label-free imaging   (CYAN + Electra1, hardcoded)
  stimulation     — optogenetic activation              (ORANGE + mScarlet3(569/582))

Simulation area: 2000x2000 µm, 15 optogenetic cells.
Per-cell radius drawn uniformly from [17, 22] µm.
brownian_d = 0.5 (baseline thermal drift preserved).
Focal plane starts in focus (focal_plane = 0.0 µm).
"""

import numpy as np


def create_sim_override():
    from virtual_microscope.sims.cell.sim import ScatteredCellSim

    sim = ScatteredCellSim(
        width=TEST_CONFIG["width"],
        height=TEST_CONFIG["height"],
        n_cells=TEST_CONFIG["n_cells"],
        cell_type=TEST_CONFIG["cell_type"],
        seed=TEST_CONFIG["seed"],
    )

    rng = np.random.default_rng(TEST_CONFIG["seed"])
    radii_um = rng.uniform(
        TEST_CONFIG["radius_min"],
        TEST_CONFIG["radius_max"],
        size=TEST_CONFIG["n_cells"],
    )

    for i, (cell, r) in enumerate(zip(sim._cells, radii_um)):
        cell.base_r = float(r)
        cell.area0 = float(np.pi * r**2)
        cell.brownian_d = TEST_CONFIG["brownian_d"]
        sim.radii[i] = r

    sim.areas[:] = np.pi * radii_um**2

    return sim


TEST_CONFIG = {
    "title": "Optogenetics — cell path-following, 15 cells, 2000x2000 µm",
    "backend": "particle",
    "cell_type": "optogenetic",
    "n_cells": 15,
    "seed": 4579,
    "width": 2000,
    "height": 2000,
    "radius_min": 17.0,   # µm
    "radius_max": 22.0,   # µm
    "brownian_d": 0.5,
    "focal_plane": 0.0,
    "phase_contrast": True,
    "slm": True,
    "channels": [
        {
            "name": "stimulation",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
