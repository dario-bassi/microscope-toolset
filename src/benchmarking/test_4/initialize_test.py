"""Test 4: optogenetic — single-cell steering with SLM.

Same setup as test 3.  The experiment targets one cell with a directional
SLM mask to steer it across the field of view.

Channels:
  phase-contrast  — brightfield / label-free imaging   (CYAN + Electra1, hardcoded)
  stimulation     — optogenetic activation              (ORANGE + mScarlet3(569/582))

Simulation area: 1000x1000 µm, 50 optogenetic cells.
Per-cell radius drawn uniformly from [17, 22] µm.
Cells are nearly static (brownian_d ≈ 0); motion is driven by stimulation.
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
    "title": "Optogenetics — single-cell steering, 50 cells, 1000x1000 µm",
    "backend": "particle",
    "cell_type": "optogenetic",
    "n_cells": 50,
    "seed": 7,
    "width": 1000,
    "height": 1000,
    "radius_min": 17.0,   # µm
    "radius_max": 22.0,   # µm
    "brownian_d": 0.5,    # near-zero; motion driven by stimulation
    "time_scale": 5.0,    # 5x real-time so optogenetic migration is visible within seconds
    "tick_hz": 10,
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
