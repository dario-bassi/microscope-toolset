"""Test 18: cell-cycle rare event detection — adaptive frame rate.

The agent must scan a large 2500x2500 µm field and, upon detecting a rare
event (a cell entering mitosis or apoptosis), switch to a higher frame rate
to capture the event in detail.

No cells start in M phase or apoptosis — both events emerge naturally from
cycle progression:
  - Mitosis:   any G2 cell completing its cycle will enter M (~24–27 sim-s);
               at time_scale=0.05 this takes ~3–9 real minutes depending on
               where in G2 the cell started.
  - Apoptosis: cells with n_div=8 or 9 will trigger apoptosis after 1–2 more
               completed cycles, adding a second class of rare event later in
               the session.

Cell distribution at start (all cells non-mitotic, non-dying):
  ~70 cells — n_div 0–5  (far from apoptosis)
  ~20 cells — n_div 6–7  (moderate)
  ~10 cells — n_div 8–9  (will enter apoptosis within a few cycles)

Frame-rate strategy (agent-side, not a simulation parameter):
  - Normal scan: slow interval (e.g. 5–10 s/tile) to cover the full area
  - Triggered burst: high frame rate (e.g. 0.5 s/frame) when rare event
    detected, to capture the full event progression

Channels:
  DAPI     — nucleus  — filter SCFP2(434/474)    + LED UV
  mScarlet — membrane — filter mScarlet3(569/582) + LED ORANGE

Simulation area: 5000×5000 µm, 100 static cell-cycle cells.
Full cycle = 9 real minutes (time_scale=0.05, 27 sim-s/cycle).
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

    # Assign n_div: ~70 low (0-5), ~20 moderate (6-7), ~10 high (8-9)
    order = rng.permutation(len(sim._cells))
    n_low, n_mid = 70, 20  # remaining 10 → n_div 8-9

    for rank, cell_idx in enumerate(order):
        cell = sim._cells[cell_idx]
        cell.is_dying = False
        cell.brownian_d = TEST_CONFIG["brownian_d"]

        if rank < n_low:
            cell.n_div = int(rng.integers(0, 6))      # 0-5
        elif rank < n_low + n_mid:
            cell.n_div = int(rng.integers(6, 8))      # 6-7
        else:
            cell.n_div = int(rng.integers(8, 10))     # 8-9

        # All cells start in G1/S/G2 — no M phase, no dying
        cell.cell_cycle_state = rng.choice(["G1", "S", "G2"],
                                           p=[0.50, 0.25, 0.25])
        cell.cell_mitosis_state = "Interphase"
        cell.current_time_life = cell._initial_random_time_life()

    return sim


TEST_CONFIG = {
    "title": "Rare event detection — adaptive frame rate, 100 cells, 5000x5000 µm",
    "backend": "particle",
    "cell_type": "cycle",
    "n_cells": 100,
    "seed": 89,
    "width": 2500,
    "height": 2500,
    "brownian_d": 0.0,
    "focal_plane": 0.0,
    "time_scale": 0.05,   # 9 min per full cycle (same as tests 8-10)
    "tick_hz": 10,
    "channels": [
        {
            "name": "DAPI",
            "filter": "SCFP2(434/474)",
            "led": "UV",
        },
        {
            "name": "mScarlet",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
