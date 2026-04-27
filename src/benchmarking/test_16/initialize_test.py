"""Test 16: C. elegans position tracking.

The agent's goal is to continuously track the worm's position by moving the
XY stage to keep it centred in the FOV. The task runs for ~10 minutes, but
the simulation itself is continuous and never stops — the worm will crawl
indefinitely until the session ends.

The worm moves at 40 px/s through a 2048×2048 world with sinusoidal body
bending, random direction changes every 3–8 s, and boundary bounces at the
arena margins. If the agent fails to follow, the worm exits the FOV.

Channels:
  DIC          — brightfield body outline  — filter Electra1(402/454) + LED CYAN
  GFP-pharynx  — pharynx fluorescence      — filter TagGFP2(483/506)  + LED GREEN
  mCherry-body — body-wall muscles         — filter mScarlet3(569/582) + LED ORANGE

time_scale=1.0 → simulation runs at real speed (1 sim-second = 1 real second).
"""


def create_sim_override():
    from virtual_microscope.backends.celegans import create_sim

    sim = create_sim(
        world_size=TEST_CONFIG["world_size"],
        worm_length=TEST_CONFIG["worm_length"],
        worm_width=TEST_CONFIG["worm_width"],
        speed=TEST_CONFIG["speed"],
        seed=TEST_CONFIG["seed"],
    )

    return sim


TEST_CONFIG = {
    "title": "C. elegans position tracking — 2048×2048 world, real-time",
    "backend": "celegans",
    "cell_type": "celegans",
    "world_size": 2048,
    "worm_length": 250.0,
    "worm_width": 18.0,
    "speed": 40.0,
    "seed": 14,
    "focal_plane": 0.0,
    "time_scale": 1.0,   # real-time: 1 sim-s = 1 real second
    "tick_hz": 10,
    "channels": [
        {
            "name": "DIC",
            "filter": "Electra1(402/454)",
            "led": "CYAN",
        },
        {
            "name": "GFP-pharynx",
            "filter": "TagGFP2(483/506)",
            "led": "GREEN",
        },
        {
            "name": "mCherry-body",
            "filter": "mScarlet3(569/582)",
            "led": "ORANGE",
        },
    ],
}
