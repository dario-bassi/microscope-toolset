#!/usr/bin/env python3
"""
Grid Scan to Find Positions with Cells — Template
===================================================

PROMPT: "Scan a grid of positions and find the one with the most cells."

Before running a stimulation or tracking experiment, you often need to find a
stage position that has enough cells (4-6+ for interesting interaction patterns).
This template scans a grid and reports cell counts at each position, then moves
to the best one.

USAGE:
    python samples/grid_scan_find_cells.py

After finding a good position, run one of the other experiment templates.
"""

import asyncio
import json
import numpy as np
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

MCP_URL = "http://127.0.0.1:5500/mcp"

# ===========================================================================
# PARAMETERS
# ===========================================================================
CENTER_X = 500.0    # um — center of scan grid
CENTER_Y = 500.0    # um
GRID_SIZE = 5       # NxN grid
STEP_UM = 500.0     # spacing between positions in um
MIN_CELLS = 4       # minimum cells to consider a position "good"

# ===========================================================================
# EXPERIMENT CODE — executed inside the microscope's Python environment
# ===========================================================================
EXPERIMENT_CODE = r'''
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def count_cells(img, threshold_sigma=1.5, min_area_px=30):
    """Quick cell count from a single image."""
    fimg = img.astype(np.float32)
    thresh = fimg.mean() + threshold_sigma * fimg.std()
    binary = (fimg > thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            count += 1
    return count


# Build grid positions
half = (GRID_SIZE - 1) / 2.0
positions = []
for gy in range(GRID_SIZE):
    for gx in range(GRID_SIZE):
        x = CENTER_X + (gx - half) * STEP_UM
        y = CENTER_Y + (gy - half) * STEP_UM
        positions.append((x, y))

print("Scanning %d positions (%dx%d grid, step=%.0f um)..." % (
    len(positions), GRID_SIZE, GRID_SIZE, STEP_UM))

results = []
stage_dev = mmc.getXYStageDevice()
for i, (x, y) in enumerate(positions):
    mmc.setXYPosition(x, y)
    mmc.waitForDevice(stage_dev)
    mmc.snapImage()
    img = mmc.getImage()
    n_cells = count_cells(img)
    results.append((x, y, n_cells))
    marker = "  <-- GOOD" if n_cells >= MIN_CELLS else ""
    print("  [%d/%d] (%.0f, %.0f): %d cells%s" % (
        i + 1, len(positions), x, y, n_cells, marker))

# Find best position
results.sort(key=lambda r: r[2], reverse=True)
best_x, best_y, best_n = results[0]
print("\nBest position: (%.0f, %.0f) with %d cells" % (best_x, best_y, best_n))

# Move to best position
mmc.setXYPosition(best_x, best_y)
mmc.waitForDevice(stage_dev)
print("Moved to best position.")

# Summary
good = [r for r in results if r[2] >= MIN_CELLS]
print("\n%d/%d positions have >= %d cells" % (len(good), len(results), MIN_CELLS))
print("DONE")
'''


def _build_params_preamble():
    """Build a Python code preamble that sets all parameter variables."""
    lines = [
        "CENTER_X = %r" % CENTER_X,
        "CENTER_Y = %r" % CENTER_Y,
        "GRID_SIZE = %r" % GRID_SIZE,
        "STEP_UM = %r" % STEP_UM,
        "MIN_CELLS = %r" % MIN_CELLS,
    ]
    return "\n".join(lines) + "\n"


async def main():
    code = _build_params_preamble() + EXPERIMENT_CODE

    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("Connected.")

            await session.call_tool("set_objective", {"label": "10x"})

            result = await session.call_tool("execute_python_code", {
                "code": code,
                "execution_mode": "live",
            })
            for b in result.content:
                if hasattr(b, "text"):
                    d = json.loads(b.text)
                    print(d.get("output", ""))
                    if d.get("error"):
                        print("ERROR:", d["error"])

            # Snap at best position and show in napari
            result = await session.call_tool("snap_image", {})
            print("Snapped at best position.")


if __name__ == "__main__":
    asyncio.run(main())
