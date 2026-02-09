#!/usr/bin/env python3
"""
Timelapse Acquisition with Cell Tracking — Template
=====================================================

PROMPT: "Acquire a timelapse of cells and track their movement over time."

This template covers basic timelapse acquisition WITHOUT stimulation. It provides:
    - MDA-based timelapse with configurable frames, exposure, and interval
    - Per-frame cell segmentation and centroid extraction
    - Nearest-neighbor cell tracking across frames
    - Displacement and motility analysis
    - Saving timelapse TIFF + tracks for napari visualization

Use this as the starting point for any observation-only experiment, or as the
foundation to add stimulation on top (see directional_stimulation.py).

USAGE:
    python samples/timelapse_with_tracking.py
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
N_FRAMES = 200
EXPOSURE = 50.0
INTERVAL = 0.1              # seconds between frames
THRESHOLD_SIGMA = 1.5
MIN_AREA_PX = 30
OUTPUT_PREFIX = "/tmp/timelapse"

# ===========================================================================
# EXPERIMENT CODE — executed inside the microscope's Python environment
# ===========================================================================
EXPERIMENT_CODE = r'''
import numpy as np
import tifffile
import time
import cv2
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist
from useq import MDAEvent


def segment_cells(img, threshold_sigma=THRESHOLD_SIGMA, min_area_px=MIN_AREA_PX):
    """Segment bright cells via adaptive thresholding + connected components."""
    fimg = img.astype(np.float32)
    thresh = fimg.mean() + threshold_sigma * fimg.std()
    binary = (fimg > thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cells = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        mask = binary_fill_holes(labels == i)
        labels[mask] = i
        cx, cy = centroids[i]
        cells.append({
            "label": i,
            "centroid_px": (float(cx), float(cy)),
            "area_px": int(mask.sum()),
        })
    cells.sort(key=lambda c: c["area_px"], reverse=True)
    return labels, cells


def track_cells(all_cells, max_dist=50.0):
    """Nearest-neighbor tracking. Returns dict mapping track_id to [(t, x, y)]."""
    tracks = {}
    next_id = 0
    prev_c = None
    prev_ids = None
    for t_idx, frame_cells in enumerate(all_cells):
        if not frame_cells:
            prev_c = None
            prev_ids = None
            continue
        curr_c = np.array([c["centroid_px"] for c in frame_cells])
        if prev_c is None or prev_ids is None:
            curr_ids = []
            for i in range(len(curr_c)):
                tracks[next_id] = [(t_idx, curr_c[i][0], curr_c[i][1])]
                curr_ids.append(next_id)
                next_id += 1
        else:
            dists = cdist(curr_c, prev_c)
            curr_ids = [None] * len(curr_c)
            used = set()
            for flat in np.argsort(dists, axis=None):
                i, j = divmod(int(flat), dists.shape[1])
                if curr_ids[i] is not None or j in used:
                    continue
                if dists[i, j] > max_dist:
                    break
                curr_ids[i] = prev_ids[j]
                tracks[prev_ids[j]].append((t_idx, curr_c[i][0], curr_c[i][1]))
                used.add(j)
            for i in range(len(curr_ids)):
                if curr_ids[i] is None:
                    tracks[next_id] = [(t_idx, curr_c[i][0], curr_c[i][1])]
                    curr_ids[i] = next_id
                    next_id += 1
        prev_c = curr_c
        prev_ids = curr_ids
    return tracks


# ---- acquisition ----
print("Snapping initial frame...")
mmc.snapImage()
img0 = mmc.getImage()
_, cells0 = segment_cells(img0)
print("Detected %d cells" % len(cells0))

timelapse = []
all_cells = []

def on_frame(img, event, meta):
    idx = len(timelapse)
    timelapse.append(img.copy())
    _, cells = segment_cells(img)
    all_cells.append(cells)
    if (idx + 1) % 50 == 0:
        print("  Frame %d/%d: %d cells" % (idx + 1, N_FRAMES, len(cells)))

events = [MDAEvent(exposure=EXPOSURE, min_start_time=i * INTERVAL,
                   index={"t": i}) for i in range(N_FRAMES)]
print("Starting MDA: %d frames..." % N_FRAMES)
t0 = time.time()
run_mda_with_feedback(events, on_frame=on_frame)
elapsed = time.time() - t0
print("MDA done: %d frames in %.1fs" % (len(timelapse), elapsed))

# ---- tracking + analysis ----
tracks_dict = track_cells(all_cells)
long = {tid: pts for tid, pts in tracks_dict.items() if len(pts) >= 10}
print("\nTracks: %d total, %d with >=10 pts" % (len(tracks_dict), len(long)))

print("\nDisplacement (tracks with 50+ pts):")
for tid, pts in sorted(long.items()):
    if len(pts) < 50:
        continue
    dx = pts[-1][1] - pts[0][1]
    dy = pts[-1][2] - pts[0][2]
    total = np.sqrt(dx**2 + dy**2)
    # Path length (total distance traveled, not just displacement)
    path = sum(np.sqrt((pts[i+1][1]-pts[i][1])**2 + (pts[i+1][2]-pts[i][2])**2)
               for i in range(len(pts)-1))
    print("  Track %d: %d pts, displacement=%.1f px, path=%.1f px, ratio=%.2f" % (
        tid, len(pts), total, path, total / (path + 1e-6)))

# ---- save ----
rows = []
for tid, pts in long.items():
    for (t_idx, cx, cy) in pts:
        rows.append([tid, t_idx, cy, cx])
tracks_arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4),
                                                                     dtype=np.float64)

stack = np.array(timelapse, dtype=np.uint8)
tifffile.imwrite(OUTPUT_PREFIX + "_timelapse.tif", stack)
np.save(OUTPUT_PREFIX + "_tracks.npy", tracks_arr)
print("\nSaved: %s_timelapse.tif, %s_tracks.npy" % (OUTPUT_PREFIX, OUTPUT_PREFIX))
print("DONE")
'''


def _build_params_preamble():
    """Build a Python code preamble that sets all parameter variables."""
    lines = [
        "N_FRAMES = %r" % N_FRAMES,
        "EXPOSURE = %r" % EXPOSURE,
        "INTERVAL = %r" % INTERVAL,
        "THRESHOLD_SIGMA = %r" % THRESHOLD_SIGMA,
        "MIN_AREA_PX = %r" % MIN_AREA_PX,
        "OUTPUT_PREFIX = %r" % OUTPUT_PREFIX,
    ]
    return "\n".join(lines) + "\n"


async def main():
    code = _build_params_preamble() + EXPERIMENT_CODE

    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("Connected.")

            await session.call_tool("set_objective", {"label": "10x"})
            await session.call_tool("move_stage", {
                "x": 200.0, "y": 1000.0, "relative": False,
            })

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

            await session.call_tool("viewer_add_image", {
                "path": OUTPUT_PREFIX + "_timelapse.tif",
                "name": "timelapse",
            })
            tracks = np.load(OUTPUT_PREFIX + "_tracks.npy")
            if tracks.shape[0] > 0:
                await session.call_tool("viewer_add_tracks", {
                    "track_data": tracks.tolist(),
                    "tail_width": "3",
                    "tail_length": "50",
                })
            print("All layers added to napari.")


if __name__ == "__main__":
    asyncio.run(main())
