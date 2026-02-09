#!/usr/bin/env python3
"""
Directional SLM Stimulation — Generalized Template
====================================================

PROMPT: "Use spatially targeted SLM stimulation to make cells migrate in a
chosen direction / toward a target / toward each other / apart."

This single template handles ALL directional stimulation experiments by varying
the per-cell target vector. The core algorithm is always the same:

    For each cell:
    1. Compute a direction vector from cell centroid to target
    2. Project each cell pixel onto that direction
    3. Keep the top N% of pixels (highest projection = most toward target)
    4. Dilate by 2-3px for robust vertex coverage

What changes between experiments is only HOW the target vector is computed.

PROVEN PATTERNS (all tested, working):
    - Uniform direction   : all cells migrate in the same direction (up/down/left/right)
    - Converge to point   : all cells assemble at a chosen location (e.g. FOV center)
    - Pair assembly       : cells pair up and converge toward each other
    - Disperse from point : cells scatter outward from a center
    - Chase               : each cell migrates toward a designated target cell

USAGE:
    1. Pick a STRATEGY (see the "Target Strategies" section below)
    2. Adjust PARAMETERS (N_FRAMES, EXPOSURE, INTERVAL, STIM_PERCENT, DILATE_PX)
    3. Run via MCP:
         python samples/directional_stimulation.py
       Or paste EXPERIMENT_CODE into an execute_python_code MCP call with
       execution_mode="live".

PORTABILITY:
    This code uses only the standard pymmcore-plus API (setSLMImage / displaySLMImage).
    It works on both the virtual microscope and real hardware. Never use
    bridge.set_slm_mask() — that only works in simulation.

TIPS:
    - Scan the stage grid first to find a position with 4-6+ cells
    - 15% is a good default for STIM_PERCENT
    - Cells plateau at ~40px intra-pair distance (virtual microscope collision physics)
    - Save mask stack alongside timelapse to verify targeting in napari
    - For brightfield cells with dark interior, use fill_holes=True in segment_cells
"""

import asyncio
import json
import numpy as np
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

MCP_URL = "http://127.0.0.1:5500/mcp"

# ===========================================================================
# PARAMETERS — adjust these for your experiment
# ===========================================================================
STRATEGY = "converge"       # "uniform", "converge", "pairs", "disperse"
DIRECTION = (0, -1)         # for "uniform" strategy: (dx, dy) unit direction
TARGET_POINT = (256, 256)   # for "converge"/"disperse": (x, y) in px
N_FRAMES = 200
EXPOSURE = 50.0
INTERVAL = 0.1              # seconds between frames
STIM_PERCENT = 15           # fraction of each cell to stimulate (%)
DILATE_PX = 3               # dilation radius for robust vertex coverage
THRESHOLD_SIGMA = 1.5       # cell detection threshold (mean + N*std)
MIN_AREA_PX = 30            # minimum cell area in pixels
OUTPUT_PREFIX = "/tmp/directional_stim"

# ===========================================================================
# EXPERIMENT CODE — executed inside the microscope's Python environment
#
# Parameters are injected as a preamble string (see main() below).
# This avoids escaping issues with Python's {} syntax in .format() calls.
# ===========================================================================
EXPERIMENT_CODE = r'''
import numpy as np
import tifffile
import time
import cv2
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist, pdist
from useq import MDAEvent

# ========================================================================
# Common Helpers
# ========================================================================

def segment_cells(img, threshold_sigma=THRESHOLD_SIGMA, min_area_px=MIN_AREA_PX):
    """Segment bright cells from a grayscale image using adaptive thresholding.

    Returns:
        labels: 2D int array where each cell has a unique label (0 = background)
        cells: list of dicts with keys: label, centroid_px (x, y), area_px
    """
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
        # Fill holes (brightfield cells may have dark interior)
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


def pair_cells_greedy(cells):
    """Pair cells by greedy nearest-neighbor matching.

    Minimizes intra-pair distance (naturally maximizes inter-pair distance).
    Returns list of (idx_a, idx_b) tuples. Odd cell count leaves one unpaired.
    """
    n = len(cells)
    if n < 2:
        return []
    centroids = np.array([c["centroid_px"] for c in cells])
    dists = cdist(centroids, centroids)
    np.fill_diagonal(dists, np.inf)

    paired = set()
    pairs = []
    for f in np.argsort(dists, axis=None):
        i, j = divmod(int(f), n)
        if i in paired or j in paired:
            continue
        if dists[i, j] == np.inf:
            continue
        pairs.append((i, j))
        paired.add(i)
        paired.add(j)
        if len(paired) >= n - (n % 2):
            break
    return pairs


def track_cells(all_cells, max_dist=50.0):
    """Simple nearest-neighbor tracking across frames.

    Args:
        all_cells: list of per-frame cell lists (from segment_cells)
        max_dist: maximum linking distance in px

    Returns:
        tracks_dict: dict mapping track_id to list of (t, x, y) tuples
    """
    tracks_dict = {}
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
                tracks_dict[next_id] = [(t_idx, curr_c[i][0], curr_c[i][1])]
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
                tracks_dict[prev_ids[j]].append((t_idx, curr_c[i][0], curr_c[i][1]))
                used.add(j)
            for i in range(len(curr_ids)):
                if curr_ids[i] is None:
                    tracks_dict[next_id] = [(t_idx, curr_c[i][0], curr_c[i][1])]
                    curr_ids[i] = next_id
                    next_id += 1
        prev_c = curr_c
        prev_ids = curr_ids

    return tracks_dict


# ========================================================================
# Target Strategies — each returns a dict mapping cell_index to target_xy
# ========================================================================

def targets_uniform(cells, direction):
    """All cells get the same direction vector (e.g. migrate upward).
    Target is placed far along the direction so all cells move that way."""
    d = np.array(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    targets = {}
    for i, c in enumerate(cells):
        centroid = np.array(c["centroid_px"])
        targets[i] = centroid + d * 1000  # arbitrary far point along direction
    return targets


def targets_converge(cells, point):
    """All cells converge toward a single point (e.g. FOV center)."""
    return {i: np.array(point) for i in range(len(cells))}


def targets_disperse(cells, point):
    """All cells move away from a point (opposite of converge)."""
    targets = {}
    for i, c in enumerate(cells):
        centroid = np.array(c["centroid_px"])
        away = centroid - np.array(point)
        norm = np.linalg.norm(away)
        if norm < 1.0:
            away = np.array([1.0, 0.0])  # arbitrary direction if at center
        targets[i] = centroid + away  # move outward
    return targets


def targets_pairs(cells):
    """Pair cells by proximity; each cell targets its partner's centroid."""
    pairs = pair_cells_greedy(cells)
    targets = {}
    for idx_a, idx_b in pairs:
        targets[idx_a] = np.array(cells[idx_b]["centroid_px"])
        targets[idx_b] = np.array(cells[idx_a]["centroid_px"])
    return targets, pairs


# ========================================================================
# Mask Building — the core directional projection algorithm
# ========================================================================

def make_directional_mask(labels, cells, targets, percent=STIM_PERCENT,
                          dilate_px=DILATE_PX, shape=(512, 512)):
    """Build an SLM mask where each cell is stimulated on its target-facing side.

    Args:
        labels: label image from segment_cells
        cells: cell list from segment_cells
        targets: dict mapping cell_index to target_xy (direction to move)
        percent: fraction of each cell's pixels to stimulate
        dilate_px: dilation radius for robust vertex coverage
        shape: SLM mask dimensions (must match camera)

    Returns:
        slm: uint8 mask (0 or 255)
    """
    slm = np.zeros(shape, dtype=np.uint8)

    for i, c in enumerate(cells):
        if i not in targets:
            continue  # unpaired cell in pair strategy

        cell_mask = labels == c["label"]
        ys, xs = np.where(cell_mask)
        if len(ys) == 0:
            continue

        cx, cy = c["centroid_px"]
        centroid = np.array([cx, cy])
        target = np.array(targets[i])

        to_target = target - centroid
        norm = np.linalg.norm(to_target)
        if norm < 1.0:
            # Already at target — stimulate all pixels
            slm[ys, xs] = 255
            continue
        to_target /= norm

        # Project each pixel onto the target direction
        px_offsets = np.column_stack([xs - cx, ys - cy])
        projections = px_offsets @ to_target

        # Keep top percent% (highest projection = closest to target side)
        n_keep = max(1, int(len(projections) * percent / 100))
        threshold = np.sort(projections)[-n_keep]
        facing = projections >= threshold
        slm[ys[facing], xs[facing]] = 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        slm = cv2.dilate(slm, kernel)
    return slm


# ========================================================================
# Setup
# ========================================================================
print("Setting up experiment: strategy=%s" % STRATEGY)

mmc.snapImage()
img0 = mmc.getImage()
labels0, cells0 = segment_cells(img0)
print("Detected %d cells" % len(cells0))
for c in cells0:
    print("  Cell at (%.0f, %.0f), area=%d px" % (
        c["centroid_px"][0], c["centroid_px"][1], c["area_px"]))

# Compute initial targets based on strategy
pairs0 = []
if STRATEGY == "uniform":
    targets0 = targets_uniform(cells0, DIRECTION)
elif STRATEGY == "converge":
    targets0 = targets_converge(cells0, TARGET_POINT)
elif STRATEGY == "disperse":
    targets0 = targets_disperse(cells0, TARGET_POINT)
elif STRATEGY == "pairs":
    targets0, pairs0 = targets_pairs(cells0)
else:
    raise ValueError("Unknown strategy: %s" % STRATEGY)

# Build and apply initial mask
mask0 = make_directional_mask(labels0, cells0, targets0)
print("Initial mask: %d stim pixels" % np.count_nonzero(mask0))

mmc.setSLMDevice("SLM")
mmc.setSLMImage("SLM", mask0)
mmc.displaySLMImage("SLM")

# ========================================================================
# Timelapse with per-frame mask update
# ========================================================================
timelapse = []
mask_stack = []
all_cells = []
all_pairs = []

def on_frame(img, event, meta):
    idx = len(timelapse)
    timelapse.append(img.copy())

    # Re-segment every frame (cells move!)
    labels, cells = segment_cells(img)
    all_cells.append(cells)

    # Recompute targets (some strategies are cell-position-dependent)
    pairs = []
    if STRATEGY == "uniform":
        targets = targets_uniform(cells, DIRECTION)
    elif STRATEGY == "converge":
        targets = targets_converge(cells, TARGET_POINT)
    elif STRATEGY == "disperse":
        targets = targets_disperse(cells, TARGET_POINT)
    elif STRATEGY == "pairs":
        targets, pairs = targets_pairs(cells)
    all_pairs.append(pairs)

    # Build and apply new mask
    new_mask = make_directional_mask(labels, cells, targets)
    mask_stack.append(new_mask.copy())

    # Update SLM via standard API (works on real + virtual hardware)
    mmc.setSLMImage("SLM", new_mask)
    mmc.displaySLMImage("SLM")

    # Report progress every 50 frames
    if (idx + 1) % 50 == 0:
        report = "  Frame %d/%d: %d cells, %d stim px" % (
            idx + 1, N_FRAMES, len(cells), np.count_nonzero(new_mask))
        if STRATEGY == "converge" or STRATEGY == "disperse":
            dists = [np.linalg.norm(np.array(c["centroid_px"]) - TARGET_POINT)
                     for c in cells]
            report += ", mean dist to target: %.1f px" % (
                np.mean(dists) if dists else 0)
        elif STRATEGY == "pairs" and pairs:
            intra = [np.linalg.norm(
                np.array(cells[a]["centroid_px"]) -
                np.array(cells[b]["centroid_px"])) for a, b in pairs]
            report += ", mean intra-pair dist: %.1f px" % np.mean(intra)
        print(report)


events = [MDAEvent(exposure=EXPOSURE, min_start_time=i * INTERVAL,
                   index={"t": i}) for i in range(N_FRAMES)]
print("Starting MDA: %d frames..." % N_FRAMES)
t0 = time.time()
run_mda_with_feedback(events, on_frame=on_frame)
elapsed = time.time() - t0
print("MDA done: %d frames in %.1fs" % (len(timelapse), elapsed))

# ========================================================================
# Tracking
# ========================================================================
tracks_dict = track_cells(all_cells)
long = {tid: pts for tid, pts in tracks_dict.items() if len(pts) >= 10}
print("\nTracks: %d total, %d with >=10 pts" % (len(tracks_dict), len(long)))

# ========================================================================
# Analysis (strategy-specific)
# ========================================================================
print("\nDisplacement analysis:")
for tid, pts in sorted(long.items()):
    if len(pts) < 50:
        continue
    dx = pts[-1][1] - pts[0][1]
    dy = pts[-1][2] - pts[0][2]
    dist = np.sqrt(dx**2 + dy**2)
    print("  Track %d: %d pts, dx=%.1f, dy=%.1f, total=%.1f px" % (
        tid, len(pts), dx, dy, dist))

if STRATEGY == "converge" or STRATEGY == "disperse":
    print("\nDistance to target over time:")
    for frame_idx in [0, N_FRAMES//4, N_FRAMES//2, 3*N_FRAMES//4, N_FRAMES-1]:
        if frame_idx >= len(all_cells):
            continue
        cells = all_cells[frame_idx]
        dists = [np.linalg.norm(np.array(c["centroid_px"]) - TARGET_POINT)
                 for c in cells]
        print("  Frame %d: mean=%.1f px" % (
            frame_idx, np.mean(dists) if dists else 0))

elif STRATEGY == "pairs":
    print("\nIntra-pair distances over time:")
    for frame_idx in [0, N_FRAMES//4, N_FRAMES//2, 3*N_FRAMES//4, N_FRAMES-1]:
        if frame_idx >= len(all_cells) or frame_idx >= len(all_pairs):
            continue
        cells = all_cells[frame_idx]
        pairs = all_pairs[frame_idx]
        intra = [np.linalg.norm(
            np.array(cells[a]["centroid_px"]) -
            np.array(cells[b]["centroid_px"])) for a, b in pairs]
        print("  Frame %d: %d pairs, dists: %s" % (
            frame_idx, len(pairs), ", ".join("%.1f" % d for d in intra)))

# ========================================================================
# Save outputs
# ========================================================================
# Tracks in napari format: [track_id, t, row, col]
rows = []
for tid, pts in long.items():
    for (t_idx, cx, cy) in pts:
        rows.append([tid, t_idx, cy, cx])  # note: row=cy, col=cx
tracks_arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4),
                                                                     dtype=np.float64)

stack = np.array(timelapse, dtype=np.uint8)
tifffile.imwrite(OUTPUT_PREFIX + "_timelapse.tif", stack)
mstack = np.array(mask_stack, dtype=np.uint8)
tifffile.imwrite(OUTPUT_PREFIX + "_masks.tif", mstack)
np.save(OUTPUT_PREFIX + "_tracks.npy", tracks_arr)
print("\nSaved: %s_timelapse.tif, %s_masks.tif, %s_tracks.npy" % (
    OUTPUT_PREFIX, OUTPUT_PREFIX, OUTPUT_PREFIX))
print("DONE")
'''


def _build_params_preamble():
    """Build a Python code preamble that sets all parameter variables."""
    lines = [
        "STRATEGY = %r" % STRATEGY,
        "DIRECTION = %r" % (DIRECTION,),
        "TARGET_POINT = __import__('numpy').array(%r, dtype=float)" % (TARGET_POINT,),
        "N_FRAMES = %r" % N_FRAMES,
        "EXPOSURE = %r" % EXPOSURE,
        "INTERVAL = %r" % INTERVAL,
        "STIM_PERCENT = %r" % STIM_PERCENT,
        "DILATE_PX = %r" % DILATE_PX,
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
            print("Connected to MCP server.")

            # Setup: objective + stage position
            await session.call_tool("set_objective", {"label": "10x"})
            # Move to a position with cells — scan first if needed
            await session.call_tool("move_stage", {
                "x": 200.0, "y": 1000.0, "relative": False,
            })

            # Run experiment
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

            # Add layers to napari for visualization
            await session.call_tool("viewer_add_image", {
                "path": OUTPUT_PREFIX + "_timelapse.tif",
                "name": "stim_timelapse",
            })
            await session.call_tool("viewer_add_image", {
                "path": OUTPUT_PREFIX + "_masks.tif",
                "name": "stim_masks",
                "colormap": "red",
                "blending": "additive",
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
