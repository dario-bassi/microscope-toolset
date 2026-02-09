#!/usr/bin/env python3
"""
Apoptosis Monitoring Experiment — Cell Health & Phase Classification
=====================================================================

PROMPT: "Monitor cells undergoing apoptosis, classify each phase in real-time,
and track population dynamics over the full 60-second death cycle."

This experiment exercises the virtual microscope's apoptosis simulation by:
    - Triggering apoptosis on a subset of cells
    - Acquiring a timelapse covering the full 60s death cycle
    - Per-frame morphometric analysis: area, velocity, circularity
    - Real-time classification into: Healthy / Shrinkage / Blebbing / Bodies / Phagocytosis
    - Post-hoc population dynamics and per-cell phase timeline

APOPTOSIS PHASES (virtual microscope):
    - Shrinkage    (0-20s):  Cell shrinks to 70%, movement stops
    - Blebbing     (20-40s): 8-10 membrane protrusions, cell stays at 70%
    - Apoptotic Bodies (40-50s): Cell fragments into 3-5 small bodies
    - Phagocytosis (50-60s): Bodies fade out, then removed from simulation

CLASSIFICATION FEATURES:
    - Area ratio:    current_area / baseline_area  (drops during shrinkage)
    - Velocity:      displacement between frames    (healthy move, apoptotic stop)
    - Circularity:   4*pi*area / perimeter^2        (drops during blebbing)

USAGE:
    1. Start napari, load virtual_cycle.cfg, start MCP server
    2. python samples/apoptosis_monitoring.py

OUTPUT:
    - Console:  Per-frame summary every 50 frames
    - Files:    timelapse TIFF, classification overlay TIFF, tracks NPY
    - napari:   timelapse + color-coded classification overlay + tracks
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
N_FRAMES = 3000             # ~60s sim time (sim scales wall-clock by 0.3x)
EXPOSURE = 50.0
INTERVAL = 0.1              # seconds between frames
THRESHOLD_SIGMA = 1.5       # cell detection threshold
MIN_AREA_PX = 30            # minimum cell area in pixels
N_APOPTOTIC = 3             # number of cells to trigger apoptosis on
MIN_CELLS = 6               # minimum cells required at chosen position
OUTPUT_PREFIX = "/tmp/apoptosis_monitor"

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

# ========================================================================
# Helpers
# ========================================================================

def segment_cells(img, threshold_sigma=THRESHOLD_SIGMA, min_area_px=MIN_AREA_PX):
    """Segment bright cells via adaptive thresholding + connected components.

    Returns:
        labels: 2D int array where each cell has a unique label (0 = bg)
        cells: list of dicts with label, centroid_px (x,y), area_px, perimeter, circularity
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
        mask_i = binary_fill_holes(labels == i)
        labels[mask_i] = i
        cx, cy = centroids[i]
        area = int(mask_i.sum())

        # Compute perimeter via contours for circularity
        mask_uint8 = mask_i.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0.0
        circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 1.0
        circularity = min(circularity, 1.0)  # cap at 1.0

        # Mean intensity within the cell
        mean_intensity = float(fimg[mask_i].mean())

        cells.append({
            "label": i,
            "centroid_px": (float(cx), float(cy)),
            "area_px": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "mean_intensity": mean_intensity,
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


def compute_velocity(cells_now, cells_prev):
    """Compute per-cell velocity (displacement) between two consecutive frames.

    Matches cells by nearest centroid (assumes small displacement).
    Returns dict mapping current cell index to velocity in px.
    """
    if cells_prev is None or not cells_prev or not cells_now:
        return {i: 0.0 for i in range(len(cells_now))}

    prev_c = np.array([c["centroid_px"] for c in cells_prev])
    curr_c = np.array([c["centroid_px"] for c in cells_now])
    dists = cdist(curr_c, prev_c)

    velocities = {}
    used = set()
    for flat in np.argsort(dists, axis=None):
        i, j = divmod(int(flat), dists.shape[1])
        if i in velocities or j in used:
            continue
        if dists[i, j] > 50.0:
            break
        velocities[i] = float(dists[i, j])
        used.add(j)

    # Cells with no match get velocity 0 (new cells)
    for i in range(len(cells_now)):
        if i not in velocities:
            velocities[i] = 0.0

    return velocities


def classify_cell(area_ratio, velocity, circularity, was_tracked, intensity_ratio=1.0):
    """Classify a cell's apoptotic phase based on morphometric features.

    Returns one of: 'Healthy', 'Shrinkage', 'Blebbing', 'Bodies', 'Phagocytosis'
    """
    # Phagocytosis: small objects with fading intensity
    if area_ratio < 0.3 and intensity_ratio < 0.5:
        return "Phagocytosis"

    # Apoptotic bodies: very small fragments
    if area_ratio < 0.3:
        return "Bodies"

    # Blebbing: moderate shrinkage + low circularity (membrane protrusions)
    if velocity < 1.5 and area_ratio <= 0.6 and circularity < 0.7:
        return "Blebbing"

    # Shrinkage: moderate area drop, cell stopped moving
    if velocity < 1.5 and area_ratio <= 0.8:
        return "Shrinkage"

    # Healthy: moving normally, area stable
    return "Healthy"


# ========================================================================
# Setup: snap baseline + trigger apoptosis
# ========================================================================
print("=== Apoptosis Monitoring Experiment ===")

from src.virtual_microscope import simulation_bridge as bridge_module
sim = bridge_module.GLOBAL_BRIDGE._sim
pixel_size = 1.0  # 10x: 10/10 = 1.0 um/px

print("Snapping baseline frame...")
mmc.snapImage()
img0 = mmc.getImage()
labels0, cells0 = segment_cells(img0)
n_cells = len(cells0)
print("Detected %d cells in baseline" % n_cells)

if n_cells == 0:
    print("ERROR: No cells detected!")
    raise RuntimeError("No cells found")

for c in cells0:
    print("  Cell at (%.0f, %.0f), area=%d px, circ=%.2f, intensity=%.1f" % (
        c["centroid_px"][0], c["centroid_px"][1], c["area_px"],
        c["circularity"], c["mean_intensity"]))

# Record baseline features
baseline_areas = {i: c["area_px"] for i, c in enumerate(cells0)}
baseline_centroids = {i: np.array(c["centroid_px"]) for i, c in enumerate(cells0)}
baseline_intensities = {i: c["mean_intensity"] for i, c in enumerate(cells0)}

# Trigger apoptosis on N_APOPTOTIC cells (leave the rest as healthy controls)
sim_cells = sim._cells
n_trigger = min(N_APOPTOTIC, max(n_cells - 1, 1))  # keep at least 1 healthy control
triggered_positions = []
triggered_det_indices = set()

camera_offset = np.array(sim.camera_offset)

for det_idx in range(n_trigger):
    det_cx, det_cy = cells0[det_idx]["centroid_px"]
    world_x = camera_offset[0] + det_cx * pixel_size
    world_y = camera_offset[1] + det_cy * pixel_size

    best_dist = float('inf')
    best_sim_idx = None
    for s_idx, sc in enumerate(sim_cells):
        if hasattr(sc, 'is_dying') and sc.is_dying:
            continue
        dx = sc.center[0] - world_x
        dy = sc.center[1] - world_y
        d = np.sqrt(dx**2 + dy**2)
        if d < best_dist:
            best_dist = d
            best_sim_idx = s_idx

    if best_sim_idx is not None:
        sim_cells[best_sim_idx].is_dying = True
        triggered_positions.append((det_cx, det_cy))
        triggered_det_indices.add(det_idx)
        print("Triggered apoptosis on sim cell %d (det cell %d at (%.0f, %.0f), dist=%.1f um)" % (
            best_sim_idx, det_idx, det_cx, det_cy, best_dist))

print("\nTriggered apoptosis on %d cells. %d healthy controls remain." % (
    len(triggered_positions), n_cells - len(triggered_positions)))

# ========================================================================
# Timelapse with per-frame analysis
# ========================================================================
timelapse = []
all_cells = []
all_classifications = []  # list of per-frame classification lists
all_features = []  # list of per-frame feature dicts
prev_cells = None
sim_times = []  # track actual simulation time per frame

# Pre-allocate classification overlay (built incrementally in on_frame)
# Only allocate after we know how many frames we'll get
class_frames = []  # list of uint8 512x512 classification images

# For linking detected cells to baseline
triggered_pos_arr = np.array(triggered_positions) if triggered_positions else np.zeros((0, 2))

# Read actual sim time from a dying cell's death_timer for ground truth
_max_sim_time = [0.0]  # mutable container for closure access
_all_dying_removed = [False]

def get_sim_time():
    """Get current apoptosis sim time from any dying cell.
    Returns monotonically increasing value (tracks max seen)."""
    for sc in sim._cells:
        if hasattr(sc, 'is_dying') and sc.is_dying and hasattr(sc, 'death_timer'):
            t = sc.death_timer
            if t > _max_sim_time[0]:
                _max_sim_time[0] = t
            return t
    # No dying cells found — either not triggered yet or all removed
    if _max_sim_time[0] > 0:
        _all_dying_removed[0] = True
    return _max_sim_time[0]


def on_frame(img, event, meta):
    global prev_cells
    idx = len(timelapse)
    timelapse.append(img.copy())

    # Track actual simulation time
    st = get_sim_time()
    sim_times.append(st)

    # Segment
    labels, cells = segment_cells(img)
    all_cells.append(cells)

    # Compute velocity relative to previous frame
    velocities = compute_velocity(cells, prev_cells)

    # Match current cells to baseline cells by nearest centroid
    # to get area_ratio and intensity_ratio
    frame_classifications = []
    frame_features = []

    if cells:
        curr_centroids = np.array([c["centroid_px"] for c in cells])
        bl_keys = sorted(baseline_centroids.keys())
        bl_arr = np.array([baseline_centroids[k] for k in bl_keys])

        for i, c in enumerate(cells):
            if len(bl_arr) > 0:
                dists_to_bl = np.linalg.norm(curr_centroids[i] - bl_arr, axis=1)
                nearest_idx = int(np.argmin(dists_to_bl))
                nearest_bl = bl_keys[nearest_idx]
                area_ratio = c["area_px"] / max(baseline_areas[nearest_bl], 1)
                intensity_ratio = c["mean_intensity"] / max(baseline_intensities[nearest_bl], 1.0)
            else:
                area_ratio = 1.0
                intensity_ratio = 1.0

            velocity = velocities.get(i, 0.0)
            circularity = c["circularity"]

            phase = classify_cell(area_ratio, velocity, circularity,
                                  was_tracked=True, intensity_ratio=intensity_ratio)

            frame_classifications.append(phase)
            frame_features.append({
                "area_ratio": area_ratio,
                "velocity": velocity,
                "circularity": circularity,
                "intensity_ratio": intensity_ratio,
                "phase": phase,
            })

    all_classifications.append(frame_classifications)
    all_features.append(frame_features)

    # Build classification overlay for this frame
    PHASE_COLORS = {"Healthy": 1, "Shrinkage": 2, "Blebbing": 3, "Bodies": 4, "Phagocytosis": 5}
    class_img = np.zeros((512, 512), dtype=np.uint8)
    for i, c in enumerate(cells):
        if i < len(frame_classifications):
            cv = PHASE_COLORS.get(frame_classifications[i], 0)
            class_img[labels == c["label"]] = cv
    class_frames.append(class_img)

    prev_cells = cells

    # Report every 100 frames
    if (idx + 1) % 100 == 0:
        phase_counts = {}
        for p in frame_classifications:
            phase_counts[p] = phase_counts.get(p, 0) + 1
        # Also get ground truth phases from sim
        gt_phases = {}
        for sc in sim._cells:
            if hasattr(sc, 'is_dying') and sc.is_dying:
                p = sc.apoptosis_death_phase
                gt_phases[p] = gt_phases.get(p, 0) + 1
        report = "  Frame %d/%d (sim_t=%.1fs): %d cells | " % (
            idx + 1, N_FRAMES, st, len(cells))
        report += " ".join("%s=%d" % (k, v) for k, v in sorted(phase_counts.items()))
        if gt_phases:
            report += " | GT: " + " ".join("%s=%d" % (k, v) for k, v in sorted(gt_phases.items()))
        print(report)


# Use a generator so we can stop early once all apoptosis is complete
def event_generator():
    extra_frames_after_done = 200  # acquire 200 more frames after all apoptosis completes
    frames_since_done = 0
    for i in range(N_FRAMES):
        if _all_dying_removed[0]:
            frames_since_done += 1
            if frames_since_done >= extra_frames_after_done:
                print("  All apoptotic cells removed + %d extra frames. Stopping at frame %d (sim_t=%.1fs)" % (
                    extra_frames_after_done, i, _max_sim_time[0]))
                return
        yield MDAEvent(exposure=EXPOSURE, min_start_time=i * INTERVAL,
                       index={"t": i})

print("\nStarting MDA: up to %d frames, %.1fs interval..." % (N_FRAMES, INTERVAL))
t0 = time.time()
run_mda_with_feedback(event_generator(), on_frame=on_frame)
elapsed = time.time() - t0
print("MDA done: %d frames in %.1fs (sim time: %.1fs)" % (
    len(timelapse), elapsed, sim_times[-1] if sim_times else 0))

# ========================================================================
# Tracking
# ========================================================================
tracks_dict = track_cells(all_cells)
long = {tid: pts for tid, pts in tracks_dict.items() if len(pts) >= 10}
print("\nTracks: %d total, %d with >=10 pts" % (len(tracks_dict), len(long)))

# ========================================================================
# Post-Analysis: per-cell phase timeline + population dynamics
# ========================================================================
print("\n=== Per-Track Summary ===")

# Map each track to its first-frame position to determine if apoptotic
for tid, pts in sorted(long.items()):
    if len(pts) < 20:
        continue
    first_x, first_y = pts[0][1], pts[0][2]

    # Check if this track started near a triggered cell
    is_apoptotic = False
    if len(triggered_pos_arr) > 0:
        dists_to_triggered = [np.sqrt((first_x - tx)**2 + (first_y - ty)**2)
                              for tx, ty in triggered_pos_arr]
        if min(dists_to_triggered) < 30:
            is_apoptotic = True

    # Collect phases for this track's frames
    track_phases = []
    for (t_idx, cx, cy) in pts:
        if t_idx < len(all_classifications):
            fc = all_classifications[t_idx]
            # Find which cell in this frame matches (nearest centroid)
            if all_cells[t_idx]:
                frame_c = np.array([c["centroid_px"] for c in all_cells[t_idx]])
                dists = np.sqrt((frame_c[:, 0] - cx)**2 + (frame_c[:, 1] - cy)**2)
                nearest = int(np.argmin(dists))
                if nearest < len(fc):
                    track_phases.append(fc[nearest])
                else:
                    track_phases.append("?")
            else:
                track_phases.append("?")

    # Summarize phase transitions
    transitions = []
    if track_phases:
        current = track_phases[0]
        transitions.append((0, current))
        for i, p in enumerate(track_phases[1:], 1):
            if p != current:
                transitions.append((i, p))
                current = p

    dx = pts[-1][1] - pts[0][1]
    dy = pts[-1][2] - pts[0][2]
    displacement = np.sqrt(dx**2 + dy**2)

    label = "APOPTOTIC" if is_apoptotic else "HEALTHY"
    print("  Track %d [%s]: %d pts, displacement=%.1f px" % (
        tid, label, len(pts), displacement))
    print("    Phase timeline: %s" % " -> ".join(
        "%s(@%d)" % (phase, t) for t, phase in transitions))

# Population dynamics over time
print("\n=== Population Dynamics ===")
print("Frame | SimTime | Total | Healthy | Shrinkage | Blebbing | Bodies | Phagocytosis")
print("-" * 85)
checkpoints = list(range(0, min(len(all_classifications), N_FRAMES), 100))
if len(all_classifications) > 0:
    checkpoints.append(len(all_classifications) - 1)
checkpoints = sorted(set(checkpoints))
for f_idx in checkpoints:
    if f_idx >= len(all_classifications):
        continue
    fc = all_classifications[f_idx]
    counts = {"Healthy": 0, "Shrinkage": 0, "Blebbing": 0, "Bodies": 0, "Phagocytosis": 0}
    for p in fc:
        if p in counts:
            counts[p] += 1
    sim_t = sim_times[f_idx] if f_idx < len(sim_times) else 0.0
    print(" %4d | %7.1f | %5d | %7d | %9d | %8d | %6d | %12d" % (
        f_idx, sim_t, len(fc),
        counts["Healthy"], counts["Shrinkage"], counts["Blebbing"],
        counts["Bodies"], counts["Phagocytosis"]))

# ========================================================================
# Build classification overlay (colored label image)
# ========================================================================
# Classification overlay already built incrementally in on_frame
class_stack = np.array(class_frames, dtype=np.uint8) if class_frames else np.zeros((1, 512, 512), dtype=np.uint8)

# ========================================================================
# Save outputs
# ========================================================================
# Tracks in napari format: [track_id, t, row, col]
rows = []
for tid, pts in long.items():
    for (t_idx, cx, cy) in pts:
        rows.append([tid, t_idx, cy, cx])  # row=cy, col=cx
tracks_arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4),
                                                                     dtype=np.float64)

stack = np.array(timelapse, dtype=np.uint8)
tifffile.imwrite(OUTPUT_PREFIX + "_timelapse.tif", stack)
tifffile.imwrite(OUTPUT_PREFIX + "_classification.tif", class_stack)
np.save(OUTPUT_PREFIX + "_tracks.npy", tracks_arr)
print("\nSaved: %s_timelapse.tif, %s_classification.tif, %s_tracks.npy" % (
    OUTPUT_PREFIX, OUTPUT_PREFIX, OUTPUT_PREFIX))
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
        "N_APOPTOTIC = %r" % N_APOPTOTIC,
        "MIN_CELLS = %r" % MIN_CELLS,
        "OUTPUT_PREFIX = %r" % OUTPUT_PREFIX,
    ]
    return "\n".join(lines) + "\n"


GRID_SCAN_CODE = r'''
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

pixel_size = 1.0
fov = 512 * pixel_size
stage_dev = mmc.getXYStageDevice()
best_pos = None
best_n = 0

for gx in range(5):
    for gy in range(5):
        x = 100 + gx * fov * 0.8
        y = 100 + gy * fov * 0.8
        mmc.setXYPosition(x, y)
        mmc.waitForDevice(stage_dev)
        mmc.snapImage()
        img = mmc.getImage()
        fimg = img.astype(np.float32)
        thresh = fimg.mean() + THRESHOLD_SIGMA * fimg.std()
        binary = (fimg > thresh).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        n = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA_PX)
        print("  (%.0f, %.0f): %d cells" % (x, y, n))
        if n > best_n:
            best_n = n
            best_pos = (x, y)
            if n >= MIN_CELLS:
                break
    if best_n >= MIN_CELLS:
        break

if best_pos:
    mmc.setXYPosition(best_pos[0], best_pos[1])
    mmc.waitForDevice(stage_dev)
    print("BEST_POS=%.1f,%.1f,%d" % (best_pos[0], best_pos[1], best_n))
else:
    print("BEST_POS=NONE")
'''


async def main():
    async with streamablehttp_client(MCP_URL, sse_read_timeout=600) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            print("Connected to MCP server.")

            # Setup objective
            await session.call_tool("set_objective", {"label": "10x"})

            # Grid scan to find position with enough cells
            print("Scanning for cells...")
            scan_code = _build_params_preamble() + GRID_SCAN_CODE
            result = await session.call_tool("execute_python_code", {
                "code": scan_code,
                "execution_mode": "live",
            })
            for b in result.content:
                if hasattr(b, "text"):
                    d = json.loads(b.text)
                    output = d.get("output", "")
                    print(output)
                    if d.get("error"):
                        print("ERROR:", d["error"])

            # Run main experiment
            code = _build_params_preamble() + EXPERIMENT_CODE
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

            # Add layers to napari
            await session.call_tool("viewer_add_image", {
                "path": OUTPUT_PREFIX + "_timelapse.tif",
                "name": "apoptosis_timelapse",
            })
            await session.call_tool("viewer_add_image", {
                "path": OUTPUT_PREFIX + "_classification.tif",
                "name": "classification",
                "colormap": "turbo",
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
