#!/usr/bin/env python3
"""
Cropped Cell Movies with Segmentation Overlay
==============================================

Generic post-processing pipeline: given a timelapse, a per-pixel classification
map, and tracks (napari format), produce a cropped RGB movie for each tracked
cell with segmentation contours and text labels burned in.

The script is experiment-agnostic.  Experiment-specific behaviour is injected
through two hooks:

    label_track(track_id, track_points, classification_stack)
        -> (short_label: str, color_bgr: tuple)
        Assign a human-readable label and contour color to a track.
        Default: labels every track "CELL" in cyan.

    select_tracks(tracks_dict)
        -> list[int]
        Choose which track IDs to process.
        Default: the N longest tracks.

Override these by editing the EXPERIMENT-SPECIFIC HOOKS section below, or by
importing this module and monkey-patching before calling main().

NOTE ON REAL EXPERIMENTS:
    The classification map fed into this script is produced *upstream* — by
    whatever segmentation/classification method suits the experiment.  For
    the virtual microscope a simple threshold + rule-based classifier works.
    For real data you would typically:
        - Replace segmentation with a pre-trained DL model (Cellpose, StarDist,
          Omnipose, etc.) that outputs instance masks.
        - Replace the phase classifier with a CNN/transformer trained on
          annotated morphology patches, or derive phases from fluorescent
          reporters (Annexin-V, caspase sensors, …).
    This script does NOT care how the classification was produced — it only
    reads the resulting label image.

INPUT:
    --timelapse   (T, H, W) uint8 TIFF
    --classes     (T, H, W) uint8 TIFF  (0 = background, >0 = class IDs)
    --tracks      napari-format .npy     [track_id, t, row, col]

OUTPUT:
    One RGB TIFF per track: (T, crop, crop, 3)
    Optionally pushed to napari via MCP.

USAGE:
    # With defaults (reads apoptosis_monitoring output):
    python samples/make_cell_crops.py

    # Custom paths:
    python samples/make_cell_crops.py \
        --timelapse /tmp/my_timelapse.tif \
        --classes   /tmp/my_classes.tif \
        --tracks    /tmp/my_tracks.npy \
        --output    /tmp/my_crops \
        --crop-size 128 \
        --max-tracks 10 \
        --no-napari
"""

import argparse
import asyncio
import json
import numpy as np
import cv2
import tifffile
from pathlib import Path

MCP_URL = "http://127.0.0.1:5500/mcp"

# ===========================================================================
# EXPERIMENT-SPECIFIC HOOKS — edit these for your experiment
# ===========================================================================

# Color palette (BGR for cv2) — maps label strings to contour colors
LABEL_COLORS = {
    "APOPTOTIC": (0, 0, 255),    # red
    "HEALTHY":   (0, 255, 0),    # green
    "CELL":      (255, 200, 0),  # cyan (default)
}


def label_track(track_id, track_points, classification_stack):
    """Assign a label and color to a track based on its content.

    Examines the classification values under the tracked cell across frames.
    Override this for experiment-specific labelling (e.g. DL-based phenotype
    classification, drug response categories, lineage labels, …).

    Args:
        track_id: integer track identifier
        track_points: list of (t, row, col) tuples
        classification_stack: (T, H, W) uint8 array of per-pixel class IDs

    Returns:
        (label: str, color_bgr: tuple of 3 ints)
    """
    # Sample the classification value at the cell centroid for a few frames
    n_sample = min(20, len(track_points))
    step = max(1, len(track_points) // n_sample)
    class_vals = []
    for i in range(0, len(track_points), step):
        t, r, c = track_points[i]
        ri, ci = int(round(r)), int(round(c))
        if 0 <= t < classification_stack.shape[0]:
            h, w = classification_stack.shape[1:3]
            ri = np.clip(ri, 0, h - 1)
            ci = np.clip(ci, 0, w - 1)
            class_vals.append(int(classification_stack[t, ri, ci]))

    # Heuristic: if the cell ever had a non-healthy class (>1), call it
    # by its dominant non-healthy class.  Otherwise "HEALTHY".
    non_bg = [v for v in class_vals if v > 0]
    non_healthy = [v for v in non_bg if v > 1]

    if non_healthy:
        label = "APOPTOTIC"
    elif non_bg:
        label = "HEALTHY"
    else:
        label = "CELL"

    color = LABEL_COLORS.get(label, LABEL_COLORS["CELL"])
    return label, color


def select_tracks(tracks_dict, max_tracks):
    """Choose which tracks to process — longest N by default.

    Override for experiment-specific selection (e.g. only tracks that start
    in frame 0, only tracks in a certain spatial region, etc.).
    """
    ranked = sorted(tracks_dict.keys(),
                    key=lambda tid: len(tracks_dict[tid]), reverse=True)
    return ranked[:max_tracks]


# ===========================================================================
# GENERIC PIPELINE — no experiment-specific logic below this line
# ===========================================================================

def parse_tracks(tracks_arr):
    """Parse napari tracks array into dict: track_id -> [(t, row, col), ...]."""
    tracks = {}
    for row in tracks_arr:
        tid = int(row[0])
        t, r, c = int(row[1]), float(row[2]), float(row[3])
        tracks.setdefault(tid, []).append((t, r, c))
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return tracks


def build_position_array(track_points, n_frames):
    """Build (row, col) for every frame.  Hold first/last known position outside track span.
    Linearly interpolate across gaps within the track."""
    positions = np.zeros((n_frames, 2), dtype=np.float64)
    frame_map = {t: (r, c) for t, r, c in track_points}
    first_t, last_t = track_points[0][0], track_points[-1][0]
    first_pos = (track_points[0][1], track_points[0][2])
    last_pos = (track_points[-1][1], track_points[-1][2])

    for f in range(n_frames):
        if f in frame_map:
            positions[f] = frame_map[f]
        elif f < first_t:
            positions[f] = first_pos
        elif f > last_t:
            positions[f] = last_pos
        else:
            prev_t = max(t for t in frame_map if t <= f)
            next_t = min(t for t in frame_map if t >= f)
            if prev_t == next_t:
                positions[f] = frame_map[prev_t]
            else:
                alpha = (f - prev_t) / (next_t - prev_t)
                pr, pc = frame_map[prev_t]
                nr, nc = frame_map[next_t]
                positions[f] = (pr + alpha * (nr - pr), pc + alpha * (nc - pc))
    return positions


def crop_centered(img, center_row, center_col, size):
    """Crop a square window, zero-padding if the window extends past image edges."""
    h, w = img.shape[:2]
    half = size // 2
    r0 = int(round(center_row)) - half
    c0 = int(round(center_col)) - half

    r0c, c0c = max(0, r0), max(0, c0)
    r1c, c1c = min(h, r0 + size), min(w, c0 + size)

    shape = (size, size, img.shape[2]) if img.ndim == 3 else (size, size)
    out = np.zeros(shape, dtype=img.dtype)
    out[r0c - r0:r0c - r0 + r1c - r0c,
        c0c - c0:c0c - c0 + c1c - c0c] = img[r0c:r1c, c0c:c1c]
    return out


def make_crop_movie(timelapse, classification, positions, track_id,
                    label, color_bgr, crop_size, contour_thickness=1):
    """Build an RGB crop movie for one track."""
    n_frames = timelapse.shape[0]
    movie = np.zeros((n_frames, crop_size, crop_size, 3), dtype=np.uint8)

    for f in range(n_frames):
        row, col = positions[f]
        gray_crop = crop_centered(timelapse[f], row, col, crop_size)
        class_crop = crop_centered(classification[f], row, col, crop_size)

        rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

        # Segmentation contour
        seg_mask = (class_crop > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, color_bgr, contour_thickness)

        # Text overlay
        cv2.putText(rgb, f"{label} #{track_id}", (2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1, cv2.LINE_AA)
        cv2.putText(rgb, f"f{f}", (2, crop_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1,
                    cv2.LINE_AA)

        movie[f] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return movie


def run_pipeline(timelapse_path, classes_path, tracks_path,
                 output_prefix, crop_size, max_tracks, push_napari):
    """Main pipeline: load → select → crop → save → (optionally) push to napari."""
    # Load
    print("Loading data...")
    timelapse = tifffile.imread(timelapse_path)
    classification = tifffile.imread(classes_path)
    tracks_arr = np.load(tracks_path)
    n_frames = timelapse.shape[0]
    print(f"  Timelapse:       {timelapse.shape}")
    print(f"  Classification:  {classification.shape}")
    print(f"  Tracks:          {tracks_arr.shape}")

    # Parse & select
    tracks = parse_tracks(tracks_arr)
    print(f"\n{len(tracks)} tracks found: {sorted(tracks.keys())}")
    target_ids = select_tracks(tracks, max_tracks)
    print(f"Processing {len(target_ids)} tracks: {target_ids}")

    if not target_ids:
        print("No tracks to process.")
        return []

    # Crop each track
    output_paths = []
    for tid in target_ids:
        pts = tracks[tid]
        label, color = label_track(tid, pts, classification)
        first_f, last_f = pts[0][0], pts[-1][0]
        print(f"\n  Track {tid} [{label}]: {len(pts)} pts, "
              f"frames {first_f}-{last_f}")

        positions = build_position_array(pts, n_frames)
        movie = make_crop_movie(timelapse, classification, positions,
                                tid, label, color, crop_size)

        out_path = f"{output_prefix}_track{tid}.tif"
        tifffile.imwrite(out_path, movie)
        output_paths.append((tid, label, out_path))
        print(f"    -> {out_path}  {movie.shape}")

    # Push to napari
    if push_napari:
        asyncio.run(_push_to_napari(output_paths))

    return output_paths


async def _push_to_napari(output_paths):
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    print("\nConnecting to MCP server...")
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            for tid, label, path in output_paths:
                name = f"crop_{label.lower()}_{tid}"
                result = await session.call_tool("viewer_add_image", {
                    "path": path,
                    "name": name,
                })
                for b in result.content:
                    if hasattr(b, "text"):
                        d = json.loads(b.text)
                        if d.get("error"):
                            print(f"  ERROR: {name}: {d['error']}")
                        else:
                            print(f"  Added layer: {name}")
    print("Done.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Cropped cell movies with segmentation overlay")
    p.add_argument("--timelapse", default="/tmp/apoptosis_monitor_timelapse.tif",
                   help="Path to (T,H,W) uint8 timelapse TIFF")
    p.add_argument("--classes", default="/tmp/apoptosis_monitor_classification.tif",
                   help="Path to (T,H,W) uint8 classification TIFF")
    p.add_argument("--tracks", default="/tmp/apoptosis_monitor_tracks.npy",
                   help="Path to napari-format tracks .npy [id, t, row, col]")
    p.add_argument("--output", default="/tmp/apoptosis_crop",
                   help="Output prefix (track ID + .tif appended)")
    p.add_argument("--crop-size", type=int, default=96,
                   help="Side length of crop window in pixels (default: 96)")
    p.add_argument("--max-tracks", type=int, default=10,
                   help="Max number of tracks to process (default: 10)")
    p.add_argument("--no-napari", action="store_true",
                   help="Skip pushing layers to napari via MCP")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        timelapse_path=args.timelapse,
        classes_path=args.classes,
        tracks_path=args.tracks,
        output_prefix=args.output,
        crop_size=args.crop_size,
        max_tracks=args.max_tracks,
        push_napari=not args.no_napari,
    )
