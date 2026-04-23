"""
Detailed performance profiler for microscope simulation at different cell counts
"""
import time
import numpy as np
from src.virtual_microscope.microscope_sim_optimized import MicroscopeSimOptmized
from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore
from src.virtual_microscope.initialize_virtual_microscope import initialize_virtual_microscope
import src.virtual_microscope.simulation_bridge as bridge_module

class TimingContext:
    """Context manager for timing code blocks"""
    def __init__(self, name):
        self.name = name
        self.start = None
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000
        print(f"  {self.name:40s}: {self.elapsed:7.2f} ms")

def profile_snap_frame(microscope, num_frames=10):
    """Profile a single snap_frame call and time each component"""

    print(f"\n{'='*80}")
    print(f"Profiling {num_frames} frames with {len(microscope._cells)} cells")
    print(f"{'='*80}\n")

    # Accumulate timings
    timings = {
        'physics_update': [],
        'cell_behaviors': [],
        'collision_detection': [],
        'rendering': [],
        'post_processing': [],
        'total': []
    }

    for frame_num in range(num_frames):
        print(f"Frame {frame_num + 1}/{num_frames}:")

        frame_start = time.perf_counter()

        # ===== PHYSICS UPDATE =====
        with TimingContext("Physics Update (Numba parallel)") as t:
            dt = 0.016 * 0.3
            microscope.update(dt)
        timings['physics_update'].append(t.elapsed)

        # ===== CELL BEHAVIORS =====
        with TimingContext("Cell Behaviors (state, constriction, etc)") as t:
            for cell in microscope._cells:
                if hasattr(cell, "update_behavior"):
                    cell.update_behavior(dt)
        timings['cell_behaviors'].append(t.elapsed)

        # ===== COLLISION DETECTION =====
        with TimingContext("Collision Detection (spatial grid)") as t:
            microscope._handle_collisions_with_spatial_grid()
        timings['collision_detection'].append(t.elapsed)

        # ===== RENDERING =====
        with TimingContext("Rendering (OpenCV drawing)") as t:
            microscope._update_mode()
            microscope._update_objectif()
            for cell in microscope._cells:
                microscope._update_cell_fluorescence(cell, microscope.mode)
            img = microscope.renderer.render_cells(
                microscope._cells, microscope.mode,
                tuple(microscope.camera_offset),
                microscope.focal_plane
            )
        timings['rendering'].append(t.elapsed)

        # ===== POST-PROCESSING =====
        with TimingContext("Post-processing (intensity, exposure, convert)") as t:
            img = (img.astype(np.float32) * 1.0 * 0.01 * 1.0).clip(0, 255).astype(np.uint8)
            import cv2
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        timings['post_processing'].append(t.elapsed)

        frame_elapsed = (time.perf_counter() - frame_start) * 1000
        timings['total'].append(frame_elapsed)

        fps = 1000.0 / frame_elapsed if frame_elapsed > 0 else 0
        print(f"  {'TOTAL FRAME TIME':40s}: {frame_elapsed:7.2f} ms  ({fps:5.1f} FPS)\n")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (average across all frames)")
    print(f"{'='*80}\n")

    for component, times in timings.items():
        avg = np.mean(times)
        std = np.std(times)
        min_t = np.min(times)
        max_t = np.max(times)
        pct = (avg / np.mean(timings['total'])) * 100 if component != 'total' else 100

        if component == 'total':
            print(f"\n{'='*80}")
            print(f"TOTAL FRAME TIME")
            print(f"{'='*80}")
            fps = 1000.0 / avg if avg > 0 else 0
            print(f"  Average:    {avg:7.2f} ms  ({fps:5.1f} FPS)")
            print(f"  Std Dev:    {std:7.2f} ms")
            print(f"  Min:        {min_t:7.2f} ms")
            print(f"  Max:        {max_t:7.2f} ms")
            target_ms = 16.67  # 60 FPS
            if avg > target_ms:
                print(f"  ⚠️  SLOW: {avg - target_ms:5.2f} ms over 60 FPS target")
            else:
                print(f"  ✓ GOOD: {target_ms - avg:5.2f} ms under 60 FPS target")
        else:
            print(f"{component:40s}")
            print(f"  Average:    {avg:7.2f} ms  ({pct:5.1f}% of frame)")
            print(f"  Std Dev:    {std:7.2f} ms")
            print(f"  Min:        {min_t:7.2f} ms")
            print(f"  Max:        {max_t:7.2f} ms")

    # Ranking
    print(f"\n{'='*80}")
    print("BOTTLENECK RANKING (what's taking the most time)")
    print(f"{'='*80}\n")

    components = {
        'Physics Update': np.mean(timings['physics_update']),
        'Cell Behaviors': np.mean(timings['cell_behaviors']),
        'Collision Detection': np.mean(timings['collision_detection']),
        'Rendering': np.mean(timings['rendering']),
        'Post-processing': np.mean(timings['post_processing']),
    }

    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, time_ms) in enumerate(sorted_components, 1):
        pct = (time_ms / np.mean(timings['total'])) * 100
        bar = "█" * int(pct / 2)
        print(f"{rank}. {name:30s} {time_ms:7.2f} ms ({pct:5.1f}%) {bar}")

def main():
    """Main profiling function"""

    print("\n" + "="*80)
    print("MICROSCOPE SIMULATION PROFILER - Performance Analysis")
    print("="*80)

    # Test different cell counts
    cell_counts = [20, 50, 100, 200, 300]

    for num_cells in cell_counts:
        print(f"\n\n{'#'*80}")
        print(f"# Testing with {num_cells} cells")
        print(f"{'#'*80}")

        try:
            # Create simulation
            microscope = MicroscopeSimOptmized(
                cell_type="cycle",
                nb_cells=num_cells,
                width=1500,
                height=1500
            )

            # Initialize PyMMCore integration
            core = UniMMCore()
            initialize_virtual_microscope(core, cell_type="cycle")

            # Profile it
            profile_snap_frame(microscope, num_frames=5)

        except Exception as e:
            print(f"❌ Error profiling {num_cells} cells: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'='*80}")
    print("PROFILING COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
