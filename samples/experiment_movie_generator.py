"""
Experiment Movie Generator for Apoptosis & Mitosis Cell Cycle Simulation

Creates multi-mode (brightfield, nucleus, membrane) TIFF movies with tracked cell data.
- 20 cells per frame (5 apoptosis, 5 mitosis, 10 interphase)
- 14 frames per cycle (5 seconds between frames)
- 125 repeating cycles with random state assignments
- Tracks cell position, state, and identity in CSV
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import random
import csv
from datetime import datetime

from src.virtual_microscope.cell_cycle import CellCycleNormal
from src.virtual_microscope.renderer import Renderer


class ExperimentMovieGenerator:
    """Generate experimental movies with tracked cell data."""

    def __init__(
        self,
        output_dir: str = "./experiment_output",
        width: int = 512,
        height: int = 512,
        num_cells: int = 20,
        num_apoptotic: int = 5,
        num_mitotic: int = 5,
        frames_per_cycle: int = 16,
        frame_interval_seconds: float = 5.0,
        num_cycles: int = 125,
        exposure: int = 50,
        intensity: float = 2.0,
        cell_drift_speed: float = 0.5,  # Minimal drift but visible
    ):
        """Initialize experiment parameters."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.width = width
        self.height = height
        self.num_cells = num_cells
        self.num_apoptotic = num_apoptotic
        self.num_mitotic = num_mitotic
        self.num_interphase = num_cells - num_apoptotic - num_mitotic

        self.frames_per_cycle = frames_per_cycle
        self.frame_interval = frame_interval_seconds
        self.num_cycles = num_cycles
        self.exposure = exposure
        self.intensity = intensity
        self.cell_drift_speed = cell_drift_speed

        # Initialize renderer for grayscale conversion
        self.renderer = Renderer(width=width, height=height)

        self.cells: List[CellCycleNormal] = []
        self.cell_positions_history: Dict[int, List[Tuple[float, float, float, str]]] = {}
        self.frame_count = 0
        self.cycle_count = 0

        print(f"[OK] Initialized ExperimentMovieGenerator")
        print(f"  - Space: {width}x{height}")
        print(f"  - Cells: {num_cells} (A:{num_apoptotic}, M:{num_mitotic}, I:{self.num_interphase})")
        print(f"  - Frames/cycle: {frames_per_cycle} @ {frame_interval_seconds}s")
        print(f"  - Total cycles: {num_cycles}")
        print(f"  - Cell drift: {cell_drift_speed} pixels/frame")
        print(f"  - Capture: Grayscale via renderer (mode 0)")

    def _initialize_cells(self) -> None:
        """Create 20 cells randomly scattered in the space with trackable IDs."""
        self.cells = []
        margin = 10  # pixels from edge to keep cells visible

        cell_id = 0
        for _ in range(self.num_cells):
            # Random position within margins
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)

            # Create cell with minimal drift
            cell = CellCycleNormal(
                width=self.width,
                height=self.height,
                base_radius=20.0,
                vertices=32,
                seed=cell_id,
            )

            # Set initial position
            cell.center = np.array([x, y], dtype=np.float64)

            # Set minimal drift velocity (random direction)
            angle = random.uniform(0, 2 * np.pi)
            cell.vel = np.array([
                self.cell_drift_speed * np.cos(angle),
                self.cell_drift_speed * np.sin(angle)
            ], dtype=np.float64)

            # Enable modest brownian motion for natural drift
            cell.brownian_d = 10.0  # Enable brownian motion

            # Store trackable cell ID that includes movie number
            trackable_id = f"movie_{self.cycle_count}_cell_{cell_id}"
            cell.trackable_id = trackable_id

            self.cells.append(cell)
            self.cell_positions_history[trackable_id] = []
            cell_id += 1

        print(f"[OK] Initialized {len(self.cells)} cells (Movie {self.cycle_count})")

    def _assign_cell_states(self) -> None:
        """Randomly assign cell states: apoptosis, mitosis, or interphase.

        Interphase cells naturally progress through G1 → S → G2 phases.
        """
        # Shuffle cell list to randomly assign states
        indices = list(range(self.num_cells))
        random.shuffle(indices)

        apoptotic_indices = set(indices[:self.num_apoptotic])
        mitotic_indices = set(indices[self.num_apoptotic:self.num_apoptotic + self.num_mitotic])
        interphase_indices = set(indices[self.num_apoptotic + self.num_mitotic:])

        for i, cell in enumerate(self.cells):
            if i in apoptotic_indices:
                # Initialize apoptosis
                cell.is_dying = True
                cell.apoptosis_death_phase = 'Shrinkage'
                cell.death_timer = 0.0
                cell.current_state = 'Apoptosis'

            elif i in mitotic_indices:
                # Set to mitotic phase
                cell.cell_cycle_state = 'M'
                cell.cell_mitosis_state = 'Prophase'
                cell.current_time_life = 480.0  # Start of M phase
                cell.current_state = 'Mitosis'

            else:
                # Interphase: randomly select starting point, let it naturally progress
                # through G1 → S → G2 phases
                random_time = random.uniform(0, 480)  # Random point in full interphase

                cell.cell_cycle_state = 'G1'  # Start with G1
                cell.cell_mitosis_state = 'Interphase'
                cell.current_time_life = random_time

                # Determine which phase cell is in based on random_time
                if random_time < 240:
                    current_phase = 'G1'
                elif random_time < 360:
                    current_phase = 'S'
                else:
                    current_phase = 'G2'

                cell.current_state = f'Interphase_{current_phase}'

    def _update_cells(self, dt: float) -> None:
        """Update cell states and positions."""
        for cell in self.cells:
            cell.update_behavior(dt)

    def _render_frame(self, mode: int = 0) -> np.ndarray:
        """Render frame in grayscale by rendering RGB and converting to grayscale.

        Args:
            mode: Visualization mode (0=brightfield, 1=nucleus, 2=membrane)
        """
        # Render RGB image in specified mode
        img_rgb = self.renderer.render_cell_cycle(
            self.cells,
            mode=mode,
            camera_offset=(0, 0),
            focal_plane=0.0
        )

        # Apply intensity and exposure
        img_rgb = np.clip(img_rgb.astype(np.float32) * self.intensity * 0.01 * self.exposure, 0, 255).astype(np.uint8)

        # Convert BGR to grayscale to simulate microscope image
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        return img_gray

    def _get_detailed_cell_state(self, cell: CellCycleNormal) -> str:
        """Get detailed cell state with specific phase information."""
        if cell.is_dying:
            # Apoptosis state with detailed phase
            return f"Apoptosis_{cell.apoptosis_death_phase}"
        elif cell.cell_cycle_state == 'M':
            # Mitosis state with detailed phase
            return f"Mitosis_{cell.cell_mitosis_state}"
        else:
            # Interphase state
            if cell.current_time_life < 240:
                phase = 'G1'
            elif cell.current_time_life < 360:
                phase = 'S'
            else:
                phase = 'G2'
            return f"Interphase_{phase}"

    def _track_cells(self, time_seconds: float) -> None:
        """Record cell positions and detailed states for ALL cells."""
        for cell in self.cells:
            # Use trackable ID that includes movie number
            trackable_id = cell.trackable_id
            state = self._get_detailed_cell_state(cell)
            self.cell_positions_history[trackable_id].append(
                (cell.center[0], cell.center[1], time_seconds, state)
            )

    def _save_tiff_stack(self, frames: List[np.ndarray], mode: int) -> Path:
        """Save grayscale frames as TIFF stack.

        Args:
            frames: List of grayscale frame arrays
            mode: Visualization mode (0=brightfield, 1=nucleus, 2=membrane)
        """
        mode_names = {0: "brightfield", 1: "nucleus", 2: "membrane"}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"experiment_{mode_names[mode]}_{timestamp}.tiff"

        # Ensure frames are uint8
        if frames[0].dtype != np.uint8:
            frames = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]

        # Use PIL for multi-page TIFF
        try:
            from PIL import Image
            # Frames are already grayscale from snap_frame
            img_pil = [Image.fromarray(f) for f in frames]
            img_pil[0].save(
                filename,
                save_all=True,
                append_images=img_pil[1:],
                duration=int(self.frame_interval * 1000),  # Convert to ms
                loop=0
            )
            print(f"  [OK] Saved TIFF: {filename} ({len(frames)} frames)")
            return filename
        except ImportError:
            # Fallback: save as individual frames
            for i, frame in enumerate(frames):
                frame_filename = self.output_dir / f"experiment_frame{i:04d}.png"
                cv2.imwrite(str(frame_filename), frame)
            print(f"  ✓ Saved {len(frames)} PNG frames (PIL not available)")
            return self.output_dir

    def _save_tracking_csv(self, mode: int) -> Path:
        """Save cell tracking data to CSV for ALL cycles and cells.

        Args:
            mode: Visualization mode (0=brightfield, 1=nucleus, 2=membrane)
        """
        mode_names = {0: "brightfield", 1: "nucleus", 2: "membrane"}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"tracking_{mode_names[mode]}_{timestamp}.csv"

        # Count total tracking entries
        total_entries = sum(len(history) for history in self.cell_positions_history.values())

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Updated header to show trackable cell ID format
            writer.writerow(['cell_id (movie_X_cell_Y)', 'x_pos', 'y_pos', 'time_seconds', 'state'])

            # Write all tracking data for each cell (sorted by movie then cell)
            for trackable_id in sorted(self.cell_positions_history.keys()):
                for x, y, t, state in self.cell_positions_history[trackable_id]:
                    writer.writerow([trackable_id, f'{x:.2f}', f'{y:.2f}', f'{t:.2f}', state])

        print(f"  [OK] Saved CSV: {filename} ({total_entries} entries)")
        print(f"    - {len(self.cell_positions_history)} unique cell identities")
        print(f"    - Format: movie_X_cell_Y where X=movie, Y=cell in movie")
        print(f"    - {self.num_cycles} cycles @ {self.frames_per_cycle} frames each")
        print(f"    - States include detailed phases (Metaphase, Shrinkage, etc.)")
        return filename

    def run_experiment(self) -> None:
        """Run the complete experiment for all 3 visualization modes."""
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {self.num_cycles} cycles × {self.frames_per_cycle} frames")
        print(f"All 3 modes: Brightfield, Nucleus, Membrane")
        print(f"Total: {self.num_cycles * self.frames_per_cycle * 3} frames ({self.num_cycles * self.frames_per_cycle} per mode)")
        print(f"{'='*60}\n")

        # Run experiment for each visualization mode
        mode_names = {0: "Brightfield", 1: "Nucleus fluorescence", 2: "Membrane fluorescence"}
        for mode in [0, 1, 2]:
            print(f"\n>>> Generating {mode_names[mode]} (mode {mode})...")
            self._run_single_mode(mode)

        print(f"\n{'='*60}")
        print(f"All modes complete! Results in {self.output_dir}")
        print(f"{'='*60}")

    def _run_single_mode(self, mode: int) -> None:
        """Run experiment for a single visualization mode.

        Args:
            mode: Visualization mode (0=brightfield, 1=nucleus, 2=membrane)
        """
        # Reset state for this mode
        all_frames = []
        self.cell_positions_history = {}
        self.frame_count = 0
        self.cycle_count = 0

        for cycle in range(self.num_cycles):
            if cycle % 10 == 0:
                print(f"  Cycle {cycle + 1}/{self.num_cycles}...")

            # Initialize cells with new random states
            self._initialize_cells()
            self._assign_cell_states()

            # Simulate frames
            for frame in range(self.frames_per_cycle):
                # Track positions and states BEFORE update
                time_seconds = (self.cycle_count * self.frames_per_cycle + frame) * self.frame_interval
                self._track_cells(time_seconds)

                # Update cell behavior
                dt = self.frame_interval
                self._update_cells(dt)

                # Capture grayscale frame in this mode
                img = self._render_frame(mode=mode)
                all_frames.append(img)

                self.frame_count += 1

            self.cycle_count += 1

        print(f"\n  Simulation Complete: {self.frame_count} frames rendered")
        print(f"  Saving results...")
        self._save_tiff_stack(all_frames, mode=mode)
        self._save_tracking_csv(mode=mode)


def run_experiment_simple(output_dir: str = "./experiment_output") -> None:
    """Simple entry point to run the experiment.

    Generates a grayscale TIFF movie with cell tracking data.
    - 16 frames per cycle (was 14)
    - 125 cycles with random state reassignment
    - Detailed state tracking: Metaphase, Shrinkage, etc.
    - Complete tracking for all cells from all cycles
    """
    generator = ExperimentMovieGenerator(
        output_dir=output_dir,
        width=512,
        height=512,
        num_cells=20,
        num_apoptotic=5,
        num_mitotic=5,
        frames_per_cycle=16,  # Changed from 14 to 16
        frame_interval_seconds=5.0,
        num_cycles=125,
        exposure=50,
        intensity=2.0,
        cell_drift_speed=0.5,  # 0.5 pixels/frame minimal drift
    )
    generator.run_experiment()


if __name__ == "__main__":
    run_experiment_simple()
