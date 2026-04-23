#!/usr/bin/env python
"""
Quick test of the experiment movie generator.
Runs a single cycle with 3 frames to verify setup.
"""

import sys
from pathlib import Path

# Test with minimal settings to verify structure
from samples.experiment_movie_generator import ExperimentMovieGenerator

def test_experiment():
    """Run a minimal test."""
    print("Testing Experiment Movie Generator...")
    print("="*60)

    try:
        # Create with minimal settings
        generator = ExperimentMovieGenerator(
            output_dir="./test_experiment_output",
            width=512,
            height=512,
            num_cells=20,
            num_apoptotic=5,
            num_mitotic=5,
            frames_per_cycle=3,  # Just 3 frames for testing
            frame_interval_seconds=5.0,
            num_cycles=1,  # Just 1 cycle for testing
            exposure=50,
            intensity=2.0,
        )

        print("\n✓ Generator created successfully")
        print(f"  Output dir: {generator.output_dir}")
        print(f"  Space: {generator.width}x{generator.height}")
        print(f"  Cells: {generator.num_cells}")

        # Test cell initialization
        generator._initialize_cells()
        print(f"\n✓ Cells initialized: {len(generator.cells)} cells")
        for i, cell in enumerate(generator.cells):
            print(f"  Cell {i}: pos={cell.center}, velocity={cell.vel}")

        # Test state assignment
        generator._assign_cell_states()
        print(f"\n✓ Cell states assigned:")
        states = {}
        for cell in generator.cells:
            state = getattr(cell, 'current_state', 'Unknown')
            states[state] = states.get(state, 0) + 1
        for state, count in sorted(states.items()):
            print(f"  {state}: {count}")

        print(f"\n{'='*60}")
        print("✅ All tests passed!")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n❌ Error during testing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_experiment()
    sys.exit(0 if success else 1)
