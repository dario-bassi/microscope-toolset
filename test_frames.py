import time
import numpy as np
import time
from src.virtual_microscope.microscope_sim_optimized import MicroscopeSimOptmized
import cv2



class TimingContext:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.start) * 1000
        print(f"{self.name:30s}: {elapsed:6.2f} ms")

# Patch snap_frame to profile each section:

def snap_frame_profiled(self, mask=None, intensity=1.0, exposure=1.0):
    with TimingContext("Physics Update"):
        self.update(0.016)

    with TimingContext("Cell Behaviors"):
        for cell in self._cells:
            if hasattr(cell, "update_behavior"):
                cell.update_behavior(0.016)

    with TimingContext("Collision Detection"):
        self._handle_collisions_with_spatial_grid()

    with TimingContext("Rendering"):
        img = self.renderer.render_cells(
            self._cells, self.mode,
            tuple(self.camera_offset),
            self.focal_plane
        )

    with TimingContext("Post-processing"):
        img = (img.astype(np.float32) * intensity * 0.01 * exposure).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# Monkey patch for testing
MicroscopeSimOptmized.snap_frame_profiled = snap_frame_profiled

if __name__ == "__name__":
    # Usage:
    microscope = MicroscopeSimOptmized(nb_cells=300)
    for _ in range(10):
        microscope.snap_frame_profiled()