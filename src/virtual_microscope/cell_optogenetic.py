"""Optogenetic cell behaviour module."""
import numpy as np
from src.virtual_microscope.cell_base import CellBase



class OptogeneticCell(CellBase):

    def __init__(self, *args, protrusion_gain: float = 0.05, impulse: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.protrusion_gain = protrusion_gain
        self.impulse = impulse
        self.is_stimulated = False


    def stimulate(self, mask: np.ndarray) -> None:
        """Apply optogenetic stimulation based on mask."""
        if mask is None or not mask.any():
            self.is_stimulated = False
            return
        

        # Get vertex position
        vertices = self.vertices_positions

        # Check which vertices fall on stimulated pixels
        inside = ((vertices[:, 0] >= 0) & (vertices[:, 0] < mask.shape[1]) & 
                  (vertices[:, 1] >= 0) & (vertices[:, 1] < mask.shape[0]))
        
        if not inside.any():
            self.is_stimulated = False
            return
        
        # Get pixels indices for vertices inside bounds
        ix = vertices[inside, 0].astype(int)
        iy = vertices[inside, 1].astype(int)

        # Check masl at vertex position
        hit = mask[iy, ix]

        if not hit.any():
            self.is_stimulated = False
            return
        
        self.is_stimulated = True

        # Find which vertices to protrude
        idx = np.where(inside)[0][hit]

        # Apply protrusion
        self.r[idx] += self.protrusion_gain * self.base_r
        self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
        self._conserve_area()

        # Apply impulse toward stimulated region
        hit_vertices = vertices[idx]
        target = np.mean(hit_vertices, axis=0)
        direction = target - self.center
        norm = np.linalg.norm(direction)

        if norm > 0:
            self.vel += (direction/norm) * self.impulse