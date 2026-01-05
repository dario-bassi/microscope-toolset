"""Normal cell without special behaviors."""
import numpy as np
from src.virtual_microscope.cell_base import CellBase


class NormalCell(CellBase):
    """Normal cell that only responds to physics and collisions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Normal cells have slightly different physics parameters
        self.friction = 4.0  # Slightly higher friction
        self.brownian_d = 60.0  # Less random movement
        self.curvature_relax = 0.08  # More stable shape
        self.radial_relax = 0.03
        self.ruffle_std = 0.04  # Less membrane ruffling
        
        # Normal cells can have varying fluorescence
        self.has_nucleus_marker = np.random.random() > 0.5  # 50% have nucleus fluorescence
        self.has_membrane_marker = np.random.random() > 0.5  # 50% have membrane fluorescence
        
        # Set fluorescence based on markers
        if self.has_nucleus_marker:
            self.nucleus_fluorescence = 0.7 + np.random.random() * 0.3  # Variable brightness
        else:
            self.nucleus_fluorescence = 0.0
            
        if self.has_membrane_marker:
            self.membrane_fluorescence = np.full(self.vertices, 0.7 + np.random.random() * 0.3)
        else:
            self.membrane_fluorescence = np.zeros(self.vertices)
    
    def respond_to_collision(self, other: 'CellBase') -> None:
        """Normal cells can deform slightly when collided."""
        # Find the direction of collision
        dvec = other.center - self.center
        dvec[0] -= self.width * np.round(dvec[0] / self.width)
        dvec[1] -= self.height * np.round(dvec[1] / self.height)
        
        if np.linalg.norm(dvec) > 0:
            # Deform slightly away from collision
            collision_angle = np.arctan2(dvec[1], dvec[0])
            
            # Find vertices facing the collision
            for i, angle in enumerate(self.angles):
                angle_diff = abs(angle - collision_angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                    
                if angle_diff < np.pi / 3:  # Vertices within 60 degrees
                    # Compress these vertices slightly
                    self.r[i] *= 0.95
            
            # Ensure constraints
            self.r = np.clip(self.r, 0.4 * self.base_r, 2.2 * self.base_r)
            self._conserve_area()
    
    def update_behavior(self, dt: float) -> None:
        """Normal cells have no special behavior, but can slowly return to round shape."""
        # Slowly return to circular shape
        self.r += 0.01 * (self.base_r - self.r) * dt
        
    def can_divide(self) -> bool:
        """Normal cells could potentially divide (for future implementation)."""
        # Could implement cell division based on time or size
        return False