"""Spatial indexing for fast collision detection."""
import numpy as np
from typing import List, Tuple, Set
from .cell_base import CellBase


class SpatialGrid:
    """Spatial grid for efficient collision detection."""

    def __init__(self, width: float, height: float, cell_size: float = 50.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.grid_width = int(np.ceil(width / cell_size))
        self.grid_height = int(np.ceil(height / cell_size))
        self.grid = {}


    def clear(self) -> None:
        """Clear the grid"""
        self.grid.clear()

    def add_cell(self, cell: CellBase, index: int) -> None:
        """Add a cell to the grid."""
        grid_pos = self._get_grid_position(cell.center)

        # Add to multiple grid cells if cell is large
        radius = np.max(cell.r)
        cell_to_check = int(np.ceil(radius / self.cell_size))

        for dx in range(-cell_to_check, cell_to_check + 1):
            for dy in range(-cell_to_check, cell_to_check+1):
                gx = (grid_pos[0] + dx) % self.grid_width
                gy = (grid_pos[1] + dy) % self.grid_height
                key = (gx, gy)

                if key not in self.grid:
                    self.grid[key] = []
                
                self.grid[key].append(index)
    

    def get_potential_collisions(self, cell: CellBase, index: int) -> Set[int]:
        """Get indices of cells that might collide with given cell."""
        grid_pos = self._get_grid_position(cell.center)
        potential = set()

        radius = np.max(cell.r)
        cells_to_check = int(np.ceil(radius / self.cell_size))

        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                gx = (grid_pos[0] + dx) % self.grid_width
                gy = (grid_pos[1] + dy) % self.grid_height
                key = (gx, gy)

                if key in self.grid:
                    for other_index in self.grid[key]:
                        if other_index != index:
                            potential.add(other_index)

        return potential

    def _get_grid_position(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert position to grid coordinates."""
        gx = int(pos[0] / self.cell_size) % self.grid_width
        gy = int(pos[1] / self.cell_size) % self.grid_height
        return (gx, gy)