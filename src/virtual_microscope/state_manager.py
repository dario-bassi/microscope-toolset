"""Cell cycle state management and population dynamics."""
from typing import List, Optional, TYPE_CHECKING
from src.virtual_microscope.cell_cycle import CellCycleNormal

if TYPE_CHECKING:
    from src.virtual_microscope.microscope_sim_optimized import MicroscopeSimOptmized


class CellCycleManager:
    """Manages cell division, apoptosis, and population dynamics.
    
    Responsibilities:
    - Check which cells are ready to divide
    - Create sister cells with reset state and copied chromatin
    - Track cell death and manage removal timing
    - Maintain population statistics
    """
    
    def __init__(self, max_divisions: int = 10, track_stats: bool = True, enable_g0: bool = False):
        """Initialize the cell cycle manager.
        
        Args:
            max_divisions: Maximum number of divisions a cell can undergo (default 10)
            track_stats: Whether to track population statistics (births, deaths, etc)
            enable_g0: Whether to enable G0 (quiescent) phase for non-cycling cells
        """
        self.max_divisions = max_divisions
        self.track_stats = track_stats
        self.enable_g0 = enable_g0
        
        # Statistics tracking
        self.n_births = 0
        self.n_deaths = 0
        self.peak_population = 0
        self.total_divisions = 0
    
    def should_divide(self, cell: CellCycleNormal) -> bool:
        """Check if a cell is ready to divide.
        
        A cell is ready to divide when:
        - Cytokinesis is complete (entering G1 state from M phase)
        
        Args:
            cell: Cell to check for division readiness
            
        Returns:
            True if cell should divide, False otherwise
        """
        # Division happens at the transition from Cytokinesis to G1
        # Check if we just completed cytokinesis
        is_in_g1 = cell.cell_cycle_state == 'G1'
        is_interphase = cell.cell_mitosis_state == 'Interphase'
        is_early_g1 = cell.current_time_life < 10  # Within first 10 seconds of G1
        is_post_cytokinesis = (
            is_in_g1 and is_interphase and is_early_g1 and
            cell.n_div < self.max_divisions
        )
        
        return is_post_cytokinesis
    
    def create_sister_cell(self, mother: CellCycleNormal) -> CellCycleNormal:
        """Create a daughter cell from a dividing mother cell.
        
        The mother cell is reset to G1 state. The daughter (sister) cell is created
        with identical initial conditions.
        
        Args:
            mother: The mother cell that is dividing
            
        Returns:
            New sister cell with reset G1 state and copied chromatin
        """
        # Create sister cell using the convenience method
        sister = mother.copy_with_reset()
        
        # Increment mother's division count (she's now divided)
        mother.n_div += 1
        
        # Track statistics
        if self.track_stats:
            self.total_divisions += 1
            self.n_births += 1
        
        return sister
    
    def update(self, cells: List[CellCycleNormal], simulation: 'MicroscopeSimOptmized') -> None:
        """Update cell population: handle divisions and deaths.
        
        This should be called once per simulation timestep. It:
        1. Identifies cells ready to divide and creates sister cells
        2. Identifies cells past max death timer and removes them
        3. Updates population statistics
        
        Args:
            cells: List of cells in the simulation (will be modified)
            simulation: Reference to the main simulation (for state access)
        """
        cells_to_add = []
        cells_to_remove = []
        
        # Check each cell
        for i, cell in enumerate(cells):
            # Check for division
            if self.should_divide(cell):
                sister = self.create_sister_cell(cell)
                cells_to_add.append(sister)
            
            # Check for death removal (when apoptosis is complete)
            if hasattr(cell, 'remove_this_cell') and cell.remove_this_cell:
                cells_to_remove.append(i)
                if self.track_stats:
                    self.n_deaths += 1
        
        # Add new cells to the population
        for sister in cells_to_add:
            cells.append(sister)
        
        # Remove dead cells (iterate in reverse to maintain indices)
        for i in reversed(cells_to_remove):
            cells.pop(i)
        
        # Update peak population
        if self.track_stats:
            self.peak_population = max(self.peak_population, len(cells))




    