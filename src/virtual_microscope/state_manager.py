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
        
        A cell is ready to divide when the bridge between daughter cells
        has disappeared during cytokinesis (constriction >= 1.1).
        Only divides once per cycle.
        
        Args:
            cell: Cell to check for division readiness
            
        Returns:
            True if cell should divide, False otherwise
        """
        # Check if cell is ready to separate (bridge has disappeared)
        # This flag is set in _physic_cell_cycle() when constriction_progress >= 1.1
        is_ready = getattr(cell, '_ready_to_separate', False)
        has_not_divided_yet = not getattr(cell, '_division_occurred_this_cycle', False)
        within_max_divisions = cell.n_div < self.max_divisions
        
        should_divide = is_ready and has_not_divided_yet and within_max_divisions
        
        # CRITICAL: Mark as divided IMMEDIATELY to prevent re-division
        # This must happen before returning, so even if the flag is true,
        # it won't trigger again this cycle
        if should_divide:
            cell._division_occurred_this_cycle = True
        
        return should_divide
    
    def create_sister_cell(self, mother: CellCycleNormal) -> CellCycleNormal:
        """Create a daughter cell from a dividing mother cell.
        
        The mother cell geometry is reset from its constricted state back to normal.
        The daughter (sister) cell is created with identical initial conditions.
        Both cells are then independent objects ready to enter G1.
        
        Note: Division count is already incremented in cell_cycle._update_cell_div_count_and_flag_apoptotic_cell()
        
        Args:
            mother: The mother cell that is dividing
            
        Returns:
            New sister cell with reset G1 state and copied chromatin
        """
        # CRITICAL: Reset flags FIRST to prevent re-triggering division
        mother._ready_to_separate = False
        
        # Create sister cell using the convenience method
        sister = mother.copy_with_reset()
        
        # Reset mother cell geometry immediately after creating sister
        # This restores mother from its fully pinched state to normal
        mother.base_r = mother.original_base_r
        mother.r = mother.original_r.copy()
        mother.constriction_progress = 0.0
        
        # Update the original radii for the next cycle
        mother.original_r = mother.r.copy()
        
        # CRITICAL: Immediately transition mother to Interphase (G1) state
        # This prevents _physic_cell_cycle() from recalculating constriction in Cytokinesis state
        mother.current_time_life = 0
        mother.cell_mitosis_state = 'Interphase'
        mother.cell_cycle_state = 'G1'
        mother._last_transitioned_mitosis_state = None
        mother._last_transitioned_cycle_state = None
        mother._division_occurred_this_cycle = False
        
        # Track statistics (division count already incremented by cell cycle logic)
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




    