from src.virtual_microscope.cell_normal import NormalCell
import numpy as np
from typing import Literal, Optional
import random

class CellCycleNormal(NormalCell):
    """Cell that loops through a cell cycle indefinitely"""

    def __init__(self, *args, initial_state: Optional[Literal['G1', 'S', 'G2', 'M']] = None,
                 initial_time: Optional[int] = None, initial_divisions: Optional[int] = None,
                 copy_chromatin_from: Optional['CellCycleNormal'] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # This cell has no fluorescence
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence = np.zeros(self.vertices)

        # G1, S, G2 -> interphase
        # M -> prophase, metaphase, anaphase, telophase
        # M -> cytokinesis
        if initial_state is not None:
            self.cell_cycle_state: Literal['M', 'G1', 'G2', 'S'] = initial_state  # type: ignore
        else:
            self.cell_cycle_state: Literal['M', 'G1', 'G2', 'S'] = self._initial_cycle_state()
            
        self.cell_mitosis_state: Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase'] = self._initial_mitosis_state()
        
        if initial_divisions is not None:
            self.n_div: int = initial_divisions
        else:
            self.n_div: int = self._initial_number_division()
            
        self.is_dying: bool = self._initialization_apoptosis()
        self.max_nb_div: int = 10
        self.time_tot_cycle: int = 660 # in seconds
        self.time_table_cycle: dict[str, int] = {'G1': 240, 'S': 360, 'G2': 480} # in seconds - if its too long halb this time.
        self.time_table_mitosis: dict[str, int] = {'P': 516, 'Met': 552, 'A': 588, 'T': 624, 'C': 660}
        # add random start time point for each cell
        if initial_time is not None:
            self.current_time_life: int = initial_time
        else:
            self.current_time_life: int = self._initial_random_time_life()
        # chromatin pts - store as offsets from center
        if copy_chromatin_from is not None:
            self.chromatin_offset = copy_chromatin_from.chromatin_offset.copy()
        else:
            self.chromatin_offset = self._initial_chromatin_offsets()
        self.chromatin_pts = self._update_chromatin_pts()
        # Physics state for telophase
        self.base_r_at_telophase = None
        # Death tracking
        self.death_timer: int = 0
        self.apoptosis_death_phase: Literal['Shrinkage', 'Blebbing', 'Apoptotic bodies', 'Phagocytosis'] = 'Shrinkage'
        self.time_table_apoptois: dict[str, int] = {'Shrinkage': 20, 'Blebbing': 40, 'Apoptotic bodies': 50, 'Phagocytosis': 60}
        self.max_death_timer: int = 60
        self.remove_this_cell = False


    def _initial_chromatin_offsets(self) -> list[tuple[float, float]]:
        """Creates chromatin control points as offsets from center"""
        # Create 3 random control points relative to center
        offsets = []
        for _ in range(3):
            # Random angle and distance for circular distribution
            angle = random.uniform(0, 2 * np.pi)
            r = self.base_r * np.sqrt(random.uniform(0, 0.8))  # Stay away from edges
            x = r * np.cos(angle)  # Offset, not absolute
            y = r * np.sin(angle)  # Offset, not absolute
            offsets.append((x, y))

        return offsets
    
    def _update_chromatin_pts(self) -> list[tuple[float, float]]:
        """Update chromatin points based on current center position"""
        return [
            (self.center[0] + offset[0], self.center[1] + offset[1])
            for offset in self.chromatin_offset
        ]
    
    def _initial_cycle_state(self) -> Literal['M', 'G1', 'G2', 'S']:
        """Randomly selecte a state for a cell."""
        cell_cycle_dict = {0: 'M', 1: 'G1', 2:'G2', 3: 'S'}
        int_rand = random.randint(0,3)

        return cell_cycle_dict[int_rand]  # type: ignore
    
    def _initial_mitosis_state(self) -> Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']:
        """Randomly select a state from the Mitosis state"""
        if self.cell_cycle_state in ('G1', 'G2', 'S'):
            return 'Interphase'
        else:
            mitosis_dict = {0:'Cytokinesis', 1:'Prophase', 2:'Metaphase', 3:'Anaphase', 4:'Telophase'}

            int_rand = random.randint(0, 4)

            
            return mitosis_dict[int_rand]  # type: ignore
        
    def _initial_number_division(self) -> int:
        """Randomly select number of division for a cell."""
        int_rand = random.randint(0,self.max_nb_div) # at initialization cell can be in apoptosis state

        return int_rand
    
    def _initialization_apoptosis(self) -> bool:
        """Flag all dying cell at start"""
        if self.n_div == 10:
            return True
        else:
            return False
        
    def _initial_random_time_life(self) -> int:
        """Randomly select time life of the cell"""
        cell_cycle_state = self.cell_cycle_state
        cell_mitosis_state = self.cell_mitosis_state # not in M, then this is None

        # time range of each state
        # G1(240): 0 -> 240
        # S(120): 241 -> 360
        # G2(120): 361 -> 480
        # M(180): 481 -> 516, 517 -> 552, 553 -> 588, 589 -> 624, 625 -> 660
        match cell_cycle_state:
            case 'G1':
                return random.randint(0, 240)
            case 'S':
                return random.randint(241, 360)
            case 'G2':
                return random.randint(361, 480)
            case 'M': # M
                match cell_mitosis_state:
                    case 'Prophase':
                        return random.randint(481, 516)
                    case 'Metaphase':
                        return random.randint(517, 552)
                    case 'Anaphase':
                        return random.randint(553, 588)
                    case 'Telophase':
                        return random.randint(589, 624)
                    case 'Cytokinesis':  # Cytokinesis or Interphase
                        return random.randint(625, 660)
                    case _:
                        return -1 # undefined
            case _:
                return -1 # undefined

    def _change_state(self) -> None:
        """Change the state of the cell."""
        if self.cell_mitosis_state == 'Interphase':
            if self.current_time_life > self.time_table_cycle[self.cell_cycle_state]:
                transiction_dict = {'G1': 'S', 'S': 'G2', 'G2': 'M'}
                # update state
                self.cell_cycle_state = transiction_dict[self.cell_cycle_state]  # type: ignore
                # update immediately from G2  to M
                if self.cell_cycle_state == 'M':
                    self.cell_mitosis_state = 'Prophase'

        else: # M
            if self.current_time_life > self.time_table_mitosis[self.cell_mitosis_state]:
                transiction_dict = {'Prophase':'Metaphase', 'Metaphase':'Anaphase', 'Anaphase':'Telophase', 'Telophase':'Cytokinesis', 'Cytokinesis':'Interphase'}

                # update mitotic state
                self.cell_mitosis_state = transiction_dict[self.cell_mitosis_state]  # type: ignore
                # Reset telophase tracker when entering telophase
                if self.cell_mitosis_state == 'Telophase':
                    self.base_r_at_telophase = self.base_r
                # update time for new starting cycle
                if self.cell_mitosis_state == 'Interphase':
                    self.current_time_life = 0
                    self.base_r_at_telophase = None
                    self._update_cell_div_count_and_flag_apoptotic_cell() # update cell count

    def _update_cell_div_count_and_flag_apoptotic_cell(self) -> None:
        """Check the division count for the cell"""
        # Division complete - increment division count
        self.n_div += 1
        # Flag as apoptotic if reached max divisions
        if self.n_div >= self.max_nb_div:
            self.is_dying = True

    def _start_apoptosis(self) -> bool:
        """Start signal for apoptosis."""
        if self.is_dying:
            return True
        
        return False
    
    def update_behavior(self, dt: float) -> None:
        """Update cell cycle state and chromatin positions."""
        # Skip normal physics if dying
        if not self.is_dying:
            super().update_behavior(dt)
        
        # Update death timer if dying
        if self.is_dying:
            self.death_timer += int(dt)
            self._update_apoptosis_phase()
            self._update_apoptotic_physics()
        
        # Update state cycle (G1 -> S -> G2 -> M) only if not dying
        if not self.is_dying:
            self._change_state()
            self.current_time_life += int(dt)

        # Update chromatin positions to follow cell center
        self.chromatin_pts = self._update_chromatin_pts()

        # Update cell cycle physics
        if not self.is_dying:
            self._physic_cell_cycle()

    def _physic_cell_cycle(self):
        """Update physics based on cell cycle state."""
        if self.cell_mitosis_state == 'Telophase':
            # Gradually grow to 2x area during telophase (sqrt(2) ~ 1.41x radius)
            if self.base_r_at_telophase is not None and hasattr(self, 'time_table_mitosis'):
                # Calculate progress through telophase
                time_in_telophase = self.current_time_life - self.time_table_mitosis['T']
                telophase_duration = self.time_table_mitosis['C'] - self.time_table_mitosis['T']
                progress = min(time_in_telophase / telophase_duration, 1.0)
                
                # Linear growth: start at 1x, end at sqrt(2) ~ 1.41x (doubled area)
                growth_factor = 1.0 + 0.41 * progress
                self.base_r = self.base_r_at_telophase * growth_factor

    def copy_with_reset(self) -> 'CellCycleNormal':
        """Create a sister cell with reset cycle state but copied chromatin.
        
        Used during cell division to create a daughter cell that:
        - Starts in G1 phase (timer=0)
        - Has 0 divisions completed
        - Inherits mother's chromatin pattern
        - Inherits velocity but position wraps naturally
        """
        sister = CellCycleNormal(
            width=self.width,
            height=self.height,
            base_radius=self.base_r,
            vertices=self.vertices,
            seed=self.seed + 1000,  # Different seed for variation
            initial_state='G1',
            initial_time=0,
            initial_divisions=0,
            copy_chromatin_from=self
        )
        # Copy position and velocity from mother
        sister.center = self.center.copy()
        sister.vel = self.vel.copy()
        return sister
    

    def _update_apoptosis_phase(self) -> None:
        """Update the current apoptotic phase"""
        
        transiction_dict = {'Shrinkage': 'Blebbing', 'Blebbing': 'Apoptotic bodies', 'Apoptotic bodies': 'Phagocytosis'}

        if self.death_timer > self.time_table_apoptois[self.apoptosis_death_phase]:

            if self.apoptosis_death_phase == 'Phagocytosis':
                # signal cell remove.
                self.remove_this_cell = True
            else:
                self.apoptosis_death_phase = transiction_dict[self.apoptosis_death_phase] # type: ignore


    def _get_shrinkage_progess(self) -> float:
        """Returns the progress through 'Shrinkage' phase (0-1)."""
        shrinkage_duration = self.time_table_apoptois['Shrinkage']
        progress = min(self.death_timer / shrinkage_duration, 1.0)
        return progress

    def _get_current_shrinkage_factor(self) -> float:
        """Returns radius multiplier of cell shrinkage (1.0 -> 0.7)."""
        progress = self._get_shrinkage_progess()
        # Linear shrinkage: start at 1.0, end at 0.7 (30% shrinkage)
        shrinkage_factor = 1.0 - (0.3 * progress)
        return shrinkage_factor

    def _get_apoptotic_body_positions(self) -> list[tuple[float, float]]:
        """Generate 3-5 body positions scattered linearly from center.
        
        Bodies scatter within 0.5 of the original cell radius from center.
        """
        num_bodies = random.randint(3, 5)
        original_radius = self.base_r
        max_scatter_radius = original_radius * 0.5
        
        positions = []
        for i in range(num_bodies):
            # Spread linearly around center
            angle = (i / num_bodies) * 2 * np.pi + random.uniform(-0.2, 0.2)
            radius = random.uniform(0, max_scatter_radius)
            
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            positions.append((x, y))
        
        return positions

    def _initialize_apoptotic_bodies(self) -> None:
        """Initialize 'Apoptotic Bodies' phase to set up nucleus fragments.
        
        Store original radius and generate body positions.
        """
        if not hasattr(self, 'apoptotic_body_positions'):
            self.apoptotic_body_positions = self._get_apoptotic_body_positions()
        if not hasattr(self, 'original_base_r_for_apoptosis'):
            self.original_base_r_for_apoptosis = self.base_r
    
    def _update_apoptotic_physics(self) -> None:
        """Disable physics for apoptotic cells.
        
        - Stop velocity
        - Disable brownian motion
        - Initialize apoptotic bodies when entering that phase
        """
        # Stop all movement
        self.vel[:] = 0.0
        
        # Apply shrinkage factor during Shrinkage phase
        if self.apoptosis_death_phase == 'Shrinkage':
            shrinkage_factor = self._get_current_shrinkage_factor()
            # Scale base_r for shrinkage (will affect nucleus size proportionally)
            if not hasattr(self, 'original_base_r_for_apoptosis'):
                self.original_base_r_for_apoptosis = self.base_r
            self.base_r = self.original_base_r_for_apoptosis * shrinkage_factor
        
        # Initialize apoptotic bodies when entering that phase
        if self.apoptosis_death_phase == 'Apoptotic bodies':
            self._initialize_apoptotic_bodies()