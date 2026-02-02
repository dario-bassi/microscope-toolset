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
        self.death_timer: float = 0.0


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

        if cell_cycle_state == 'G1':
           return random.randint(0, 240)
        elif cell_cycle_state == 'S':
            return random.randint(241, 360)
        elif cell_cycle_state == 'G2':
            return random.randint(361, 480)
        else: # M
            if cell_mitosis_state == 'Prophase':
                return random.randint(481, 516)
            elif cell_mitosis_state == 'Metaphase':
                return random.randint(517, 552)
            elif cell_mitosis_state == 'Anaphase':
                return random.randint(553, 588)
            elif cell_mitosis_state == 'Telophase':
                return random.randint(589, 624)
            else:  # Cytokinesis or Interphase
                return random.randint(625, 660)


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

    def _start_apoptosis(self):
        """Start signal for apoptosis."""
        raise NotImplementedError()
    
    def update_behavior(self, dt: float) -> None:
        """Update cell cycle state and chromatin positions."""
        super().update_behavior(dt)
        
        # Update death timer if dying
        if self.is_dying:
            self.death_timer += dt
        
        # Update state cycle (G1 -> S -> G2 -> M)
        self._change_state()
        self.current_time_life += int(dt)

        # Update chromatin positions to follow cell center
        self.chromatin_pts = self._update_chromatin_pts()

        # Update cell cycle physics
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
    