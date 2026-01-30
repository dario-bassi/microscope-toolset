from src.virtual_microscope.cell_normal import NormalCell
import numpy as np
from typing import Literal, Optional
import random

class CellCycleNormal(NormalCell):
    """Cell that loops through a cell cycle indefinitely"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # This cell has no fluorescence
        self.nucleus_fluorescence = 0.0
        self.membrane_fluorescence = np.zeros(self.vertices)

        # G1, S, G2 -> interphase
        # M -> prophase, metaphase, anaphase, telophase
        # M -> cytokinesis
        self.cell_cycle_state: Literal['M', 'G1', 'G2', 'S'] = self._initial_cycle_state()
        self.cell_mitosis_state: Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase'] = self._initial_mitosis_state()
        self.n_div: int = self._initial_number_division()
        self.is_dying: bool = self._initialization_apoptosis()
        self.max_nb_div: int = 10
        self.time_tot_cycle: int = 660 # in seconds
        self.time_table_cycle: dict[str, int] = {'G1': 240, 'S': 360, 'G2': 480} # in seconds
        self.time_table_mitosis: dict[str, int] = {'P': 516, 'Met': 552, 'A': 588, 'T': 624, 'C': 660}
        # add random start time point for each cell
        self.current_time_life: int = self._initial_random_time_life()
        # chromatin pts
        self.chromatin_pts = self._initial_chromatin_creation()


    def _initial_chromatin_creation(self) -> list[tuple[float, float]]:
        """Creates long chromatin uncondensed"""
        # Create 3 random control points within the circle
        pts = []
        for _ in range(3):
            # Random angle and distance for circular distribution
            angle = random.uniform(0, 2 * np.pi)
            r = self.r * np.sqrt(random.uniform(0, 0.8)) # Stay away from edges
            x = int(self.center[0] + r * np.cos(angle))
            y = int(self.center[1] + r * np.sin(angle))
            pts.append((x, y))

        return pts
    
    def _initial_cycle_state(self) -> str:
        """Randomly selecte a state for a cell."""
        cell_cycle_dict = {0: 'M', 1: 'G1', 2:'G2', 3: 'S'}
        int_rand = random.randint(0,3)

        return cell_cycle_dict[int_rand]
    
    def _initial_mitosis_state(self) -> str:
        """Randomly select a state from the Mitosis state"""
        if self.cell_cycle_state in ('G1', 'G2', 'S'):
            return 'Interphase'
        else:
            mitosis_dict = {0:'Cytokinesis', 1:'Prophase', 2:'Metaphase', 3:'Anaphase', 4:'Telophase'}

            int_rand = random.randint(0, 4)

            
            return mitosis_dict[int_rand]
        
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
            elif cell_mitosis_state == 'Cytokinesis':
                return random.randint(625, 660)





    def _change_state(self) -> None:
        """Change the state of the cell."""
        if self.cell_mitosis_state == 'Interphase':
            if self.current_time_life > self.time_table_cycle[self.cell_cycle_state]:
                transiction_dict = {'G1': 'S', 'S': 'G2', 'G2': 'M'}
                # update state
                self.cell_cycle_state = transiction_dict[self.cell_cycle_state]
                # update immediately from G2  to M
                if self.cell_cycle_state == 'M':
                    self.cell_mitosis_state = 'Prophase'

        else: # M
            if self.current_time_life > self.time_table_mitosis[self.cell_mitosis_state]:
                transiction_dict = {'Prophase':'Metaphase', 'Metaphase':'Anaphase', 'Telophase':'Cytokenesis', 'Cytokenesis':'Interphase'}

                # update mitotic state
                self.cell_mitosis_state = transiction_dict[self.cell_mitosis_state]
                # update time for new starting cycle
                if self.cell_mitosis_state == 'Interphase':
                    self.current_time_life = 0


    def _start_apoptosis(self):
        """Start signal for apoptosis."""
        raise NotImplementedError()


    def _physic_cell_division(self):
        pass


    def _physic_cell_cycle(self):
        pass

    def update_cell_state(self):
        pass

    # add function if needed
    