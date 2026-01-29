from src.virtual_microscope.cell_normal import NormalCell
import numpy as np
from typing import Literal


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
        self.cell_mitosis_state: Literal['Cytokinesis', 'Interphase', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']
        self.n_div: int
        self.is_dying: bool
        self.cell_cycle_state: Literal['M', 'G1', 'G2', 'S']
        self.max_nb_div: int = 10
        self.time_tot_cycle: int = 660 # in seconds
        self.time_table_cycle: dict[str, int] = {'M': 180, 'G1': 240, 'S': 120, 'G2': 120} # in seconds



    def _change_state(self):
        pass

    def _start_division(self):
        pass

    def _start_apoptosis(self):
        pass


    def _physic_cell_division(self):
        pass


    def _physic_cell_cycle(self):
        pass

    def update_cell_state(self):
        pass

    # add function if needed
    