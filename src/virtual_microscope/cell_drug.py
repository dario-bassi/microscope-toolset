import numpy as np
from src.virtual_microscope.cell_base import CellBase


class DrugResponseCell(CellBase):
    """Cell with drug response capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drug_concentration = 0.0
        self.drug_response_rate = 0.1
        self.max_response = 2.0

    def apply_drug(self, concentration: float, drug_type: str = "growth") -> None:
        """Apply drug effects based on concentration and type"""

        self.drug_concentration = concentration

        if drug_type == "growth":
            self._apply_growth_drug()
        elif drug_type == "mobility":
            self._apply_mobility_drug()
        elif drug_type == "apoptosis":
            self._apply_apoptosis_drug()


    def _apply_growth_drug(self) -> None:
        """Simulate growth factor drug effects."""
        growth_factor = 1.0 + self.drug_concentration * self.drug_response_rate # to check if maths is correct
        growth_factor = min(growth_factor, self.max_response)

        # Increase cell size
        self.base_r *= growth_factor
        self.r *= growth_factor
        self.area0 = np.pi * self.base_r ** 2

    def _apply_mobility_drug(self) -> None:
        """Simulate mobility-enhancing drug effects."""
        mobility_factor = 1.0 + self.drug_concentration * self.drug_response_rate * 5
        self.brownian_d *= mobility_factor
        self.friction /= mobility_factor


    def _apply_apoptosis_drug(self) -> None:
        """Simulate apoptotic drug effects."""
        shrink_factor = max(0.1, 1.0 - self.drug_concentration * self.drug_response_rate)
        self.base_r *= shrink_factor
        self.r *= shrink_factor
        self.area0 = np.pi * self.base_r ** 2

