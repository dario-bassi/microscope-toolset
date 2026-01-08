"""
pymmcore_slm_sim.py

A virtual SLM device for pymmcore that simulates a spatial light modulator (SLM).
"""
import time
import numpy as np
from typing import ClassVar

from pymmcore_plus.experimental.unicore import UniMMCore
from pymmcore_plus.experimental.unicore.devices._slm import SLMDevice

class SimSLMDevice(SLMDevice):
    """Virtual SLM device for simulation."""

    WIDTH: ClassVar[int] = 512
    HEIGHT: ClassVar[int] = 512
    BYTES_PER_PIXEL: ClassVar[int] = 1  # 8-bit grayscale
    NAME: ClassVar[str] = "SimSLM"
    DESCRIPTION: ClassVar[str] = "Virtual SLM device for simulation."
    PIXEL_TYPE: ClassVar[str] = "GRAY8"
    VERSION: ClassVar[str] = "1.0"

    def __init__(self, core: UniMMCore | None = None, color: bool = False) -> None:
        super().__init__()
        self._exposure: float = 1000.0  # milliseconds
        self._color = color
        self._current_image: np.ndarray | None = None
        self._image_displayed = False
        self._core = core

    def get_name(self) -> str:
        return self.NAME

    def get_description(self) -> str:
        return self.DESCRIPTION

    def get_pixel_type(self) -> str:
        return self.PIXEL_TYPE

    def get_metadata_width(self) -> int:
        return self.WIDTH

    def get_metadata_height(self) -> int:
        return self.HEIGHT

    def get_version(self) -> str:
        return self.VERSION

    def get_width(self) -> int:
        return self.WIDTH

    def get_height(self) -> int:
        return self.HEIGHT

    def get_bytes_per_pixel(self) -> int:
        return self.BYTES_PER_PIXEL

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the SLM image buffer."""
        if self._color:
            return (512, 512, 3)  # Example color shape
        return (512, 512)         # Example grayscale shape

    def dtype(self):
        """Return the data type of the image buffer."""
        return np.uint8

    def set_image(self, pixels: np.ndarray) -> None:
        """Load the image into the SLM device adapter."""
        if pixels.shape != self.shape():
            raise ValueError(f"Image must be shape {self.shape()}")
        if pixels.dtype != self.dtype():
            raise ValueError(f"Image must be {self.dtype()}")
        self._current_image = pixels.copy()
        self._image_displayed = False

    def get_image(self) -> np.ndarray:
        """Get the current image from the SLM device adapter."""
        if self._current_image is None:
            raise RuntimeError("No image loaded")
        return self._current_image.copy()

    def display_image(self) -> None:
        """Command the SLM to display the loaded image."""
        if self._current_image is None:
            raise RuntimeError("No image loaded")
        self._image_displayed = True

    def set_exposure(self, interval_ms: float) -> None:
        """Command the SLM to turn off after a specified interval."""
        self._exposure = interval_ms
        self.core.events.SLMExposureChanged.emit(self.get_label(), interval_ms)

    def get_exposure(self) -> float:
        """Find out the exposure interval of an SLM."""
        return self._exposure

    def set_pixels_to(self, intensity: int) -> None:
        self._image.fill(intensity)
        self._displayed = True

    def set_pixels_to_rgb(self, red: int, green: int, blue: int) -> None:
        avg = int((int(red) + int(green) + int(blue)) / 3)
        self._image.fill(avg)
        self._displayed = True

    def get_number_of_components(self) -> int:
        return 1  # grayscale

# Example usage
def _example():
    core = UniMMCore()
    core.loadPyDevice("SLM", SimSLMDevice())
    core.initializeDevice("SLM")
    core.setSLMDevice("SLM")
    # Set a test pattern
    slm = core.getSLMDevice()
    print(f"SLM name: {slm}")
    core.setSLMExposure(99,slm)
    exposure = core.getSLMExposure(slm)
    print(f"SLM exposure set to {exposure} ms.")

if __name__ == "__main__":
    _example()
