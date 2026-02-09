import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable, Optional

from pymmcore_plus import PropertyType
import numpy as np
from numpy.typing import DTypeLike
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
from pymmcore_plus.experimental.unicore import pymm_property
import src.virtual_microscope.simulation_bridge as bridge_module

class SimCameraDevice(CameraDevice):
    """
    pymmcore_camera_sim.py

    A virtual camera device for pymmcore that generates images using the microscope_sim.py simulation.
    """
    _exposure: float = 50.0
    _brightness: float = 1.0
    _binning: int = 2 # default 2x2
    _mask: Optional[np.ndarray] = None
    _led_channel: str = None
    _filter_wheel_channel: str = None

    def __init__(self) -> None:

        super().__init__()
        #if microscope_sim is None:
        #    raise RuntimeError('microscope_sim must be provided')
        self.bridge = bridge_module.GLOBAL_BRIDGE
        self._mask = None
        # change limits of binning
        self.set_property_limits("Binning", (0, 20))
        #self.set_property_sequence_max_length(Keyword.Exposure, 10)

    def _get_bridge(self):
        """Get bridge, updating from global if necessary"""
        if self.bridge is None:
            self.bridge = bridge_module.GLOBAL_BRIDGE
        return self.bridge

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure
        self.core.events.exposureChanged.emit(self.get_label(), exposure)

    def shape(self) -> tuple[int, int]:
        # Use the simulation's dimensions
        # change it with the viewpoint
        # if self._sim.n_channel == 2:
        #     return self._sim.viewport_height, self._sim.viewport_width
        # else:
        #     return self._sim.viewport_height, self._sim.viewport_width, self._sim.n_channel
        bridge = self._get_bridge()
        if bridge is None:
            return 512, 512  # Default fallback dimensions
        return bridge._sim.viewport_height, bridge._sim.viewport_width

    def dtype(self) -> DTypeLike:
        return np.uint8

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self._mask = mask

    # Minimum interval between frames (seconds) to avoid starving the Qt main
    # thread.  The rendering holds the GIL, so without a gap the UI freezes.
    _MIN_FRAME_INTERVAL: float = 0.08  # ~12 FPS max

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:

        count = 0
        while n is None or count < n:
            t0 = time.perf_counter()
            bridge = self._get_bridge()
            self._mask = bridge.get_slm_mask() # type: ignore
            surf = bridge.snap(brightness=self._brightness, exposure=self._exposure) # type: ignore

            buf = get_buffer(self.shape(), self.dtype())
            buf[:] = surf

            yield {
                "data": buf,
                "timestamp": time.time()
                }
            count += 1

            # Throttle: ensure minimum interval so the main thread gets GIL time
            elapsed = time.perf_counter() - t0
            remaining = self._MIN_FRAME_INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # define property brightness
    @pymm_property(
        limits=(0.0,100.0),
        sequence_max_length=100,
        name="brightness",
        property_type=PropertyType.Float
    )
    def brightness(self) -> float:
        """
        Get the brightness of the virtual camera.
        """
        return self._brightness

    # setter methods
    @brightness.setter
    def brightness(self, value: float) -> None:
        """
        Send the values to the virtual hardware to update the brightness.
        """
        self._brightness = value

    @brightness.sequence_loader
    def _load_brightness_sequence(self, sequence: Sequence[float]) -> None:
        print(f"Loading brightness sequence: {sequence}")

    @brightness.sequence_starter
    def _start_brightness_sequence(self) -> None:
        print("Starting brightness sequence")

    def get_binning(self) -> int:
        """
        Return the current binning of the virtual camera.
        """
        return self._binning

    def set_binning(self, binning: int) -> None:
        """
        Set the current binning of the virtual camera.
        """
        self._binning = binning
    #
    # def load_exposure_sequence(self, prop_name: str, sequence: Sequence[float]) -> None:
    #     self._exposure_sequence = tuple(sequence)
    #
    # def start_exposure_sequence(self) -> None:
    #     self._exposure_sequence_started = True
    #
    # def stop_exposure_sequence(self) -> None:
    #     self._exposure_sequence_stopped = True




def test():
    # Example usage
    core = UniMMCore()
    core.loadPyDevice("Camera", SimCameraDevice())
    core.initializeDevice("Camera")
    core.setCameraDevice("Camera")
    core.setExposure(42)

    try:
        from pymmcore_widgets import ExposureWidget, ImagePreview, LiveButton, SnapButton
        from qtpy.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget

        app = QApplication([])

        window = QWidget()
        window.setWindowTitle("Sim Microscope Camera Example")
        layout = QVBoxLayout(window)

        top = QHBoxLayout()
        top.addWidget(SnapButton(mmcore=core))
        top.addWidget(LiveButton(mmcore=core))
        top.addWidget(ExposureWidget(mmcore=core))
        layout.addLayout(top)
        layout.addWidget(ImagePreview(mmcore=core))
        window.setLayout(layout)
        window.resize(800, 600)
        window.show()
        app.exec()
    except Exception:
        print("run `pip install pymmcore-widgets[image] PyQt6` to run the GUI example")
        core.snapImage()
        image = core.getImage()
        print("Image shape:", image.shape)
        print("Image dtype:", image.dtype)
        print("Image data:", image)