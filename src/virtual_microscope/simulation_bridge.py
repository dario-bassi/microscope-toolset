from .microscope_sim_optimized import MicroscopeSimOptmized
import numpy as np

# Singleton to make available for all pyDevice the bridge
GLOBAL_BRIDGE = None

class SimulationBridge:
    """
    This class is a simulation bridge that connect the python device from UniMMcore
    to the simulation. In this way we differentiate totally any devices and we 
    will be able add pyDevice from within a configuration file. In addition, any other
    change in the simulation, will directly change in the SimulationBridge, without
    modifing the pyDevice.
    """
    # Half-width of the base 10x render in µm (512 px at 1 µm/px).
    _BASE_HALF = 256

    def __init__(self, microscope_sim : MicroscopeSimOptmized) -> None:

        if microscope_sim is None:
            raise ValueError("The microscope simulation must be initialized.")
        self._sim = microscope_sim
        self._current_slm_mask = None
        self._stage_position = (0.0, 0.0)

    
    def snap(self, exposure: float, brightness: float, **kwargs) -> np.ndarray:
        mask = self.get_slm_mask()
        return self._sim.snap_frame(mask=mask, exposure=exposure, intensity=brightness, *kwargs)
    
    def set_stage(self, x: float, y: float) -> None:
        """Stage position is the FOV center; convert to rendering origin (top-left of base 10x image)."""
        self._stage_position = (x, y)
        self._sim.camera_offset = np.array([x - self._BASE_HALF, y - self._BASE_HALF])

    def set_focus(self, z: float) -> None:
        self._sim.set_focal_plane(z)

    def update_state(self, dict_state: dict) -> None:
        self._sim.state_devices.update(dict_state)

    def set_slm_mask(self, mask: np.ndarray) -> None:
        """Called by slm device when pattern changes."""
        self._current_slm_mask = mask

    def get_slm_mask(self) -> np.ndarray:
        """Called by camera device when capturing."""
        if self._current_slm_mask is not None:
            return self._current_slm_mask
        # Default: no stimulation
        return np.zeros((self._sim.viewport_height, self._sim.viewport_width), dtype=bool)