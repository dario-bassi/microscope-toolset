from pymmcore_plus.experimental.unicore import StageDevice
from pymmcore_plus import FocusDirection
import src.virtual_microscope.simulation_bridge as bridge_module

class SimZStageDevice(StageDevice):

    def __init__(self):
        super().__init__()
        self._z_current = 0.0
        self._z_old = 0.0
        self._z_origin = 0.0 # see later
        self._direction = 0
        #self._microscope_sim = microscope_sim
        self.bridge = bridge_module.GLOBAL_BRIDGE

    def home(self) -> None:
        self._z_current=0.0
        self._z_old=0.0
        self.update_z_camera()

    def stop(self) -> None:
        """
        Stop the movement of the stage
        """
        return

    def set_origin(self) -> None:
        """Set the current position as origin"""
        # TODO
        # to change -> see tests
        self._z_current = 0.0
        self._z_old = 0.0
        self.update_z_camera()

    def set_position_um(self, val: float) -> None:
        self._z_old = self._z_current
        self._z_current = val
        # update sim
        self.update_z_camera()
        # update focus direction
        self._direction = self._calculate_direction()
        self.core.events.stagePositionChanged.emit(self.get_label(), val)

    def get_position_um(self) -> float:
        return self._z_current

    def update_z_camera(self):
        #self._microscope_sim.set_focal_plane(self._z_current)
        self.bridge.set_focus(self._z_current)

    def _calculate_direction(self):
        """Calculate the direction of the stage"""
        return self._z_current - self._z_old

    def get_focus_direction(self) -> FocusDirection:
        """Translate the movement of the z stage"""
        if self._direction > 0.0:
            return FocusDirection.FocusDirectionTowardSample
        else:
            return FocusDirection.FocusDirectionAwayFromSample

    def set_focus_direction(self, sign) -> None:
        """Set the focus direction of the stage
        if sign > 0.0 direction towards sample
        if sign < 0.0 direction away from sample
        """
        pass
        # update z pos
        #new_position = self._z_current + sign
        #self.set_position_um(new_position)
        if sign > 0.0:
            self._direction = FocusDirection.FocusDirectionTowardSample
        if sign < 0.0:
            self._direction = FocusDirection.FocusDirectionAwayFromSample





