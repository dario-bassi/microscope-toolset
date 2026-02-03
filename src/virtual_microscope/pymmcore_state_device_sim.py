from pymmcore_plus.experimental.unicore import StateDevice
import src.virtual_microscope.simulation_bridge as bridge_module

"""
This class contains all the State Device of the microscope simulation
"""

class FilterWheelDevice(StateDevice):
    """This is a Filter Wheel device"""

    def __init__(self) -> None:

        super().__init__({
            0:"Electra1(402/454)", 
            1:"SCFP2(434/474)",
            2:"TagGFP2(483/506)", 
            3:"obeYFP(514/528)", 
            4:"mRFP1-Q667(549/570)", 
            5:"mScarlet3(569/582)", 
            6:"miRFP670(642/670)"
        })
        self._current_state = 0 # default position
        self._current_label = self._state_to_label.get(self._current_state)
        self._name = "Filter Wheel"

        self.bridge = bridge_module.GLOBAL_BRIDGE

        self.update_microscope_simulation()

    def get_state(self) -> int:
        """
        Return the current state of the filter wheel
        """
        return self._current_state

    def set_state(self, position: int | str) -> None:
        """
        Set the current state of the filter wheel
        """
        if isinstance(position, str):
            position = int(position)

        # if self._current_state != position:
        #     self._current_state = position
        #     # update microscope stateDevice
        #     self.update_microscope_simulation()
        #if self._current_state != position:
        self._current_state = position
        self._current_label = self._state_to_label.get(self._current_state)
        # update microscope stateDevice
        self.update_microscope_simulation()


    def update_microscope_simulation(self) -> None:
        """
        Update the states of the virtual microscope simulation
        """
        # Only update if bridge is available
        if self.bridge is not None:
            self.bridge.update_state({self._name : {"state": str(self._current_state), "label": self._current_label}})



class LEDDevice(StateDevice):
    """This is a LED device"""
    def __init__(self) -> None:
        
        super().__init__({
            0:"UV", 
            1:"BLUE", 
            2:"CYAN", 
            3:"GREEN", 
            4:"YELLOW", 
            5:"ORANGE", 
            6:"RED"
        })
        self.bridge = bridge_module.GLOBAL_BRIDGE
        self._current_state = 0 # default position
        self._current_label = self._state_to_label.get(self._current_state)
        self._name = "LED"

        # Only update if bridge is available
        if self.bridge is not None:
            self.update_microscope_simulation()

    def get_state(self) -> int:
        """
        Return the current state of the filter wheel
        """
        return self._current_state

    def set_state(self, position: int | str) -> None:
        """
        Set the current state of the filter wheel
        """
        if isinstance(position, str):
            position = int(position)

        # if self._current_state != position:
        #     self._current_state = position
        #     # update microscope stateDevice
        #     self.update_microscope_simulation()
        #if self._current_state != position:
        self._current_state = position
        self._current_label = self._state_to_label.get(self._current_state)
        # update microscope stateDevice
        self.update_microscope_simulation()


    def update_microscope_simulation(self) -> None:
        """
        Update the states of the virtual microscope simulation
        """
        # Only update if bridge is available
        if self.bridge is not None:
            self.bridge.update_state({self._name : {"state": str(self._current_state), "label": self._current_label}})



class ObjectiveDevice(StateDevice):
    """This is a objective device"""
    def __init__(self) -> None:

        super().__init__({
            0: "10x", 
            1: "20x", 
            2:"40x"
        })
        self._current_state = 0 # default position
        self._current_label = self._state_to_label.get(self._current_state)
        self._name = "Objective"

        self.bridge = bridge_module.GLOBAL_BRIDGE

        # Only update if bridge is available
        if self.bridge is not None:
            self.update_microscope_simulation()

    def get_state(self) -> int:
        """
        Return the current state of the filter wheel
        """
        return self._current_state

    def set_state(self, position: int | str) -> None:
        """
        Set the current state of the filter wheel
        """
        if isinstance(position, str):
            position = int(position)

        # if self._current_state != position:
        #     self._current_state = position
        #     # update microscope stateDevice
        #     self.update_microscope_simulation()
        #if self._current_state != position:
        self._current_state = position
        self._current_label = self._state_to_label.get(self._current_state)
        # update microscope stateDevice
        self.update_microscope_simulation()


    def update_microscope_simulation(self) -> None:
        """
        Update the states of the virtual microscope simulation
        """
        # if self._name in self._microscope_sim.state_devices.keys():
        #     print("Used")
        #     self._microscope_sim.state_devices[self._name]["state"] = str(self._current_state)
        #     self._microscope_sim.state_devices[self._name]["label"] = self._current_label

        # self._microscope_sim.state_devices[self._name]["state"] = str(self._current_state)
        # self._microscope_sim.state_devices[self._name]["label"] = self._current_label
        #self._microscope_sim.state_devices.update({self._name : {"state": str(self._current_state), "label": self._current_label}})
        # Only update if bridge is available
        if self.bridge is not None:
            self.bridge.update_state({self._name : {"state": str(self._current_state), "label": self._current_label}})