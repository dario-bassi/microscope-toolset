"""
This class is needed to check the call of the microscope hardware
"""
from typing import Any
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore


class GatekeeperCore:

    def __init__(self, mmc: CMMCorePlus | UniMMCore) -> None:
        self._mmc = mmc
        self.pending_changes = []

    
    def __getattribute__(self, name: str) -> Any:
        """
        This is called when the agent call a function that is not defined in this class
        """
        # Get the function from the core
        attr = getattr(self._mmc, name)

        # Allowed functions
        if callable(attr):
            # For example: mmc.getConfigData()
            if name.startswith('get') or name.startswith('is'): # add more functions
                return attr
            

            # Any other function add into the pending changing list
            return self._create_intercept(name)
        
        return attr
    

    def _create_intercept(self, function_name):
        def wrapper(*args, **kwargs):
            # register the changes
            self.pending_changes.append((function_name, args, kwargs))
            print(f"Shadow: Intercepted {function_name}{args}")

            return None # Agent thinks that the code worked!
        
        return wrapper
    

    # Override specific functions Here