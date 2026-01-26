"""
This class is needed to check the call of the microscope hardware
"""
from typing import Any, Callable, Optional, Tuple, List
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore
import hashlib
import json


class GatekeeperCore:
    """
    Shadow wrapper for a pymmcore-plus core instance.
    - Intercepts non-get/is calls and buffers them with optional predicates.
    - Provides snapshot/commit semantics and result caching for non-idempotent ops.
    """

    NON_IDEMPOTENT_OPS = {
        "snapImage", "snap", "getImage", "getLastImage",
        "startSequenceAcquisition", "stopSequenceAcquisition",
        "startContinuousSequenceAcquisition"
    }

    def __init__(self, mmc: CMMCorePlus | UniMMCore) -> None:
        self._mmc = mmc
        # each item: (function_name, args, kwargs, predicate_fn_or_None, cmd_hash)
        self.pending_changes: List[Tuple[str, tuple, dict, Optional[Callable[[], bool]], str, bool]] = []
        self._result_cache: dict[str, Any] = {}
        self._state_snapshot = None
        self._sequence_running = False

    
    def __getattr__(self, name: str) -> Any:
        """
        This is called when the agent call a function that is not defined in this class
        """
        # Get the function from the core
        attr = getattr(self._mmc, name)

        # Allowed functions
        if callable(attr):
            # For example: mmc.getConfigData()
            if name.startswith(('get', 'is')): # add more functions
                return attr
            

            # Any other function add into the pending changing list
            return self._create_intercept(name)
        
        return attr
    
    def _cmd_hash(self, function_name: str, args, kwargs) -> str:
        payload = {"fn": function_name, "args": args, "kwargs": kwargs}
        raw = json.dumps(payload, default=str, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()
    

    def _default_predicate(self, function_name: str, args, kwargs) -> Optional[Callable[[], bool]]:
        """
        Best-effort predicate: for common setter try to compare corresponding getter value.
        Return a callable that returns True if the action should be applied (i.e., current != desired).
        If unknown, return None
        """
        # NON-IDEMPONENT OPS; return None (cache only, never replay)
        if function_name in self.NON_IDEMPOTENT_OPS:
            return None
        
        # CONFIG/LIFECYCLE OPS: snapshot-dependentant
        if function_name == "loadSystemConfiguration":
            # Only load if snapshot indicates no configuration loaded yet
            def pred_load_config():
                # Always allow first load; if snapshot changes, abort commit
                return True
            
            return pred_load_config
        
        # unloadAllDevices
        # initializeDevice
        # initializeAllDevices
        # initializeDevice

        # IDEMPOTENT SETTERS WITH PREDICATES
        if function_name == "setProperty" and len(args) >= 3:
            device, prop, desired = args[0], args[1], args[2]

            def pred_property():
                try:
                    current = self._mmc.getProperty(device, prop)
                    try:
                        return float(current) != float(desired)
                    except Exception:
                        return str(current) != str(desired)
                except Exception:
                    return True
                
            return pred_property
        
        if function_name == "setExposure" and len(args) >= 2:
            device, desired = args[0], args[1]
            def pred_exposure():
                try:
                    current = self._mmc.getExposure()
                    if current is None:
                        return True
                    return abs(float(current) - float(desired)) >= 1e-3
                except Exception:
                    return True
                
            return pred_exposure
        
        # setConfig
        # setConfig(group, config) -> getConfig(group)
        if function_name == "setConfig" and len(args) >= 2:
            group, desired = args[0], args[1]
            def pred_config():
                try:
                    current = self._mmc.getCurrentConfig(group)
                    return current != desired
                except Exception:
                    return True
            return pred_config
        
        # setCameraDevice
        if function_name == "setCameraDevice" and len(args) >= 1:
            desired = args[0]
            def pred_camera():
                try:
                    current = self._mmc.getCameraDevice()
                    return current != desired
                except Exception:
                    return True
                
            return pred_camera
        
        # setZ
        if function_name in ('setPosition', 'setZPosition') and len(args) >= 1:
            desired = args[0]
            z_getter = "getZPosition" if function_name == "setZPosition" else "getPosition"
            def pred_z():
                try:
                    current = getattr(self._mmc, z_getter)()
                    return abs(float(current) - float(desired)) > 1e-3
                except Exception:
                    return True
            return pred_z
        
        # setXYPosition
        if function_name == "setXYPosition" and len(args) >= 2:
            desired = (args[0], args[1])
            def pred_xy():
                try:
                    current = self._mmc.getXYPosition()
                    tol = 1e-3
                    return any(abs(float(c) - float(d)) > tol for c, d, in zip(current, desired))
                except Exception:
                    return True
                
            return pred_xy


        # Example: setExposure -> getExposure
        if function_name.startswith("set") and len(function_name) > 3:
            get_name = "get" + function_name[3:]
            if hasattr(self._mmc, get_name):
                desired = args[0] if args else kwargs.get("value")

                def pred():
                    try:
                        current = getattr(self._mmc, get_name)()
                        # numeric tolerance for floats
                        try:
                            return float(current) != float(desired)
                        except Exception:
                            return current != desired
                    except Exception:
                        return True
                    
                return pred
        
        # default: unknown -> always apply
        return None
    

    def _create_intercept(self, function_name):
        def wrapper(*args, **kwargs):
            cmd_hash = self._cmd_hash(function_name, args, kwargs)
            pred = self._default_predicate(function_name, args, kwargs)
            is_non_idempotent = function_name in self.NON_IDEMPOTENT_OPS
            # store predicate so commit will call function only if predicate() is True
            # register the changes
            self.pending_changes.append((function_name, args, kwargs, pred, cmd_hash, is_non_idempotent))
            print(f"Shadow: Intercepted {function_name}{args}")

            # If we have a cahed result for this command (non-idempotent executed earlier), return it
            return self._result_cache.get(cmd_hash)
        
        return wrapper
    

    def snapshot_state(self, snapshot_fn: Optional[Callable[[Any], Any]] = None) -> Any:
        """
        Capture a lightweight microscope state
        Custom snapshot_fn(mmcmock) -> hashable can be provided.
        Default snapshot: tuple of (config name, device list) when available.
        """
        if snapshot_fn is not None:
            self._state_snapshot = snapshot_fn(self._mmc)
            return self._state_snapshot
        
        snapshot_dict = {"type": "microscope_snapshot"}

        try:
            # System configuration
            complete_dict={}
            for x in self._mmc.getAvailableConfigGroups():
                tmp_dict={}
                for y in self._mmc.getAvailableConfigs(x):
                    configuration = self._mmc.getConfigData(x,y).dict()
                    tmp_dict[y] = configuration
                    complete_dict[x]=tmp_dict
            
            snapshot_dict["config_state"] = complete_dict
        except Exception:
            snapshot_dict["config_state"] = None

        
        try:
            devices = self._mmc.getLoadedDevices()
            snapshot_dict["devices"] = tuple(devices)
        except Exception:
            snapshot_dict["devices"] = ()

        # Focus (Z) device
        try:
            focus_device = self._mmc.getFocusDevice()
            snapshot_dict["focus_device"] = focus_device
        except Exception:
            snapshot_dict["focus_device"] = None

        # XY stage device
        try:
            xy_device = self._mmc.getXYStageDevice()
            snapshot_dict["xy_device"] = xy_device
        except Exception:
            snapshot_dict["xy_device"] = None

        # Camera device
        try:
            camera_device = self._mmc.getCameraDevice()
            snapshot_dict["camera_device"] = camera_device
        except Exception:
            snapshot_dict["camera_device"] = None

        # Current XY position (rounded for snapshot stability)
        try:
            xy = self._mmc.getXYPosition()
            snapshot_dict["xy_position"] = (round(xy[0], 2), round(xy[1], 2))
        except Exception:
            snapshot_dict["xy_position"] = None

        # Current Z position (rounded)
        try:
            z_get = getattr(self._mmc, "getZPosition", None) or getattr(self._mmc, "getPosition", None)
            if z_get:
                z = z_get()
                snapshot_dict["z_position"] = round(z, 2)
            else:
                snapshot_dict["z_position"] = None
        except Exception:
            snapshot_dict["z_position"] = None

        # Sequence running state
        snapshot_dict["sequence_running"] = self._sequence_running


        # Convert to hashable tuple
        self._state_snapshot = tuple(sorted(snapshot_dict.items()))
        return self._state_snapshot
    
    def commit(self, real_mmc: Optional[CMMCorePlus | UniMMCore] = None, check_snapshot: Optional[Any] = None):
        """
        Replay pending changes against real_mmc (or self._mmc if not provided).
        - check_snapshot: if provided, abort if it doesn't match saved snapshot.
        - predicates are evaluated before executing each command; if predicate returns False, command skipped.
        - results of executed commands are cached by command hash.
        """
        target = real_mmc or self._mmc
        # Verify snapshot hasn't changed
        if check_snapshot is not None and self._state_snapshot is not None:
            if check_snapshot != self._state_snapshot:
                raise RuntimeError("Microscope state changed; aborting commit.")
            
        for (function_name, args, kwargs, pred, cmd_hash, is_non_idempotent) in list(self.pending_changes):
            try:
                # Check if command result is already cached (skip re-execution)
                if cmd_hash in self._result_cache:
                    self.pending_changes.remove((function_name, args, kwargs, pred, cmd_hash, is_non_idempotent))
                    continue

                need_apply = True
                if pred is not None:
                    try:
                        # predicate return True is we need apply (current != desired)
                        need_apply = pred()
                    except Exception:
                        need_apply = True
                
                if not need_apply:
                    # drop from queue
                    try:
                        self.pending_changes.remove((function_name, args, kwargs, pred, cmd_hash, is_non_idempotent))
                    except ValueError:
                        pass
                    continue

                func = getattr(target, function_name)
                res = func(*args, **kwargs)
                # cache the result so repeated identical commands can return the same value
                self._result_cache[cmd_hash] = res

                # Update sequence state tracking
                if function_name in ("startSequenceAcquisition", "startContinuousSequenceAcquisition"):
                    self._sequence_running = True
                elif function_name in ("stopSequenceAcquisition"):
                    self._sequence_running = False

                self.pending_changes.remove((function_name, args, kwargs, pred, cmd_hash, is_non_idempotent))
            except Exception as e:
                # on failure, abort commit and leave prending changes to be handled later
                raise RuntimeError(f"Commit failed for {function_name}: {e}")
            
        return True
    


    def clear_pending(self):
        self.pending_changes.clear()
        self._result_cache.clear()
    

    # Override specific functions Here