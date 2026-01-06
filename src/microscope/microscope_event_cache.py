from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore
from collections import deque
from datetime import datetime
import threading
from typing import Any

class MicroscopeEventCache:
    """
    This class wants to keep all the signal that occurred from the napari-micromanager GUI
    and saved them in a cache
    """
    def __init__(self, mmc: CMMCorePlus | UniMMCore) -> None:
        self._mmc = mmc
        self._cache = deque(maxlen=1000) # if the length its limiting, we will change it
        self._lock = threading.Lock()

        # Connect pymmcore signals
        self._mmc.events.exposureChanged.connect(self._on_exposure_changed)
        self._mmc.events.XYStagePositionChanged.connect(self._on_xy_stage_position_changed)
        self._mmc.events.stagePositionChanged.connect(self._on_stage_position_changed)
        self._mmc.events.imageSnapped.connect(self._on_image_snapped)
        self._mmc.events.SLMExposureChanged.connect(self._on_slm_exposure_changed)
        self._mmc.events.autoShutterSet.connect(self._on_autoshutter_set)
        self._mmc.events.channelGroupChanged.connect(self._on_channel_group_changed)
        self._mmc.events.configDefined.connect(self._on_config_defined)
        self._mmc.events.configDeleted.connect(self._on_config_deleted)
        self._mmc.events.configGroupChanged.connect(self._on_config_group_changed)
        self._mmc.events.configGroupDeleted.connect(self._on_config_group_deleted)
        self._mmc.events.configSet.connect(self._on_config_set)
        self._mmc.events.propertiesChanged.connect(self._on_properties_changed)
        self._mmc.events.propertyChanged.connect(self._on_property_changed)
        self._mmc.events.roiSet.connect(self._on_roi_set)
        self._mmc.events.shutterOpenChanged(self._on_shutter_open_changed)
        # TO ADD

    def _add_events(self, event_type: str, data: dict, source: str = "Unknown"):
        with self._lock:
            self._cache.append({
                "time": datetime.now().isoformat(),
                "event_type": event_type,
                "data": data,
                "source": source # Either agent or user
            })
    def _on_exposure_changed(self, device: str, new_exposure: float):
        """Emit signal when the exposure changes."""
        self._add_events("exposure_changed", {f"{device}": f"{new_exposure}"})

    def _on_xy_stage_position_changed(self, name: str, xpos: float, ypos: float):
        """Emit signal when XY positions change."""
        self._add_events("xy_stage_position_changed", {"device": name, "new_x_pos": xpos, "new_y_pos": ypos})

    def _on_stage_position_changed(self, name: str, pos: float):
        """Emit signal when Z pos changes."""
        self._add_events("z_stage_position_changed", {"device": name, "new_z_pos": pos})

    def _on_image_snapped(self):
        """Emit signal when an image is snapped. --New-- in pymmcore-plus."""
        self._add_events("snap_image", {"action": "snapped image"})

    def _on_slm_exposure_changed(self, name: str, newExposure: float):
        """Emit signal when the exposure of the SLM device changes."""
        self._add_events("slm_exposure_changed", {"device": name, "new_exposure": newExposure})
    
    def _on_autoshutter_set(self, on: bool):
        """Emit signal when the auto shutter setting is changed."""
        self._add_events("autoshutter_setting_changed", {"autoshutter_settings": on})

    def _on_channel_group_changed(self, newChannelGroupName: str):
        """Emit signal when a channel group has changed."""
        self._add_events("channel_group_changed", {"new_channel_group_set": newChannelGroupName})

    def _on_config_defined(self, groupName: str, configName: str, deviceLabel: str, propName: str, value: Any):
        """Emit signal when a config is defined."""
        self._add_events("config_defined", {"groupName": groupName,"configName": configName, "deviceLabel": deviceLabel, "propName": propName, "value": value})

    def _on_config_deleted(self, groupName: str, configName: str):
        """Emit signal when a config is deleted."""
        self._add_events("config_deleted", {"groupName": groupName, "configName": configName})

    def _on_config_group_changed(self, groupName: str, newConfigName: str):
        """Emit signal when a config group name has changed."""
        self._add_events("config_group_changed", {"groupName": groupName, "newConfigName": newConfigName})

    def _on_config_group_deleted(self, group: str):
        """Emit signal when a config group was deleted."""
        self._add_events("config_group_deleted", {"config_group": group})

    def _on_config_set(self, groupName: str, configName: str):
        """Emit signal when a config has been set."""
        self._add_events("config_set", {"groupName": groupName, "configName": configName})

    def _on_properties_changed(self):
        """Emit signal with no arguments when properties have changed."""
        self._add_events("properties_changed", {"action": "Multiple properties have changed."})

    def _on_property_changed(self, name: str, propName: str, propValue: str):
        """Emit signal when a specific property has changed."""
        self._add_events("property_changed", {"device": name, "propName": propName, "propValue": propValue})

    def _on_roi_set(self, label: str, x: int, y: int, width: int, height: int):
        """Emit signal when an region of interst(ROI) is set."""
        self._add_events("region_of_interest_set", {"device": label, "x": x, "y" : y, "width" : width, "height" : height})

    def _on_shutter_open_changed(self, name: str, isOpen: bool):
        """Emit signal when the shutter open state has changed."""
        self._add_events("shutter_open_changed", {"device": name, "is_open" : isOpen})
    
    def _on_load_system_configuration(self):
        """Emit signal when the system configuration has been loaded."""
        self._add_events("loaded_system_configuration", {"action": "The file system configuration (.cfg) was loaded."})
            

    def get_recent_events(self, limit: int = 100):
        """
        Get recent events.
        """
        # TODO add filters for which events or keep all
        with self._lock:
            events = list(self._cache)

        return events[-limit:]
    
    def get_last_event(self):
        """
        Get most recent event.
        """
        events = self.get_recent_events(limit=1)
        # If there is not previous event just return None
        return events[0] if events else None

