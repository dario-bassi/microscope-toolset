"""Utilities for inspecting Micro-Manager .cfg files."""
import re
from typing import Literal

# Matches '#py pyDevice,<name>,<module>,<class>' — Python virtual device line
_PY_DEVICE  = re.compile(r'^\s*#py\s+pyDevice\s*,')
# Matches 'Device,<name>,<library>,<adapter>' — standard C++ driver line
_CPP_DEVICE = re.compile(r'^\s*Device\s*,')


def classify_cfg(cfg_path: str) -> Literal["real", "virtual", "mixed"]:
    """Classify a .cfg file by the type of devices it registers.

    Scans only device-loading lines:
      '#py pyDevice,...'  → Python device (UniMMCore / virtual simulation)
      'Device,...'        → C++ device   (standard Micro-Manager driver)

    Returns:
      'virtual' - all device-loading lines use #py (pure Python devices)
      'real'    - all device-loading lines are C++ Device entries, or the
                  file contains no device-loading lines at all
      'mixed'   - both kinds are present (not supported yet)

    On any read error the file is treated as 'real' so napari-micromanager
    can handle it normally.
    """
    has_py  = False
    has_cpp = False
    try:
        with open(cfg_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if _PY_DEVICE.match(line):
                    has_py = True
                elif _CPP_DEVICE.match(line):
                    has_cpp = True
                if has_py and has_cpp:
                    break
    except Exception:
        return "real"

    if has_py and has_cpp:
        return "mixed"
    if has_py:
        return "virtual"
    return "real"
