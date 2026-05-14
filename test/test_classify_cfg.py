"""Tests for src/utils/cfg_utils.classify_cfg."""

from utils import classify_cfg

# ── helpers ──────────────────────────────────────────────────────────────────


def _write(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ── virtual (pure #py pyDevice) ──────────────────────────────────────────────


def test_virtual_bacteria_style(tmp_path):
    cfg = _write(
        tmp_path,
        "bacteria.cfg",
        """\
# bacteria backend
#py pyDevice,SimServer,virtual_microscope.devices.sim_server,SimServer
#py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice
#py pyDevice,XYStage,virtual_microscope.devices.stage,SimStageDevice
#py Property,Core,Camera,Camera
""",
    )
    assert classify_cfg(cfg) == "virtual"


def test_virtual_old_style(tmp_path):
    """Old-style virtual_normal.cfg — has some non-#py Property lines but no C++ Device lines."""
    cfg = _write(
        tmp_path,
        "virtual_normal.cfg",
        """\
# Reset
Property,Core,Initialize,0
#py pyDevice,Camera,src.virtual_microscope.pymmcore_camera_sim,SimCameraDevice
#py pyDevice,XYStage,src.virtual_microscope.pymmcore_stage_sim,SimStageDevice
Property,Core,Initialize,1
Property,Core,AutoShutter,1
#py Property,Core,Camera,Camera
""",
    )
    assert classify_cfg(cfg) == "virtual"


def test_virtual_only_comments_and_properties(tmp_path):
    """No device lines at all → treated as real (safe default)."""
    cfg = _write(
        tmp_path,
        "props_only.cfg",
        """\
# Just some properties
Property,Core,Initialize,0
Property,Core,AutoShutter,1
""",
    )
    assert classify_cfg(cfg) == "real"


# ── real (pure C++ Device lines) ─────────────────────────────────────────────


def test_real_demo_config(tmp_path):
    cfg = _write(
        tmp_path,
        "demo.cfg",
        """\
# Micro-Manager Demo config
Property,Core,Initialize,0
Device,Camera,DemoCamera,DCam
Device,XYStage,DemoCamera,DXYStage
Device,ZStage,DemoCamera,DZStage
Property,Core,Camera,Camera
""",
    )
    assert classify_cfg(cfg) == "real"


def test_real_no_device_lines(tmp_path):
    """Empty file → real."""
    cfg = _write(tmp_path, "empty.cfg", "")
    assert classify_cfg(cfg) == "real"


def test_real_comments_only(tmp_path):
    cfg = _write(
        tmp_path,
        "comments.cfg",
        """\
# This is just a comment file
# Nothing here
""",
    )
    assert classify_cfg(cfg) == "real"


# ── mixed (both #py pyDevice and Device) ─────────────────────────────────────


def test_mixed_py_and_cpp(tmp_path):
    cfg = _write(
        tmp_path,
        "mixed.cfg",
        """\
# Hypothetical mixed config
Device,Camera,DemoCamera,DCam
#py pyDevice,ZStage,virtual_microscope.devices.z_stage,SimZStageDevice
""",
    )
    assert classify_cfg(cfg) == "mixed"


def test_mixed_cpp_first(tmp_path):
    """Order shouldn't matter — mixed is detected regardless."""
    cfg = _write(
        tmp_path,
        "mixed2.cfg",
        """\
#py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice
Device,ZStage,DemoCamera,DZStage
""",
    )
    assert classify_cfg(cfg) == "mixed"


# ── edge cases ───────────────────────────────────────────────────────────────


def test_missing_file():
    """Unreadable / missing file → safe default 'real'."""
    assert classify_cfg("/nonexistent/path/to/config.cfg") == "real"


def test_device_word_in_comment_not_counted(tmp_path):
    """A comment mentioning 'Device' should not be counted as a C++ device line."""
    cfg = _write(
        tmp_path,
        "comment_device.cfg",
        """\
# Device,Camera,DemoCamera,DCam  ← this is commented out
#py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice
""",
    )
    assert classify_cfg(cfg) == "virtual"


def test_leading_whitespace(tmp_path):
    """Lines with leading spaces/tabs are still matched."""
    cfg = _write(
        tmp_path,
        "indented.cfg",
        """\
   #py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice
    Device,XYStage,DemoCamera,DXYStage
""",
    )
    assert classify_cfg(cfg) == "mixed"
