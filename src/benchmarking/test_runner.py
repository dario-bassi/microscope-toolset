"""Test runner for benchmarking experiments.

Each test lives in src/benchmarking/test_<N>/ and defines its configuration
in one of three ways:

  test.yaml only          — pure YAML test; backend create_sim() is called
                            with all non-runner kwargs from the file.
  test.yaml + initialize_test.py — hybrid; YAML supplies the config dict,
                            initialize_test.py provides create_sim_override().
  initialize_test.py only — legacy Python test; defines TEST_CONFIG dict and
                            optionally create_sim_override().

This module validates the config, pre-creates the simulation, sets
GLOBAL_BRIDGE so SimServer skips its own sim creation, and generates a
dynamic .cfg file with the user-facing channel names.

Usage from plugin_napari.py:
    cfg_path = run_test("test_1")
    MCPServer(auto_config=str(cfg_path))

CLI (for quick sanity-check):
    python -m src.benchmarking.test_runner test_1
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locate test modules
# ---------------------------------------------------------------------------

_BENCHMARKING_DIR = Path(__file__).parent
_YAML_FILENAME = "test.yaml"

# Runner-level keys that are extracted explicitly in run_test() and must not
# be forwarded to backend create_sim() as unknown kwargs.
_RUNNER_KEYS = frozenset({
    "title", "backend", "focal_plane", "channels",
    "cell_type", "phase_contrast", "slm",
    "time_scale", "tick_hz", "initial_properties",
})


class _TestSpec:
    """Duck-typed object returned for YAML-based tests.

    Satisfies the interface expected by run_test() and test_server.py:
      .TEST_CONFIG          — dict (same structure as Python TEST_CONFIG)
      .__doc__              — task description string
      .create_sim_override  — optional callable (only present when a Python hook exists)
    """

    def __init__(self, config: dict, doc: str = "", override_fn=None):
        self.TEST_CONFIG = config
        self.__doc__ = doc
        if override_fn is not None:
            self.create_sim_override = override_fn


def _load_yaml_config(yaml_file: Path) -> dict:
    """Parse a test.yaml file and return the config dict.

    Validates required fields and normalises initial_properties rows from
    YAML lists to tuples (matching the Python TEST_CONFIG convention).
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for YAML test configs.  "
            "Install with: pip install pyyaml"
        ) from exc

    with yaml_file.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"{yaml_file}: expected a YAML mapping at the top level")
    if "backend" not in data:
        raise ValueError(f"{yaml_file}: required field 'backend' is missing")
    if "channels" not in data:
        raise ValueError(f"{yaml_file}: required field 'channels' is missing")

    # YAML loads [[a,b,c], ...] as list-of-lists; convert to list-of-tuples
    # so they match the Python convention and unpack cleanly in _generate_cfg.
    if "initial_properties" in data:
        data["initial_properties"] = [tuple(row) for row in data["initial_properties"]]

    return data


def _load_py_module(test_name: str, init_file: Path):
    """Import initialize_test.py from disk (used for legacy and hybrid tests)."""
    spec = importlib.util.spec_from_file_location(f"test_{test_name}", init_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_test_module(test_name: str):
    """Load a test specification — YAML-first with Python fallback.

    Resolution order for ``src/benchmarking/<test_name>/``:

    1. **test.yaml only** — pure YAML test.  Config comes from YAML;
       ``create_sim()`` is called with the non-runner kwargs.
    2. **test.yaml + initialize_test.py** — hybrid test.  YAML supplies the
       config; the Python file must define ``create_sim_override()`` which
       receives control instead of the standard ``create_sim()`` call.
    3. **initialize_test.py only** — legacy Python test (unchanged behaviour).
    4. **Neither** — raises ``FileNotFoundError``.
    """
    test_dir = _BENCHMARKING_DIR / test_name
    yaml_file = test_dir / _YAML_FILENAME
    init_file = test_dir / "initialize_test.py"

    has_yaml = yaml_file.exists()
    has_py   = init_file.exists()

    if not has_yaml and not has_py:
        raise FileNotFoundError(
            f"Test not found: {test_dir}\n"
            f"Create test.yaml or initialize_test.py with a TEST_CONFIG dict."
        )

    if has_yaml:
        cfg = _load_yaml_config(yaml_file)
        doc = cfg.pop("description", "")
        override_fn = None
        if has_py:
            py_mod = _load_py_module(test_name, init_file)
            # Inject the YAML config into the module namespace so that
            # create_sim_override() can reference TEST_CONFIG without the
            # Python file needing to duplicate it.
            py_mod.TEST_CONFIG = cfg
            if hasattr(py_mod, "create_sim_override"):
                override_fn = py_mod.create_sim_override
        return _TestSpec(cfg, doc=doc, override_fn=override_fn)

    # Legacy: Python-only path
    return _load_py_module(test_name, init_file)


def list_tests() -> list[dict]:
    """Return metadata for every test_* folder that has a test.yaml or initialize_test.py."""
    results = []
    for test_dir in sorted(_BENCHMARKING_DIR.glob("test_*")):
        if not test_dir.is_dir():
            continue
        has_yaml = (test_dir / _YAML_FILENAME).exists()
        has_py   = (test_dir / "initialize_test.py").exists()
        if not has_yaml and not has_py:
            continue
        entry = {"name": test_dir.name, "title": "", "backend": "", "channels": []}
        try:
            module = _load_test_module(test_dir.name)
            cfg: dict = getattr(module, "TEST_CONFIG", {})
            entry["title"] = cfg.get("title", "")
            entry["backend"] = cfg.get("backend", "")
            entry["channels"] = [ch.get("name", "") for ch in cfg.get("channels", [])]
        except Exception as e:
            entry["title"] = f"(could not load: {e})"
        results.append(entry)
    return results


def print_tests() -> None:
    """Print all available tests to stdout."""
    tests = list_tests()
    if not tests:
        print("No tests found in src/benchmarking/test_*/")
        return
    print("Available tests:\n")
    for t in tests:
        title = f"  {t['title']}" if t["title"] else ""
        channels = ", ".join(t["channels"]) if t["channels"] else "—"
        print(f"  [{t['name']}]{title}")
        if t["backend"]:
            print(f"    backend: {t['backend']}  |  channels: {channels}")
    print(f"\nUsage: python -m src.plugin_napari --test <name>")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_channels(channels: list[dict], sim) -> None:
    """Raise ValueError if any channel's (filter, led) pair is not in the sim's mode_map."""
    mode_map: dict = getattr(sim, "_mode_map", {})
    if not mode_map:
        return  # no mode_map → backend is phase-contrast only, skip

    for ch in channels:
        name = ch.get("name", "<unnamed>")
        filt = ch.get("filter")
        led = ch.get("led")
        if filt is None or led is None:
            raise ValueError(f"Channel '{name}' is missing 'filter' or 'led' keys.")
        key = (filt, led)
        if key not in mode_map:
            valid = ", ".join(str(k) for k in mode_map)
            raise ValueError(
                f"Channel '{name}': (filter={filt!r}, led={led!r}) not in mode_map.\n"
                f"Valid pairs: {valid}"
            )


# ---------------------------------------------------------------------------
# Dynamic .cfg generation
# ---------------------------------------------------------------------------

_CFG_HEADER = """\
# Auto-generated test configuration — do not edit
# Generated by src/benchmarking/test_runner.py
#py pyDevice,SimServer,virtual_microscope.devices.sim_server,SimServer
#py Property,SimServer,Backend,{backend}
#py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice
#py pyDevice,XYStage,virtual_microscope.devices.stage,SimStageDevice
#py pyDevice,ZStage,virtual_microscope.devices.z_stage,SimZStageDevice
#py pyDevice,LED,virtual_microscope.devices.state,LEDDevice
#py pyDevice,Filter Wheel,virtual_microscope.devices.state,FilterWheelDevice
#py pyDevice,Shutter,virtual_microscope.devices.shutter,SimShutterDevice
#py pyDevice,Objective,virtual_microscope.devices.state,ObjectiveDevice

#py Property,Core,Camera,Camera
#py Property,Core,XYStage,XYStage
#py Property,Core,Focus,ZStage
#py Property,Core,Shutter,Shutter

#py Property,Core,Initialize,1
"""

_PHASE_CONTRAST_CHANNEL = """\
#py ConfigGroup,Channel,phase-contrast,LED,Label,CYAN
#py ConfigGroup,Channel,phase-contrast,Filter Wheel,Label,Electra1(402/454)
"""

_SLM_DEVICE = """\
#py pyDevice,SLM,virtual_microscope.devices.slm,SimSLMDevice
#py Property,Core,SLM,SLM
"""


def _generate_cfg(backend: str, channels: list[dict], output_dir: Path,
                  cell_type: str = "normal", phase_contrast: bool = False,
                  slm: bool = False,
                  initial_properties: list[tuple[str, str, str]] | None = None) -> Path:
    """Write a .cfg with custom channel names and return its path.

    The filename uses the pattern ``virtual_<cell_type>.cfg`` so that
    ``_extract_cell_type()`` in mcp_server_gui.py returns the right value
    when pre-initialising the legacy SimulationBridge.

    ``initial_properties`` is an optional list of (device, property, value)
    tuples appended as ``#py Property,<device>,<property>,<value>`` lines
    after channels — e.g. [("Objective", "Label", "40x")].
    """
    lines = [_CFG_HEADER.format(backend=backend)]
    if slm:
        lines.append(_SLM_DEVICE)
    if phase_contrast:
        lines.append(_PHASE_CONTRAST_CHANNEL)
    for ch in channels:
        name = ch["name"]
        led = ch["led"]
        filt = ch["filter"]
        lines.append(f"#py ConfigGroup,Channel,{name},LED,Label,{led}\n")
        lines.append(f"#py ConfigGroup,Channel,{name},Filter Wheel,Label,{filt}\n")

    if initial_properties:
        for device, prop, value in initial_properties:
            lines.append(f"#py Property,{device},{prop},{value}\n")

    cfg_text = "".join(lines)
    cfg_path = output_dir / f"virtual_{cell_type}.cfg"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_test(test_name: str) -> Path:
    """Set up a test simulation and return a cfg path ready for MCPServer.

    Steps:
    1. Load TEST_CONFIG from initialize_test.py
    2. Create the simulation with custom params
    3. Set focal_plane (out-of-focus start if specified)
    4. Validate that all channels exist in the sim's mode_map
    5. Set virtual_microscope GLOBAL_BRIDGE so SimServer skips its own init
    6. Generate a .cfg file with the user-facing channel names
    7. Return the cfg path

    The caller (plugin_napari.py) passes the returned path to MCPServer(auto_config=).
    """
    module = _load_test_module(test_name)

    cfg: dict[str, Any] = getattr(module, "TEST_CONFIG", None)
    if cfg is None:
        raise AttributeError(
            f"initialize_test.py in '{test_name}' must define a TEST_CONFIG dict."
        )

    backend: str = cfg.get("backend", "particle")
    focal_plane: float = cfg.get("focal_plane", 0.0)
    channels: list[dict] = cfg.get("channels", [])

    # --- Create simulation ---
    # If initialize_test.py defines create_sim_override(), use it directly.
    # This is the escape hatch for backends whose create_sim() silently drops
    # kwargs (e.g. particle drops cell_type).
    # Otherwise call the backend's standard create_sim(**sim_kwargs).
    if hasattr(module, "create_sim_override"):
        sim = module.create_sim_override()
    else:
        import importlib as _il
        sim_kwargs = {k: v for k, v in cfg.items() if k not in _RUNNER_KEYS}
        backend_mod = _il.import_module(f"virtual_microscope.backends.{backend}")
        sim = backend_mod.create_sim(**sim_kwargs)

    # Apply out-of-focus start (tissue_z stays at 0.0; focal_plane offset creates blur)
    sim.set_focal_plane(focal_plane)

    # Validate channels against the sim's mode_map
    _validate_channels(channels, sim)

    # Pre-set GLOBAL_BRIDGE so SimServer.initialize() skips creating its own sim
    from virtual_microscope.engine.simulation_bridge import SimulationBridge, set_global_bridge
    from virtual_microscope.engine.realtime import RealtimeEngine
    bridge = SimulationBridge(sim)
    set_global_bridge(bridge)

    # SimServer.initialize() returns early when bridge is pre-set, so start the
    # engine here instead.  time_scale and tick_hz are read from TEST_CONFIG so
    # each test can tune simulation speed independently.
    # Default time_scale=0.05 preserves existing cell-cycle timing (9 min/cycle).
    if getattr(sim, 'continuous', False) and hasattr(sim, 'step'):
        time_scale: float = cfg.get("time_scale", 0.05)
        tick_hz: int = cfg.get("tick_hz", 10)
        engine = RealtimeEngine(sim, time_scale=time_scale, tick_hz=tick_hz,
                                idle_timeout=30.0, bridge=bridge)
        engine.patch_snap_frame()
        engine.start()
        bridge._engine = engine

    # Generate cfg in a persistent temp dir (survives until process exits)
    cell_type: str = cfg.get("cell_type", "normal")
    phase_contrast: bool = cfg.get("phase_contrast", False)
    slm: bool = cfg.get("slm", False)
    initial_properties: list = cfg.get("initial_properties", [])
    output_dir = _BENCHMARKING_DIR / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _generate_cfg(backend, channels, output_dir, cell_type=cell_type,
                             phase_contrast=phase_contrast, slm=slm,
                             initial_properties=initial_properties)

    print(f"[test_runner] Test '{test_name}' ready:")
    print(f"  backend     : {backend}")
    print(f"  sim         : {type(sim).__name__}")
    print(f"  focal_plane : {focal_plane} µm  (tissue_z=0.0, DOF=6.0 µm at 10x)")
    print(f"  channels    : {[ch['name'] for ch in channels]}")
    print(f"  cfg         : {cfg_path}")

    return cfg_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    if len(sys.argv) < 2:
        print_tests()
        sys.exit(0)

    test_name = sys.argv[1]
    try:
        cfg_path = run_test(test_name)
        print(f"\nCfg written to: {cfg_path}")
        print("Pass this path to MCPServer(auto_config=...) or use --test in plugin_napari.py")
    except (FileNotFoundError, ValueError, AttributeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
