"""Full signal coverage test for RemoteMMCore against the virtual microscope.

Tests all core and MDA signals across 5 groups:
  1. Property & config signals
  2. Stage movement signals
  3. Shutter & snap signals
  4. Config CRUD & ROI signals
  5. MDA signals

Uses particle.cfg (UniMMCore) as the backend.
Opt-in via:  VIRTUAL_MICROSCOPE_TESTS=1 pytest test/test_all_signals.py -v

Known virtual-microscope limitations (signals that cannot be tested here)
-------------------------------------------------------------------------
shutterOpenChanged
    SimShutterDevice.set_open() only writes to self._shutter in Python; it
    never calls the underlying C++ callback that would trigger the signal.
    The test is explicitly skipped.

propertiesChanged
    UniMMCore does not emit this batch-property signal at all. It is excluded
    from the test list rather than skipped so as not to produce misleading
    output.

awaitingEvent
    Only fires when interval > per-frame acquisition time so that a positive
    remaining-wait value exists between events. The particle camera takes
    4-5 s per frame (numba JIT on first run) while any practical test
    interval is shorter, so remaining is always ≤ 0. Excluded from
    MDA_SIGS.

sequencePauseToggled (in test_all_mda_signals_fire)
    Requires an explicit pause call during a running sequence. Tested in
    its own dedicated test (test_pause_and_resume) instead of lumped into
    the bulk signal check.
"""

from __future__ import annotations

import os
import pathlib
import socket
import threading
import time

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).parent.parent
_PARTICLE_CFG = _ROOT / "virtual-microscope/src/virtual_microscope/backends/particle/particle.cfg"

_SKIP = pytest.mark.skipif(
    os.environ.get("VIRTUAL_MICROSCOPE_TESTS") != "1",
    reason="Set VIRTUAL_MICROSCOPE_TESTS=1 to run virtual-microscope signal tests",
)

pytestmark = _SKIP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def virtual_core():
    """UniMMCore loaded with particle.cfg."""
    pytest.importorskip("pymmcore_proxy", reason="pymmcore_proxy not installed")
    pytest.importorskip("virtual_microscope", reason="virtual_microscope not installed")

    if not _PARTICLE_CFG.exists():
        pytest.skip(f"particle.cfg not found at {_PARTICLE_CFG}")

    import os as _os

    _old = _os.environ.get("PYMM_SIGNALS_BACKEND")
    _os.environ["PYMM_SIGNALS_BACKEND"] = "psygnal"
    try:
        from pymmcore_plus.experimental.unicore import UniMMCore

        core = UniMMCore()
    finally:
        if _old is None:
            _os.environ.pop("PYMM_SIGNALS_BACKEND", None)
        else:
            _os.environ["PYMM_SIGNALS_BACKEND"] = _old

    core.loadSystemConfiguration(str(_PARTICLE_CFG))
    return core


@pytest.fixture(scope="module")
def server_url(virtual_core):
    """Start a ProxyServer backed by the virtual core; yield base URL."""
    import uvicorn
    from pymmcore_proxy import ProxyServer

    port = _free_port()
    proxy = ProxyServer(virtual_core, port=port)
    config = uvicorn.Config(proxy.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15.0
    import httpx

    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("Virtual microscope proxy server failed to start")

    yield url

    server.should_exit = True
    thread.join(timeout=5.0)


@pytest.fixture()
def core(server_url):
    """RemoteMMCore connected to the virtual microscope proxy."""
    from pymmcore_proxy import RemoteMMCore

    client = RemoteMMCore(server_url, connect_signals=True)
    time.sleep(0.3)  # let signal listener connect
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subscribe(core_events, signal_names: list[str]):
    done: dict[str, threading.Event] = {}
    handlers: dict[str, object] = {}

    def make_handler(name):
        def h(*args):
            done[name].set()

        return h

    for s in signal_names:
        sig = getattr(core_events, s, None)
        if sig is not None:
            done[s] = threading.Event()
            handlers[s] = make_handler(s)
            sig.connect(handlers[s])

    return done, handlers


def _unsubscribe(core_events, signal_names: list[str], handlers: dict):
    for s in signal_names:
        sig = getattr(core_events, s, None)
        if sig is not None and s in handlers:
            try:
                sig.disconnect(handlers[s])
            except Exception:
                pass


def _wait(done: dict, name: str, timeout: float = 3.0) -> str:
    return "OK" if done.get(name, threading.Event()).wait(timeout) else "TIMEOUT"


# ---------------------------------------------------------------------------
# Group 1 — Property & config signals
# ---------------------------------------------------------------------------


class TestPropertyAndConfigSignals:
    # propertiesChanged excluded — UniMMCore never emits this batch signal (see module docstring)
    SIGS = ["exposureChanged", "propertyChanged", "configSet"]

    def test_exposure_changed(self, core):
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setExposure(42.0)
            assert _wait(done, "exposureChanged") == "OK", "exposureChanged timed out"
        finally:
            _unsubscribe(core.events, self.SIGS, handlers)

    def test_property_changed(self, core):
        # Use setConfig to switch channels — guaranteed to change LED + Filter Wheel
        # properties even if already at a given value. Directly setting a property
        # to its current value does NOT fire propertyChanged (pymmcore skips it).
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setConfig("Channel", "membrane")
            core.setConfig("Channel", "DAPI")  # changes LED + Filter Wheel → fires signal
            assert _wait(done, "propertyChanged") == "OK", "propertyChanged timed out"
        finally:
            _unsubscribe(core.events, self.SIGS, handlers)

    def test_config_set(self, core):
        # particle.cfg: Channel group with presets phase-contrast, DAPI, membrane
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setConfig("Channel", "DAPI")
            assert _wait(done, "configSet") == "OK", "configSet timed out"
        finally:
            _unsubscribe(core.events, self.SIGS, handlers)


# ---------------------------------------------------------------------------
# Group 2 — Stage movement signals
# ---------------------------------------------------------------------------


class TestStageSignals:
    SIGS = ["XYStagePositionChanged", "stagePositionChanged"]

    def test_xy_stage_position_changed(self, core):
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setXYPosition(10.0, 20.0)
            core.waitForDevice(core.getXYStageDevice())
            assert _wait(done, "XYStagePositionChanged") == "OK", "XYStagePositionChanged timed out"
        finally:
            _unsubscribe(core.events, self.SIGS, handlers)

    def test_stage_position_changed(self, core):
        focus_dev = core.getFocusDevice()
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setPosition(focus_dev, 5.0)
            core.waitForDevice(focus_dev)
            assert _wait(done, "stagePositionChanged") == "OK", "stagePositionChanged timed out"
        finally:
            _unsubscribe(core.events, self.SIGS, handlers)


# ---------------------------------------------------------------------------
# Group 3 — Shutter & snap signals
# ---------------------------------------------------------------------------


class TestShutterAndSnapSignals:
    SIGS = ["autoShutterSet", "shutterOpenChanged", "imageSnapped"]

    def test_auto_shutter_set(self, core):
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setAutoShutter(False)
            assert _wait(done, "autoShutterSet") == "OK", "autoShutterSet timed out"
        finally:
            core.setAutoShutter(True)
            _unsubscribe(core.events, self.SIGS, handlers)

    def test_shutter_open_changed(self, core):
        # Skipped: SimShutterDevice.set_open() only writes self._shutter in Python.
        # No C++ callback is invoked, so the shutterOpenChanged signal is never emitted
        # regardless of how or when the shutter is toggled. This is a virtual-microscope
        # limitation, not a proxy bug. See module docstring for full explanation.
        pytest.skip("shutterOpenChanged not emitted by virtual SimShutterDevice")

    def test_image_snapped(self, core):
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.snapImage()
            assert _wait(done, "imageSnapped") == "OK", "imageSnapped timed out"
            img = core.getImage()
            assert isinstance(img, np.ndarray)
            assert img.ndim == 2
        finally:
            _unsubscribe(core.events, self.SIGS, handlers)


# ---------------------------------------------------------------------------
# Group 4 — Config CRUD & ROI signals
# ---------------------------------------------------------------------------


class TestConfigCrudAndRoiSignals:
    SIGS = ["roiSet", "configDefined", "configDeleted", "configGroupDeleted"]

    def test_roi_set(self, core):
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.setROI(0, 0, 256, 256)
            assert _wait(done, "roiSet", 2.0) == "OK", "roiSet timed out"
        finally:
            core.clearROI()
            _unsubscribe(core.events, self.SIGS, handlers)

    def test_config_defined_and_deleted(self, core):
        done, handlers = _subscribe(core.events, self.SIGS)
        try:
            core.defineConfig("_TestGroup", "_TestCfg", "LED", "Label", "CYAN")
            assert _wait(done, "configDefined", 2.0) == "OK", "configDefined timed out"

            done["configDeleted"] = threading.Event()
            core.deleteConfig("_TestGroup", "_TestCfg")
            assert _wait(done, "configDeleted", 2.0) == "OK", "configDeleted timed out"

            done["configGroupDeleted"] = threading.Event()
            core.deleteConfigGroup("_TestGroup")
            assert _wait(done, "configGroupDeleted", 2.0) == "OK", "configGroupDeleted timed out"
        finally:
            try:
                core.deleteConfigGroup("_TestGroup")
            except Exception:
                pass
            _unsubscribe(core.events, self.SIGS, handlers)


# ---------------------------------------------------------------------------
# Group 5 — MDA signals
# ---------------------------------------------------------------------------


class TestMDASignals:
    # awaitingEvent excluded — see module docstring. The particle camera's per-frame
    # cost always exceeds any practical interval, so remaining is never positive.
    # sequencePauseToggled excluded here — it needs an explicit pause mid-run and is
    # covered by the dedicated test_pause_and_resume test below.
    MDA_SIGS = [
        "sequenceStarted",
        "sequenceFinished",
        "eventStarted",
        "frameReady",
    ]

    def test_all_mda_signals_fire(self, core):
        from useq import MDASequence

        done, handlers = _subscribe(core.mda.events, self.MDA_SIGS)
        try:
            seq = MDASequence(
                channels=[{"config": "DAPI", "exposure": 10}],
                time_plan={"loops": 2, "interval": 0},
            )
            core.mda.run(seq)
            for s in self.MDA_SIGS:
                if s in done:
                    assert _wait(done, s, 10.0) == "OK", f"{s} timed out"
        finally:
            _unsubscribe(core.mda.events, self.MDA_SIGS, handlers)

    def test_pause_and_resume(self, core):
        import threading as _threading

        from useq import MDASequence

        paused_signals = []
        frames = []
        # Wait for frame 2 before pausing: by then numba JIT is warm and frames arrive
        # fast, so the MDA is guaranteed to still be running when set_paused(True) is
        # sent. Using loops=100 ensures the sequence outlasts the pause/resume cycle.
        # After pause+resume we cancel so the test doesn't wait for all 100 frames.
        # (With loops=4 and warm JIT, all frames complete in ~0.4s and set_paused
        # arrives after the MDA has already finished — it becomes a no-op.)
        _triggered = _threading.Event()

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 2 and not _triggered.is_set():
                _triggered.set()
                core.mda.set_paused(True)

                def resume_and_cancel():
                    time.sleep(0.5)
                    core.mda.set_paused(False)
                    time.sleep(0.5)
                    core.mda.cancel()

                _threading.Thread(target=resume_and_cancel, daemon=True).start()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequencePauseToggled.connect(lambda p: paused_signals.append(p))
        try:
            # Use enough loops so the sequence outlasts the pause/resume cycle
            seq = MDASequence(time_plan={"loops": 100, "interval": 0})
            core.mda.run(seq)
            assert len(paused_signals) >= 2, "Expected pause + unpause signals"
        finally:
            core.mda.events.frameReady.disconnect(on_frame)
            core.mda.events.sequencePauseToggled.disconnect()

    def test_frame_ready_delivers_array(self, core):
        from useq import MDASequence

        frames = []
        core.mda.events.frameReady.connect(lambda img, ev, meta: frames.append(img))
        try:
            seq = MDASequence(
                channels=[{"config": "membrane", "exposure": 50}],
                time_plan={"loops": 2, "interval": 0},
            )
            core.mda.run(seq)
            assert len(frames) == 2
            for img in frames:
                assert isinstance(img, np.ndarray)
                assert img.ndim == 2
        finally:
            core.mda.events.frameReady.disconnect()

    def test_cancel_stops_sequence(self, core):
        from useq import MDASequence

        frames = []
        canceled = []

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 2:
                core.mda.cancel()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequenceCanceled.connect(lambda seq: canceled.append(seq))
        try:
            seq = MDASequence(time_plan={"loops": 50, "interval": 0})
            core.mda.run(seq)
            assert len(frames) < 50, "Sequence was not canceled"
            assert len(canceled) == 1
        finally:
            core.mda.events.frameReady.disconnect(on_frame)
            core.mda.events.sequenceCanceled.disconnect()
