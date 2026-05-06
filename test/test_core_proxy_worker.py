"""Tests for CoreProxyWorker.

Strategy:
- Signal wiring and error paths: use pytest-qt (qtbot) — no microscope needed.
- Real-device happy path: uses pymmcore-plus bundled MMConfig_demo.cfg (C++ DemoCamera
  drivers, no real hardware required) — runs automatically when pymmcore_proxy is present.
- Virtual happy path: skipped unless VIRTUAL_MICROSCOPE_TESTS=1 (needs virtual_microscope).

Note: the worker does NOT validate cfg path existence — it starts an empty proxy and
emits server_ready(url, cfg_path); the caller loads the cfg via RPC after the proxy is up.
A nonexistent path is therefore not an error at the worker level.
"""

import os

import pytest

pytest.importorskip("PyQt6", reason="PyQt6 not available")

from PyQt6.QtCore import QThread

from src.utils import CoreProxyWorker

# ── error path: mixed cfg ─────────────────────────────────────────────────────


def test_mixed_cfg_emits_server_error(qtbot, tmp_path):
    """A cfg with both #py pyDevice and Device lines must emit server_error immediately."""
    mixed_cfg = tmp_path / "mixed.cfg"
    mixed_cfg.write_text(
        "#py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice\n"
        "Device,Stage,DemoCamera,DCamStage\n",
        encoding="utf-8",
    )

    worker = CoreProxyWorker(cfg_path=str(mixed_cfg), port=15904)

    errors = []
    readys = []
    worker.server_error.connect(errors.append)
    worker.server_ready.connect(readys.append)

    with qtbot.waitSignal(worker.server_error, timeout=5_000):
        worker.run()

    assert len(errors) == 1
    assert len(readys) == 0
    assert "mixed" in errors[0].lower() or "not supported" in errors[0].lower()


# ── signal wiring: worker runs in QThread ────────────────────────────────────


def test_worker_signals_cross_thread(qtbot, tmp_path):
    """Signals emitted from the worker thread must reach slots on the main thread.

    Uses a mixed cfg (both #py and Device lines) to trigger server_error fast —
    the check happens before any heavy imports so the test stays quick.
    """
    mixed_cfg = tmp_path / "mixed_cross_thread.cfg"
    mixed_cfg.write_text(
        "#py pyDevice,Camera,virtual_microscope.devices.camera,SimCameraDevice\n"
        "Device,Stage,DemoCamera,DCamStage\n",
        encoding="utf-8",
    )

    thread = QThread()
    worker = CoreProxyWorker(cfg_path=str(mixed_cfg), port=15902)
    worker.moveToThread(thread)

    received = []
    worker.server_error.connect(received.append)
    thread.started.connect(worker.run)

    with qtbot.waitSignal(worker.server_error, timeout=5_000):
        thread.start()

    thread.quit()
    thread.wait(2_000)

    assert len(received) == 1


# ── integration: full server start with real (C++ demo) cfg ──────────────────


def test_full_server_start_real(qtbot):
    """Full integration: loads MMConfig_demo.cfg (C++ DemoCamera), starts proxy,
    emits server_ready. No real hardware needed — DemoCamera is a software stub."""
    pytest.importorskip("pymmcore_proxy", reason="pymmcore_proxy not installed")
    import pathlib

    import pymmcore_plus

    mm_path = pymmcore_plus.find_micromanager()
    if mm_path is None:
        pytest.skip("Micro-Manager installation not found (run `mmcore install` first)")
    cfg = pathlib.Path(mm_path) / "MMConfig_demo.cfg"
    if not cfg.exists():
        pytest.skip(f"MMConfig_demo.cfg not found at {cfg}")

    worker = CoreProxyWorker(cfg_path=str(cfg), port=15905)

    readys = []
    errors = []
    worker.server_ready.connect(readys.append)
    worker.server_error.connect(errors.append)

    with qtbot.waitSignal(worker.server_ready, timeout=20_000):
        worker.run()

    assert len(readys) == 1
    assert readys[0].startswith("http://127.0.0.1:15905")
    assert len(errors) == 0


# ── integration: full server start with virtual cfg (opt-in) ─────────────────


@pytest.mark.skipif(
    os.environ.get("VIRTUAL_MICROSCOPE_TESTS") != "1",
    reason="Set VIRTUAL_MICROSCOPE_TESTS=1 to run full proxy integration tests",
)
def test_full_server_start_virtual(qtbot):
    """Full integration: loads bacteria.cfg (virtual), starts proxy, emits server_ready."""
    pytest.importorskip("pymmcore_proxy", reason="pymmcore_proxy not installed")
    pytest.importorskip("virtual_microscope", reason="virtual_microscope not installed")

    import pathlib

    project_root = pathlib.Path(__file__).parent.parent
    cfg = project_root / "virtual-microscope/src/virtual_microscope/backends/bacteria/bacteria.cfg"
    if not cfg.exists():
        pytest.skip(f"bacteria.cfg not found at {cfg}")

    worker = CoreProxyWorker(cfg_path=str(cfg), port=15906)

    readys = []
    errors = []
    worker.server_ready.connect(readys.append)
    worker.server_error.connect(errors.append)

    with qtbot.waitSignal(worker.server_ready, timeout=20_000):
        worker.run()

    assert len(readys) == 1
    assert readys[0].startswith("http://127.0.0.1:15906")
    assert len(errors) == 0
