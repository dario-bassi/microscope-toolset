import os
import subprocess
import sys
import logging
import threading
import signal
import time
from datetime import datetime

from typing import Any

import napari
from PyQt6.QtCore import Qt, QObject, pyqtSlot, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from src.mcp_microscopetoolset.utils import get_user_information
from src.start_subprocess.servers import _start_server, wait_for_es
from src.mcp_microscopetoolset.server_setup import create_mcp_server
from src.mcp_microscopetoolset.agents_init import initialize_agents
from src.mcp_microscopetoolset.viewer import NapariViewerMC
from src.microscope.microscope_event_cache import MicroscopeEventCache
from src.benchmarking.benchmark_logger import BenchmarkLogger

logger = logging.getLogger("MCPServer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


def _extract_cell_type(cfg_path: str) -> str:
    """'virtual_optogenetic.cfg' → 'optogenetic'."""
    basename = os.path.splitext(os.path.basename(cfg_path))[0]
    return basename.split("_", 1)[1] if "_" in basename else "normal"


def _is_old_style_virtual_cfg(cfg_path: str) -> bool:
    """True when the cfg loads devices from src.virtual_microscope (old simulation)."""
    try:
        with open(cfg_path) as f:
            return any("src.virtual_microscope" in line for line in f)
    except Exception:
        return False


class ThreadSafeViewerProxy(QObject):
    """Proxy to execute viewer operations on the main Qt thread"""
    execute_on_main_thread = pyqtSignal(str, dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.result = None
        self.error = None
        self.done_event = threading.Event()

        self.execute_on_main_thread.connect(
            self._execute_viewer_method,
            type=Qt.ConnectionType.QueuedConnection
        )

    @pyqtSlot(str, dict)
    def _execute_viewer_method(self, method_name, kwargs):
        try:
            method = getattr(self.viewer, method_name)
            self.result = method(**kwargs)
            self.error = None
        except Exception as e:
            logger.error(f"Error executing viewer method {method_name}: {e}")
            self.result = None
            self.error = e
        finally:
            self.done_event.set()

    def call_on_main_thread(self, method_name, **kwargs):
        self.result = None
        self.error = None
        self.done_event.clear()
        self.execute_on_main_thread.emit(method_name, kwargs)
        if not self.done_event.wait(timeout=10):
            raise RuntimeError(f"Timeout waiting for {method_name} to complete on main thread")
        if self.error:
            raise self.error
        return self.result


class MCPWorker(QObject):
    start_thread = pyqtSignal()
    stop_thread = pyqtSignal()
    status_update = pyqtSignal(str)
    servers_ready = pyqtSignal()
    servers_stopped = pyqtSignal()
    add_napari_micromanager = pyqtSignal()
    mcp_server_ready = pyqtSignal()

    def __init__(self, viewer: Any, viewer_proxy=None):
        super().__init__()
        self._elastic_search_process = None
        self._mmc = None
        self._last_cfg_path = None
        self._viewer = viewer
        self._viewer_proxy = viewer_proxy

    @pyqtSlot()
    def run_mcp_server(self):
        """
        Phase 1: Start Elasticsearch (optional) and add the napari-micromanager plugin.
        Core selection (CMMCorePlus vs UniMMCore) is handled automatically by
        napari-micromanager 0.3.0 based on whether the loaded .cfg contains #py device lines.
        Phase 2 (agents + MCP server) starts after the user loads a .cfg.
        """
        try:
            ui = get_user_information()
            es_home = ui.get("elastic_search_path_home", "")

            if es_home and os.path.isdir(es_home):
                self.status_update.emit("Loading Elasticsearch server...")
                if sys.platform.startswith("win"):
                    exe = f"{es_home}\\bin\\elasticsearch.bat"
                else:
                    exe = f"{es_home}/bin/elasticsearch"

                logger.info(f"Launching Elasticsearch: {exe}")
                self._elastic_search_process = _start_server([exe, "-d", "-p", "pid"])
                logger.info(f"Elasticsearch server started with PID={self._elastic_search_process.pid}")
                self.status_update.emit("Waiting for Elasticsearch to be ready...")
                try:
                    wait_for_es(max_wait=60)
                    logger.info("Elasticsearch is ready!")
                except Exception as e:
                    logger.warning(f"ES startup failed (continuing without it): {e}")
                    self.status_update.emit("Elasticsearch unavailable - starting without database tools")
            else:
                logger.info("Elasticsearch not configured - skipping")
                self.status_update.emit("Starting without Elasticsearch...")
                self._elastic_search_process = None

            self.add_napari_micromanager.emit()

            self.status_update.emit("Load a .cfg file via napari-micromanager to continue")
            self.servers_ready.emit()

        except Exception as e:
            logger.error(f"Error starting servers: {e}")
            self.status_update.emit(f"Error: {e}")
        finally:
            self.stop_thread.emit()

    def _stop_mcp_server_only(self) -> None:
        """Stop the MCP server without touching Elasticsearch."""
        if hasattr(self, '_uvicorn_server') and self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            logger.info("Signalled uvicorn to shut down")
        if hasattr(self, '_fastmcp_thread') and self._fastmcp_thread is not None:
            self._fastmcp_thread.join(timeout=15)
            if self._fastmcp_thread.is_alive():
                logger.warning("MCP thread did not stop in time — port 5500 may still be in use")
            self._fastmcp_thread = None
            self._uvicorn_server = None
            logger.info("Previous MCP server stopped")

    def _on_config_loaded(self):
        """Phase 2: stop any running MCP server, then start a fresh one."""
        cfg_path = self._last_cfg_path or ""
        logger.info(f"Configuration loaded: {cfg_path}")
        self._stop_mcp_server_only()

        def _init_mcp():
            try:
                self.status_update.emit("Initializing MCP server tools...")

                logger.info("Initializing agents...")
                agents = initialize_agents(mmc=self._mmc)

                logger.info("Creating MCP server...")
                viewer_instance = NapariViewerMC(self._viewer)

                if self._mmc is None:
                    raise RuntimeError("Microscope core not initialized - load a .cfg file first")
                event_cache = MicroscopeEventCache(self._mmc)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"benchmark_{timestamp}"
                ui = get_user_information()
                benchmark_agent = ui.get('benchmark_agent_enable', '')
                if benchmark_agent == 'false':
                    agent_type = 'untrained'
                elif benchmark_agent == 'true':
                    agent_type = 'trained'
                else:
                    agent_type = 'untrained'
                benchmark_logger = BenchmarkLogger(agent_type=agent_type, run_id=run_id)
                logger.info(f"Benchmark logger initialized: {run_id}")

                mcp_server = create_mcp_server(
                    database_agent=agents["database_agent"],
                    microscope_status=agents["microscope_status"],
                    executor=agents["executor"],
                    viewer=viewer_instance,
                    event_cache=event_cache,
                    viewer_proxy=self._viewer_proxy,
                    benchmark_logger_instance=benchmark_logger
                )

                import uvicorn, anyio
                logger.info("Starting FastMCP server...")
                starlette_app = mcp_server.streamable_http_app()
                config = uvicorn.Config(
                    starlette_app,
                    host=mcp_server.settings.host,
                    port=mcp_server.settings.port,
                    log_level=mcp_server.settings.log_level.lower(),
                )
                self._uvicorn_server = uvicorn.Server(config)
                self.status_update.emit("MCP server ready! You can connect from Claude Code.")
                self.mcp_server_ready.emit()
                try:
                    anyio.run(self._uvicorn_server.serve)
                except OSError as e:
                    if "10048" in str(e) or "address already in use" in str(e).lower():
                        logger.error("Port 5500 still in use — previous MCP server did not release it in time.")
                        self.status_update.emit("Error: port 5500 still in use — try loading the config again")
                    else:
                        raise

            except Exception as e:
                logger.exception(f"FastMCP error: {e}")
                self.status_update.emit(f"Error initializing MCP: {e}")

        self._fastmcp_thread = threading.Thread(target=_init_mcp, daemon=True)
        self._fastmcp_thread.start()

    @pyqtSlot()
    def stop_mcp_server(self):
        """Stop the MCP server"""
        self.status_update.emit("Stopping servers...")

        try:
            if self._elastic_search_process is not None:
                self.status_update.emit("Stopping Elasticsearch server...")
                try:
                    if sys.platform.startswith("win"):
                        subprocess.call(["taskkill", "/F", "/IM", "java.exe"])
                    else:
                        os.killpg(os.getpgid(self._elastic_search_process.pid), signal.SIGTERM)
                    logger.info("Stopped Elasticsearch")
                except Exception as e:
                    logger.warning(f"Could not stop Elasticsearch: {e}")

                time.sleep(2)

            self._stop_mcp_server_only()

            if self._mmc is not None:
                self._mmc = None

            logger.info("All servers stopped")
            self.status_update.emit("All servers stopped")
            self.servers_stopped.emit()

        except Exception as e:
            logger.error(f"Error stopping servers: {e}")
            self.status_update.emit(f"Error stopping servers: {e}")
            self.servers_stopped.emit()


class MCPServer(QWidget):

    def __init__(self, auto_config: str | None = None):
        super().__init__()
        self.mmc = None
        self.viewer = napari.current_viewer()
        self._auto_config = auto_config

        self.setObjectName("MCPServer")
        self.setWindowTitle("MCP Server")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(3)

        label_title = QLabel("Microscope Toolset MCP Server")
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setStyleSheet("font-weight: bold; font-size: 12px;")

        self.status_label = QLabel("Ready to start")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px; color: #666; padding: 2px;")

        self.start_button = QPushButton("Start Servers")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.start_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.stop_button = QPushButton("Stop Servers")
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.stop_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.stop_button.setEnabled(False)

        btn_width = max(self.start_button.sizeHint().width(),
                        self.stop_button.sizeHint().width())
        self.start_button.setFixedWidth(btn_width)
        self.stop_button.setFixedWidth(btn_width)

        main_layout.addWidget(label_title)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.start_button.clicked.connect(self.click_start_server)
        self.stop_button.clicked.connect(self.click_stop_server)

        self.mcp_thread = None
        self.mcp_worker = None
        self._viewer_proxy = None
        self._mmwin = None
        self._last_cfg_path = None
        # Prevents _capturing_load from triggering Phase 2 on the recursive inner call
        # that napari-micromanager makes during a CMMCorePlus↔UniMMCore core swap.
        self._in_user_load = False

        if self._auto_config is not None:
            QTimer.singleShot(500, self.click_start_server)

    def click_start_server(self):
        viewer_instance = NapariViewerMC(self.viewer)
        self._viewer_proxy = ThreadSafeViewerProxy(viewer_instance)

        self.mcp_thread = QThread()
        self.mcp_worker = MCPWorker(
            viewer=viewer_instance,
            viewer_proxy=self._viewer_proxy
        )
        self.mcp_worker.moveToThread(self.mcp_thread)

        self.mcp_thread.started.connect(self.mcp_worker.run_mcp_server)
        self.mcp_worker.stop_thread.connect(self.mcp_thread.quit)

        self.mcp_worker.status_update.connect(self.update_status_label)
        self.mcp_worker.servers_ready.connect(self.on_servers_ready)
        self.mcp_worker.servers_stopped.connect(self.on_servers_stopped)
        self.mcp_worker.add_napari_micromanager.connect(self.add_napari_micromanager_plugin)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Starting servers...")

        self.mcp_thread.start()

    def click_stop_server(self):
        if self.mcp_thread is None:
            return
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.status_label.setText("Stopping servers...")
        self.mcp_worker.stop_mcp_server()
        if self.mcp_thread:
            self.mcp_thread.quit()

    @pyqtSlot(str)
    def update_status_label(self, message):
        self.status_label.setText(message)

        base = "font-size: 11px; padding: 2px;"
        if "ready" in message.lower() or "you can start" in message.lower():
            self.status_label.setStyleSheet(f"{base} color: #4CAF50; font-weight: bold;")
        elif "error" in message.lower() or "failed" in message.lower():
            self.status_label.setStyleSheet(f"{base} color: #f44336; font-weight: bold;")
        elif "loading" in message.lower() or "starting" in message.lower() or "stopping" in message.lower() or "waiting" in message.lower():
            self.status_label.setStyleSheet(f"{base} color: #FF9800; font-weight: bold;")
        elif "load a .cfg" in message.lower():
            self.status_label.setStyleSheet(f"{base} color: #2196F3; font-weight: bold;")
        else:
            self.status_label.setStyleSheet(f"{base} color: #666;")

    @pyqtSlot()
    def on_servers_ready(self):
        """Called when Phase 1 is complete — napari-micromanager plugin is live."""
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        if self._auto_config is not None and self._mmwin is not None:
            cfg = self._auto_config
            logger.info(f"Auto-loading config: {cfg}")
            self.status_label.setText(f"Auto-loading {os.path.basename(cfg)}...")
            self._mmwin.core.loadSystemConfiguration(cfg)

    @pyqtSlot()
    def on_servers_stopped(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Ready to start")
        self.status_label.setStyleSheet("font-size: 11px; color: #666; padding: 2px;")

    @pyqtSlot()
    def add_napari_micromanager_plugin(self):
        """Add the napari-micromanager plugin and wire the Phase 2 trigger.

        On restart (Stop → Start) the plugin is already present. _capturing_load
        closes over `self`, so self.mcp_worker always refers to the current worker —
        no re-wiring needed.
        """
        import traceback
        try:
            if self._mmwin is not None:
                logger.info("napari-micromanager already present — skipping re-wiring")
                return

            logger.info("Adding napari-micromanager plugin...")
            self.viewer.window.add_plugin_dock_widget(plugin_name="napari-micromanager")
            logger.info("Successfully added napari-micromanager plugin")

            from napari_micromanager.main_window import get_main_window
            win = get_main_window()
            self._mmwin = win

            def _install_hooks(core):
                """Replace core.loadSystemConfiguration with a wrapper that triggers Phase 2."""
                _nm_load = core.loadSystemConfiguration

                def _capturing_load(path):
                    # During a CMMCorePlus↔UniMMCore swap, napari-micromanager's
                    # _auto_detect_load calls loadSystemConfiguration on the new core
                    # from inside _nm_load — which hits our wrapper again. The flag
                    # lets that inner call pass through without starting a second Phase 2.
                    if self._in_user_load:
                        _nm_load(path)
                        return
                    self._in_user_load = True
                    try:
                        self._last_cfg_path = str(path)
                        if _is_old_style_virtual_cfg(str(path)):
                            from src.virtual_microscope.initialize_virtual_microscope import (
                                initialize_virtual_microscope_from_configuration,
                            )
                            cell_type = _extract_cell_type(str(path))
                            logger.info(f"Pre-init old simulation bridge: cell_type={cell_type}")
                            initialize_virtual_microscope_from_configuration(
                                core=core, cell_type=cell_type
                            )
                        _nm_load(path)
                        # Load fully complete (including any core swap) — start Phase 2 once.
                        self._start_phase2()
                    finally:
                        self._in_user_load = False

                core.loadSystemConfiguration = _capturing_load

            # Patch set_core so _capturing_load survives a CMMCorePlus↔UniMMCore swap.
            _original_set_core = win.set_core

            def _patched_set_core(core):
                _original_set_core(core)
                _install_hooks(win.core)

            win.set_core = _patched_set_core
            _install_hooks(win.core)

        except Exception as e:
            logger.error(
                f"Failed to add napari-micromanager plugin: {e}\n"
                + traceback.format_exc()
            )

    def _start_phase2(self) -> None:
        """Called once per user-initiated cfg load; hands the loaded core to the worker."""
        if self.mcp_worker is None:
            return
        mmc = self._mmwin.core
        logger.info(f"Config loaded: {self._last_cfg_path!r}  core: {type(mmc).__name__}")
        self.mcp_worker._mmc = mmc
        self.mcp_worker._last_cfg_path = self._last_cfg_path or ""
        self.mcp_worker._on_config_loaded()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
