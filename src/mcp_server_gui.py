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
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.core import _mmcore_plus as _mmcore_plus_module
from pymmcore_plus.experimental.unicore import UniMMCore

from src.mcp_microscopetoolset.utils import get_user_information
from src.start_subprocess.servers import _start_server, wait_for_es
from src.mcp_microscopetoolset.server_setup import create_mcp_server
from src.mcp_microscopetoolset.agents_init import initialize_agents
from src.mcp_microscopetoolset.viewer import NapariViewerMC
from src.microscope.microscope_event_cache import MicroscopeEventCache
from src.virtual_microscope.initialize_virtual_microscope import initialize_virtual_microscope_from_configuration
from src.benchmarking.benchmark_logger import BenchmarkLogger

#  logger
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
    """Extract cell type from config filename. E.g. 'virtual_normal.cfg' -> 'normal'."""
    basename = os.path.splitext(os.path.basename(cfg_path))[0]  # "virtual_normal"
    return basename.split("_", 1)[1] if "_" in basename else "normal"


def _is_virtual_config(cfg_path: str) -> bool:
    """Check if a config file is a virtual microscope config (contains #py lines)."""
    try:
        with open(cfg_path, "r") as f:
            for line in f:
                if line.strip().startswith("#py "):
                    return True
    except Exception:
        pass
    return False


class ThreadSafeViewerProxy(QObject):
    """Proxy to execute viewer operations on the main Qt thread"""
    execute_on_main_thread = pyqtSignal(str, dict)  # Don't pass viewer through signal to avoid Qt threading issues

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.result = None
        self.error = None
        self.done_event = threading.Event()

        # Connect signal with explicit QueuedConnection for cross-thread safety
        # This ensures the slot is called in the receiver's thread (main thread)
        self.execute_on_main_thread.connect(
            self._execute_viewer_method,
            type=Qt.ConnectionType.QueuedConnection
        )

    @pyqtSlot(str, dict)
    def _execute_viewer_method(self, method_name, kwargs):
        """Execute viewer method on main thread"""
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
        """Execute a viewer method on main thread and wait for result"""
        self.result = None
        self.error = None
        self.done_event.clear()

        # Emit signal to execute on main thread
        self.execute_on_main_thread.emit(method_name, kwargs)

        # Wait for result (with timeout to prevent deadlock)
        if not self.done_event.wait(timeout=10):
            raise RuntimeError(f"Timeout waiting for {method_name} to complete on main thread")

        if self.error:
            raise self.error

        return self.result


class MCPWorker(QObject):
    start_thread = pyqtSignal()
    stop_thread = pyqtSignal()
    # New signals for status updates
    status_update = pyqtSignal(str)  # For status messages
    servers_ready = pyqtSignal()  # When both servers are ready
    servers_stopped = pyqtSignal()  # When both servers are stopped
    add_napari_micromanager = pyqtSignal()  # Signal to add napari-micromanager
    mcp_server_ready = pyqtSignal()  # When MCP server is fully initialized

    def __init__(self, viewer: Any, viewer_proxy=None):
        super().__init__()
        self._elastic_search_process = None

        self._mmc = None
        self._viewer = viewer
        self._viewer_proxy = viewer_proxy
        self._last_cfg_path = None

    @pyqtSlot()
    def run_mcp_server(self):
        """
        Phase 1: Set up UniMMCore singleton and Elasticsearch.
        The MCP server tools are initialized later when a .cfg is loaded.
        """
        try:
            # Try to start Elasticsearch (optional)
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

            # Phase 1: Create UniMMCore and set as CMMCorePlus singleton
            self.status_update.emit("Setting up virtual microscope core...")
            self._mmc = UniMMCore()
            _mmcore_plus_module._instance = self._mmc  # napari-micromanager will use this singleton

            # Wrap loadSystemConfiguration to track the loaded config path
            # and initialize SimulationBridge BEFORE devices are loaded
            # (devices like SLM access GLOBAL_BRIDGE during init)
            _original_load = self._mmc.loadSystemConfiguration

            def _tracking_load(*args, **kwargs):
                cfg_path = args[0] if args else kwargs.get('fileName', '')
                self._last_cfg_path = cfg_path
                # Init SimulationBridge before loading config so devices can access it
                if cfg_path and _is_virtual_config(cfg_path):
                    cell_type = _extract_cell_type(cfg_path)
                    logger.info(f"Pre-init SimulationBridge for cell_type={cell_type}")
                    initialize_virtual_microscope_from_configuration(
                        core=self._mmc, cell_type=cell_type
                    )
                result = _original_load(*args, **kwargs)
                # Trigger Phase 2 directly after config loads
                # (signal-based approach doesn't work because the worker QThread
                #  event loop has already quit by this point)
                self._on_config_loaded()
                return result

            self._mmc.loadSystemConfiguration = _tracking_load
            logger.info("UniMMCore singleton set up, waiting for config load...")

            # Add napari-micromanager plugin (will use our singleton)
            self.add_napari_micromanager.emit()

            # Phase 1 complete - waiting for user to load a .cfg
            self.status_update.emit("Load a .cfg file via napari-micromanager to continue")
            self.servers_ready.emit()

        except Exception as e:
            logger.error(f"Error starting servers: {e}")
            self.status_update.emit(f"Error: {e}")
        finally:
            self.stop_thread.emit()

    def _on_config_loaded(self):
        """
        Phase 2: Called when systemConfigurationLoaded fires.
        Initialize SimulationBridge (if virtual), agents, and MCP server.
        """
        cfg_path = self._last_cfg_path or ""
        logger.info(f"Configuration loaded: {cfg_path}")

        def _init_mcp():
            try:
                self.status_update.emit("Initializing MCP server tools...")

                # SimulationBridge was already initialized in _tracking_load
                # (before devices were loaded, so they can access GLOBAL_BRIDGE)

                # Initialize agents (mmc is already configured with devices)
                logger.info("Initializing agents...")
                agents = initialize_agents(mmc=self._mmc)

                logger.info("Creating MCP server...")
                viewer_instance = NapariViewerMC(self._viewer)

                # Create event cache (guard against mmc not yet initialized)
                if self._mmc is None:
                    raise RuntimeError("Microscope core not initialized - load a .cfg file first")
                event_cache = MicroscopeEventCache(self._mmc)

                # Initialize benchmark logger if needed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = f"benchmark_{timestamp}"
                ui = get_user_information()
                benchmark_agent = ui.get('benchmark_agent_enable', '')
                if benchmark_agent == 'false':
                    agent_type = 'untrained'
                elif benchmark_agent == 'true':
                    agent_type = 'trained'
                else:
                    agent_type = 'untrained' # default
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

                # Run the server — create uvicorn explicitly so we can shut it down later
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
                anyio.run(self._uvicorn_server.serve)

            except Exception as e:
                logger.exception(f"FastMCP error: {e}")
                self.status_update.emit(f"Error initializing MCP: {e}")

        self._fastmcp_thread = threading.Thread(target=_init_mcp, daemon=True)
        self._fastmcp_thread.start()

    @pyqtSlot()
    def stop_mcp_server(self):
        """
        Stop the MCP server
        """
        self.status_update.emit("Stopping servers...")

        try:
            # Stop Elasticsearch if it was started
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

            # Gracefully shut down uvicorn so it releases port 5500
            if hasattr(self, '_uvicorn_server') and self._uvicorn_server is not None:
                self._uvicorn_server.should_exit = True
                logger.info("Signalled uvicorn to shut down")
            if hasattr(self, '_fastmcp_thread') and self._fastmcp_thread is not None:
                self._fastmcp_thread.join(timeout=5)
                self._fastmcp_thread = None
                self._uvicorn_server = None
                logger.info("MCP server thread stopped")

            # Clear the singleton
            if self._mmc is not None:
                _mmcore_plus_module._instance = None
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

        # ---GUI----
        self.setObjectName("MCPServer")
        self.setWindowTitle("MCP Server")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(3)

        # Title label
        label_title = QLabel("Microscope Toolset MCP Server")
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setStyleSheet("font-weight: bold; font-size: 12px;")

        # Status label
        self.status_label = QLabel("Ready to start")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px; color: #666; padding: 2px;")

        # Start button
        self.start_button = QPushButton("Start Servers")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.start_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Stop button
        self.stop_button = QPushButton("Stop Servers")
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.stop_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.stop_button.setEnabled(False)

        # Make both buttons the same width (use the wider one's hint)
        btn_width = max(self.start_button.sizeHint().width(),
                        self.stop_button.sizeHint().width())
        self.start_button.setFixedWidth(btn_width)
        self.stop_button.setFixedWidth(btn_width)

        # Layout: title, buttons centered + stacked, status at bottom
        main_layout.addWidget(label_title)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Connect signals
        self.start_button.clicked.connect(self.click_start_server)
        self.stop_button.clicked.connect(self.click_stop_server)

        # QThread setup
        self.mcp_thread = None
        self.mcp_worker = None
        self._viewer_proxy = None  # Store proxy as instance variable

        # Auto-start: trigger after Qt event loop is running
        if self._auto_config is not None:
            QTimer.singleShot(500, self.click_start_server)

    def click_start_server(self):
        """Handle start button click"""
        # Create viewer wrapper and proxy on main thread (before moving worker)
        # Store as instance variable to prevent garbage collection
        viewer_instance = NapariViewerMC(self.viewer)
        self._viewer_proxy = ThreadSafeViewerProxy(viewer_instance)

        # Create new Worker and thread for each start
        self.mcp_thread = QThread()
        self.mcp_worker = MCPWorker(
            viewer=viewer_instance,
            viewer_proxy=self._viewer_proxy
        )
        self.mcp_worker.moveToThread(self.mcp_thread)

        # Connect worker signal
        self.mcp_thread.started.connect(self.mcp_worker.run_mcp_server)
        self.mcp_worker.stop_thread.connect(self.mcp_thread.quit)

        # Connect status update
        self.mcp_worker.status_update.connect(self.update_status_label)
        self.mcp_worker.servers_ready.connect(self.on_servers_ready)
        self.mcp_worker.servers_stopped.connect(self.on_servers_stopped)

        self.mcp_worker.add_napari_micromanager.connect(self.add_napari_micromanager_plugin)

        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Starting servers...")

        self.mcp_thread.start()

    def click_stop_server(self):
        """Handle stop button click"""
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
        """Update the status label with current operation"""
        self.status_label.setText(message)

        # Change color based on message content
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
        """Called when Phase 1 is complete (singleton set up, waiting for config)"""
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

        # Auto-load config if specified — the wrapped loadSystemConfiguration
        # handles SimulationBridge init + Phase 2 (agents + MCP server) automatically
        if self._auto_config is not None and self.mcp_worker is not None:
            cfg = self._auto_config
            logger.info(f"Auto-loading config: {cfg}")
            self.status_label.setText(f"Auto-loading {os.path.basename(cfg)}...")
            self.mcp_worker._mmc.loadSystemConfiguration(cfg)

    @pyqtSlot()
    def on_servers_stopped(self):
        """Called when servers are stopped"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Ready to start")
        self.status_label.setStyleSheet("font-size: 11px; color: #666; padding: 2px;")

    @pyqtSlot()
    def add_napari_micromanager_plugin(self):
        """Add a new napari micromanager plugin"""
        try:
            logger.info("Adding new napari micromanager plugin...")
            self.viewer.window.add_plugin_dock_widget(plugin_name="napari-micromanager")
            logger.info("Successfully added napari micromanager plugin")
        except Exception as e:
            logger.error(f"Failed to add new napari micromanager plugin: {e}")

    def closeEvent(self, event):
        """Handle window close event"""
        self.hide()
        event.ignore()
