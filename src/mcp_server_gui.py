import os
import subprocess
import sys
import logging
import threading
import signal
import time

from typing import Any
from functools import wraps
import threading

import napari
from PyQt6.QtCore import Qt, QObject, pyqtSlot, QThread, pyqtSignal, QCoreApplication
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QMainWindow, QApplication
from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore

from src.mcp_microscopetoolset.utils import get_user_information
from src.start_subprocess.servers import _start_server, wait_for_es
from src.mcp_microscopetoolset.server_setup import create_mcp_server, run_server
from src.mcp_microscopetoolset.agents_init import initialize_agents
from src.mcp_microscopetoolset.viewer import NapariViewerMC
from src.microscope.microscope_event_cache import MicroscopeEventCache

#  logger
logger = logging.getLogger("MCPServer")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(fh)


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
    def __init__(self,viewer: Any, microscope_type: str = "real", viewer_proxy=None):
        super().__init__()
        self._elastic_search_process = None
        #self._fastmcp_process: subprocess.Popen = None
        self._mmc = None
        self._viewer = viewer
        self._microscope_type = microscope_type
        self._viewer_proxy = viewer_proxy

    def set_microscope_type(self, microscope_type: str):
        self._microscope_type = microscope_type
        logger.info(f"Setting microscope type to {microscope_type}")

    @pyqtSlot()
    def run_mcp_server(self):
        """
        Start the MCP server
        """
        try:
            # Start Elasticsearch first
            self.status_update.emit("Loading Elasticsearch server...")

            ui = get_user_information()
            es_home = ui["elastic_search_path_home"]
            if sys.platform.startswith("win"):
                exe = f"{es_home}\\bin\\elasticsearch.bat"
            else:
                exe = f"{es_home}/bin/elasticsearch"

            logger.info(f"Launching Elasticsearch: {exe}")
            self._elastic_search_process = _start_server([exe, "-d", "-p", "pid"])
            logger.info(f"Elasticsearch server started with PID={self._elastic_search_process.pid}")
            # Wait until server has started and ping with Elasticsearch Client
            self.status_update.emit("Waiting for Elasticsearch to be ready...")
            try:
                wait_for_es(max_wait=60)
                logger.info("Elasticsearch is ready!")
            except Exception as e:
                logger.error(f"ES startup failed: {e}")
                self._elastic_search_process.terminate()
                self.status_update.emit("Elasticsearch failed to start")
                return

            # Start FastMCP server
            self.status_update.emit("Loading FastMCP server...")
            #self._fastmcp_process = _start_server([sys.executable, "-m", "src.start_subprocess.fastmcp_server_script"])
            #logger.info(f"FastMCP server started with PID={self._fastmcp_process.pid}")
            self._start_fastmcp_in_process()
            # New: start the napari-micromanager plugin
            self.add_napari_micromanager.emit()

            # Both servers are ready
            logger.info("Both processes started successfully")
            self.status_update.emit("Both servers ready! You can start VS Code or Cursor.")
            self.servers_ready.emit()

        except Exception as e:
            logger.error(f"Error starting servers: {e}")
            self.status_update.emit(f"Error: {e}")
        finally:
            self.stop_thread.emit()

    def _start_fastmcp_in_process(self):
        """Start FastMCP server in the same process but different thread"""

        def run_fastmcp():

            try:
                logger.info("Initializing agents...}")
                if self._microscope_type == "real":
                    logger.info("Real Microscope...")
                    self._mmc = CMMCorePlus().instance() # get global singleton
                    agents = initialize_agents(mmc=self._mmc, microscope_type=self._microscope_type)
                else:
                    logger.info("Virtual Microscope...")
                    self._mmc = UniMMCore()
                    agents = initialize_agents(mmc=self._mmc, microscope_type=self._microscope_type)

                logger.info("Creating MCP server...")
                # Viewer instance for MCP tools (NOT for execution namespace!)
                # Viewer operations must use thread-safe tools (viewer_screenshot, etc.)
                # not direct calls from executed code (causes Qt/OpenGL threading errors)
                executor = agents["executor"]
                viewer_instance = NapariViewerMC(self._viewer)
                # DO NOT call: executor.set_viewer(viewer_instance) - causes threading issues
                # Create event cache
                event_cache = MicroscopeEventCache(self._mmc)
                
                mcp_server = create_mcp_server(
                    database_agent=agents["database_agent"],
                    microscope_status=agents["microscope_status"],
                    no_coding_agent=agents["no_coding_agent"],
                    executor=executor,
                    logger_agent=agents["logger_agent"], 
                    viewer=viewer_instance,
                    event_cache=event_cache
                    viewer_proxy=self._viewer_proxy  # Pass the proxy created on main thread
                )

                # Run the server
                logger.info("Starting FastMCP server...")
                run_server(mcp_server)

            except Exception as e:
                logger.exception(f"FastMCP error: {e}")

        self._fastmcp_thread = threading.Thread(target=run_fastmcp, daemon=True)
        self._fastmcp_thread.start()


    @pyqtSlot()
    def stop_mcp_server(self):
        """
        Stop the MCP server
        """
        self.status_update.emit("Stopping servers...")

        try:
            if sys.platform.startswith("win"):
                # Stop Elasticsearch
                self.status_update.emit("Stopping Elasticsearch server...")
                subprocess.call(["taskkill", "/F", "/IM", "java.exe"])
                logger.info("Stopped Elasticsearch")

            else:
                self.status_update.emit("Stopping Elasticsearch server...")
                #os.kill(self._elastic_search_process.pid, signal.SIGTERM)
                logger.info("Stopped Elasticsearch")
                os.killpg(os.getpgid(self._elastic_search_process.pid), signal.SIGTERM)

            time.sleep(5)

            # shutdown all process
            # TODO:
            #  For the moment we don't handle the running of the server so we don't have access the the server instance of uvicorn
            #  because of this we just stop all process
            # Stop FastMCP
            os.kill(os.getpid(), signal.SIGTERM)
            logger.info("All servers stopped")
            self.status_update.emit("All servers stopped")
            self.servers_stopped.emit()

        except Exception as e:
            logger.error(f"Error stopping servers: {e}")
            self.status_update.emit(f"Error stopping servers: {e}")
            self.servers_stopped.emit()


class MCPServer(QWidget):

    def __init__(self):
        super().__init__()
        self.mmc = None
        self.viewer = napari.current_viewer()

        print(napari.current_viewer())

        # ---GUI----
        self.setObjectName("MCPServer")
        self.resize(300, 150)
        self.setWindowTitle("MCP Server - Microscope Toolset") # maybe change the name

        main_layout = QVBoxLayout(self)

        # Title label
        label_title = QLabel()
        label_title.setText("Microscope Toolset MCP Server")
        label_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")

        # Microscope type selection
        microscope_label = QLabel("Microscope type selection:")
        microscope_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        microscope_label.setStyleSheet("font-size: 12px; margin-bottom: 5px;")

        self.microscope_combo = QComboBox()
        self.microscope_combo.addItems(["Select microscope type...", "Virtual Microscope", "Real Microscope"])
        self.microscope_combo.setStyleSheet(
            "QComboBox { "
            "    border-radius: 8px; "
            "    padding: 8px 12px; "
            "    font-size: 12px; "
            "} "
            "QComboBox::drop-down { "
            "    border: none; "
            "} "
            "QComboBox::down-arrow { "
            "    width: 12px; "
            "    height: 12px; "
            "}"
        )
        # "    background-color: white; "
        #  "    border: 2px solid #cccccc; "
        self.microscope_combo.currentTextChanged.connect(self.on_microscope_type_changed)


        # Status label
        self.status_label = QLabel()
        self.status_label.setText("Ready to start servers")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 12px; color: #666; margin-bottom: 15px; padding: 5px;")

        # Start button
        self.start_button = QPushButton()
        self.start_button.setText("Start Servers")
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; font-size: 12px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.start_button.setEnabled(False)

        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setText("Stop Servers")
        self.stop_button.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; border-radius: 8px; padding: 10px 20px; font-size: 12px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.stop_button.setEnabled(False)

        # Add components to layout
        main_layout.addWidget(label_title)
        main_layout.addWidget(microscope_label)
        main_layout.addWidget(self.microscope_combo, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Connect signals
        self.start_button.clicked.connect(self.click_start_server)
        self.stop_button.clicked.connect(self.click_stop_server)

        # Initial button states
        self.stop_button.setEnabled(False)

        # QThread setup
        self.mcp_thread = None
        self.mcp_worker = None
        self._current_microscope_type = None
        self._viewer_proxy = None  # Store proxy as instance variable
        #self.mcp_worker.moveToThread(self.mcp_thread)

        # Connect worker signals
        #self.mcp_thread.started.connect(self.mcp_worker.run_mcp_server)
        #self.mcp_worker.stop_thread.connect(self.mcp_thread.quit)

        # Connect status update signals
        #self.mcp_worker.status_update.connect(self.update_status_label)
        #self.mcp_worker.servers_ready.connect(self.on_servers_ready)
        #self.mcp_worker.servers_stopped.connect(self.on_servers_stopped)

    def on_microscope_type_changed(self, text: str):
        """Handle microscope type selection"""
        if text == "Virtual Microscope":
            self._current_microscope_type = "virtual"
            self.start_button.setEnabled(True)
            self.status_label.setText("Virtual microscope selected - Ready to start servers")
            self.status_label.setStyleSheet(
                "font-size: 12px; color: #2196F3; margin-bottom: 15px; padding: 5px; font-weight: bold;")
            logger.info("Virtual microscope selected")

        elif text == "Real Microscope":
            self._current_microscope_type = "real"
            self.start_button.setEnabled(True)
            self.status_label.setText("Real microscope selected - Ready to start servers")
            self.status_label.setStyleSheet(
                "font-size: 12px; color: #4CAF50; margin-bottom: 15px; padding: 5px; font-weight: bold;")
            logger.info("Real microscope selected")

        else:
            self._current_microscope_type = None
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Please select a microscope type to continue")
            self.status_label.setStyleSheet("font-size: 12px; color: #666; margin-bottom: 15px; padding: 5px;")



    def click_start_server(self):
        """Handle start button click"""
        if self._current_microscope_type is None:
            return

        # Create viewer proxy on main thread (before moving worker)
        # Store as instance variable to prevent garbage collection
        self._viewer_proxy = ThreadSafeViewerProxy(self.viewer)

        # Create new Worker and thread for each start
        self.mcp_thread = QThread()
        self.mcp_worker = MCPWorker(
            microscope_type=self._current_microscope_type, 
            viewer=NapariViewerMC(self.viewer),
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
        self.microscope_combo.setEnabled(False)
        self.status_label.setText(f"Starting servers for {self._current_microscope_type} microscope...")

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
        if "ready" in message.lower() or "you can start" in message.lower():
            self.status_label.setStyleSheet(
                "font-size: 12px; color: #4CAF50; margin-bottom: 15px; padding: 5px; font-weight: bold;")
        elif "error" in message.lower() or "failed" in message.lower():
            self.status_label.setStyleSheet(
                "font-size: 12px; color: #f44336; margin-bottom: 15px; padding: 5px; font-weight: bold;")
        elif "loading" in message.lower() or "starting" in message.lower() or "stopping" in message.lower() or "waiting" in message.lower():
            self.status_label.setStyleSheet(
                "font-size: 12px; color: #FF9800; margin-bottom: 15px; padding: 5px; font-weight: bold;")
        elif "virtual" in message.lower():
            self.status_label.setStyleSheet(
                "font-size: 12px; color: #2196F3; margin-bottom: 15px; padding: 5px; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("font-size: 12px; color: #666; margin-bottom: 15px; padding: 5px;")

    @pyqtSlot()
    def on_servers_ready(self):
        """Called when both servers are ready"""
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

    @pyqtSlot()
    def on_servers_stopped(self):
        """Called when both servers are stopped"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.microscope_combo.setEnabled(True)

        if self._current_microscope_type == "virtual":
            self.status_label.setText("Virtual microscope - Ready to start servers")
            self.status_label.setStyleSheet("font-size: 12px; color: #2196F3; margin-bottom: 15px; padding: 5px;")
        else:
            self.status_label.setText("Real microscope - Ready to start servers")
            self.status_label.setStyleSheet("font-size: 12px; color: #4CAF50; margin-bottom: 15px; padding: 5px;")

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
        print("MCP Server window closed")