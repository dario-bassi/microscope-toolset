import os
import sys
import logging
import threading
import time
from datetime import datetime
from typing import Any

import napari
from PyQt6.QtCore import Qt, QObject, pyqtSlot, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSizePolicy, QFrame, QLineEdit, QGroupBox, QMessageBox,
)
from src.mcp_microscopetoolset.utils import get_user_information
from src.start_subprocess.servers import _start_server, _stop_server, wait_for_es
from src.mcp_microscopetoolset.server_setup import create_mcp_server
from src.mcp_microscopetoolset.agents_init import initialize_agents
from src.mcp_microscopetoolset.viewer import NapariViewerMC
from src.microscope.microscope_event_cache import MicroscopeEventCache
from src.benchmarking.benchmark_logger import BenchmarkLogger
from src.utils.cfg_utils import classify_cfg as _classify_cfg
from src.utils.core_proxy_worker import CoreProxyWorker

logger = logging.getLogger("MCPServer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

# ── Shared style constants ──────────────────────────────────────────────────
_DOT_STOPPED  = "background:#aaaaaa;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_DOT_BUSY     = "background:#FF9800;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_DOT_OK       = "background:#4CAF50;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_DOT_ERROR    = "background:#f44336;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_BTN_GREEN    = ("QPushButton{background:#4CAF50;color:white;border-radius:4px;"
                 "padding:2px 8px;font-size:11px;}"
                 "QPushButton:hover{background:#45a049;}"
                 "QPushButton:disabled{background:#cccccc;color:#666;}")
_BTN_RED      = ("QPushButton{background:#f44336;color:white;border-radius:4px;"
                 "padding:2px 8px;font-size:11px;}"
                 "QPushButton:hover{background:#da190b;}"
                 "QPushButton:disabled{background:#cccccc;color:#666;}")


# ── Reusable status panel ───────────────────────────────────────────────────

class ServicePanel(QFrame):
    """[dot] Name  url/msg  [Start | Stop]"""

    def __init__(self, name: str, start_label: str = "Start",
                 stop_label: str = "Stop", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._start_label = start_label
        self._stop_label  = stop_label

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("font-size:11px;font-weight:bold;")
        name_lbl.setFixedWidth(90)

        self._url_lbl = QLabel("")
        self._url_lbl.setStyleSheet("font-size:10px;color:#555;")
        self._url_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.btn = QPushButton(start_label)
        self.btn.setFixedWidth(72)

        row.addWidget(self._dot)
        row.addWidget(name_lbl)
        row.addWidget(self._url_lbl)
        row.addWidget(self.btn)

        self.set_stopped()

    def set_stopped(self):
        self._dot.setStyleSheet(_DOT_STOPPED)
        self._url_lbl.setText("")
        self.btn.setText(self._start_label)
        self.btn.setStyleSheet(_BTN_GREEN)
        self.btn.setEnabled(True)

    def set_busy(self, msg: str = ""):
        self._dot.setStyleSheet(_DOT_BUSY)
        self._url_lbl.setText(msg or "…")
        self.btn.setEnabled(False)

    def set_connected(self, url: str = ""):
        self._dot.setStyleSheet(_DOT_OK)
        self._url_lbl.setText(url)
        self.btn.setText(self._stop_label)
        self.btn.setStyleSheet(_BTN_RED)
        self.btn.setEnabled(True)

    def set_error(self, msg: str = ""):
        self._dot.setStyleSheet(_DOT_ERROR)
        self._url_lbl.setText(msg[:48])
        self.btn.setText(self._start_label)
        self.btn.setStyleSheet(_BTN_GREEN)
        self.btn.setEnabled(True)


# ── Thread-safe viewer proxy ────────────────────────────────────────────────

class ThreadSafeViewerProxy(QObject):
    execute_on_main_thread = pyqtSignal(str, dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.result = None
        self.error  = None
        self.done_event = threading.Event()
        self.execute_on_main_thread.connect(
            self._execute_viewer_method,
            type=Qt.ConnectionType.QueuedConnection,
        )

    @pyqtSlot(str, dict)
    def _execute_viewer_method(self, method_name, kwargs):
        try:
            self.result = getattr(self.viewer, method_name)(**kwargs)
            self.error  = None
        except Exception as e:
            logger.error(f"Viewer method {method_name} failed: {e}")
            self.result = None
            self.error  = e
        finally:
            self.done_event.set()

    def call_on_main_thread(self, method_name, **kwargs):
        self.result = None
        self.error  = None
        self.done_event.clear()
        self.execute_on_main_thread.emit(method_name, kwargs)
        if not self.done_event.wait(timeout=10):
            raise RuntimeError(f"Timeout waiting for {method_name} on main thread")
        if self.error:
            raise self.error
        return self.result


# ── Service workers ─────────────────────────────────────────────────────────

class ElasticsearchWorker(QObject):
    started = pyqtSignal(str)   # url
    error   = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._process = None

    @pyqtSlot()
    def run(self):
        try:
            ui = get_user_information()
            es_home = ui.get("elastic_search_path_home", "")
            if not es_home or not os.path.isdir(es_home):
                self.error.emit("ELASTICSEARCH path not configured in .env")
                return

            exe = (f"{es_home}\\bin\\elasticsearch.bat"
                   if sys.platform.startswith("win")
                   else f"{es_home}/bin/elasticsearch")
            logger.info(f"Launching Elasticsearch: {exe}")
            self._process = _start_server([exe, "-d", "-p", "pid"])
            logger.info(f"Elasticsearch PID={self._process.pid}")

            try:
                wait_for_es(max_wait=60)
            except Exception as e:
                self.error.emit(f"ES did not become ready: {e}")
                return

            url = get_user_information().get("elasticsearch_url", "http://localhost:4500")
            logger.info(f"Elasticsearch ready at {url}")
            self.started.emit(url)

        except Exception as e:
            logger.exception(f"ElasticsearchWorker: {e}")
            self.error.emit(str(e))

    def stop_es(self):
        """May be called from any thread."""
        if self._process is not None:
            try:
                _stop_server(self._process)
                logger.info("Elasticsearch stopped")
            except Exception as e:
                logger.warning(f"Could not stop Elasticsearch: {e}")
            finally:
                self._process = None
        self.stopped.emit()


class PostgreSQLWorker(QObject):
    connected    = pyqtSignal(str)   # "host:port"
    error        = pyqtSignal(str)
    disconnected = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._db_conn = None

    @pyqtSlot()
    def run(self):
        try:
            from src.postqrl.connection import DBConnection
            self._db_conn = DBConnection()
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            self.connected.emit(f"{host}:{port}")
        except Exception as e:
            logger.exception(f"PostgreSQLWorker: {e}")
            self.error.emit(str(e))

    def disconnect_pg(self):
        """May be called from any thread."""
        if self._db_conn is not None:
            try:
                self._db_conn.disconnect()
            except Exception:
                pass
            self._db_conn = None
        self.disconnected.emit()


class MCPServerWorker(QObject):
    started = pyqtSignal(str)   # "http://host:port"
    error   = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, mmc: Any, viewer: Any, viewer_proxy: ThreadSafeViewerProxy,
                 host: str = "127.0.0.1", port: int = 5500):
        super().__init__()
        self._mmc          = mmc
        self._viewer       = viewer
        self._viewer_proxy = viewer_proxy
        self._host         = host
        self._port         = port
        self._uvicorn_server = None
        self._fastmcp_thread = None

    @pyqtSlot()
    def run(self):
        self._stop_uvicorn()

        def _init_mcp():
            try:
                logger.info("Initializing agents…")
                agents = initialize_agents(mmc=self._mmc)

                viewer_instance = NapariViewerMC(self._viewer)
                event_cache     = MicroscopeEventCache(self._mmc)

                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                ui       = get_user_information()
                is_trained   = ui.get("benchmark_agent_enable", "") == "true"
                bench_logger = BenchmarkLogger(
                    agent_type="trained" if is_trained else "untrained",
                    run_id=f"benchmark_{ts}",
                )

                mcp_server = create_mcp_server(
                    database_agent=agents["database_agent"],
                    microscope_status=agents["microscope_status"],
                    executor=agents["executor"],
                    viewer=viewer_instance,
                    event_cache=event_cache,
                    viewer_proxy=self._viewer_proxy,
                    benchmark_logger_instance=bench_logger,
                    host=self._host,
                    port=self._port,
                )

                import uvicorn, anyio
                app    = mcp_server.streamable_http_app()
                config = uvicorn.Config(
                    app,
                    host=self._host,
                    port=self._port,
                    log_level=mcp_server.settings.log_level.lower(),
                )
                self._uvicorn_server = uvicorn.Server(config)
                url = f"http://{self._host}:{self._port}"
                self.started.emit(url)
                try:
                    anyio.run(self._uvicorn_server.serve)
                except OSError as e:
                    if "10048" in str(e) or "address already in use" in str(e).lower():
                        logger.error(f"Port {self._port} still in use")
                        self.error.emit(f"Port {self._port} still in use — try restarting")
                    else:
                        raise

            except Exception as e:
                logger.exception(f"FastMCP error: {e}")
                self.error.emit(str(e))

        self._fastmcp_thread = threading.Thread(target=_init_mcp, daemon=True)
        self._fastmcp_thread.start()

    def _stop_uvicorn(self):
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._fastmcp_thread is not None:
            self._fastmcp_thread.join(timeout=15)
            if self._fastmcp_thread.is_alive():
                logger.warning("MCP thread did not stop in time — port 5500 may still be held")
            self._fastmcp_thread  = None
            self._uvicorn_server  = None

    def stop_mcp(self):
        """May be called from any thread."""
        self._stop_uvicorn()
        self.stopped.emit()


# ── Main widget ─────────────────────────────────────────────────────────────

class MCPServer(QWidget):

    def __init__(self, auto_config: str | None = None):
        super().__init__()
        self._auto_config = auto_config
        self.viewer       = napari.current_viewer()
        self._mmc         = None

        self.setObjectName("MCPServer")
        self.setWindowTitle("MCP Server")

        viewer_mc          = NapariViewerMC(self.viewer)
        self._viewer_proxy = ThreadSafeViewerProxy(viewer_mc)

        # Persistent settings
        self._settings = QSettings("MicroscopeToolset", "MCPServer")

        # napari-micromanager state
        self._mmwin         = None
        self._last_cfg_path = None
        self._in_user_load  = False
        self._proxy_thread  = None
        self._proxy_worker  = None
        self._remote_connected    = False
        self._remote_url          = ""     # stored on connect; used for locality check
        self._cfg_pending_restart = False  # auto-restart MCP after cfg reload

        # service worker/thread pairs
        self._es_worker  = None;  self._es_thread  = None;  self._es_running  = False
        self._pg_worker  = None;  self._pg_thread  = None;  self._pg_running  = False
        self._mcp_worker = None;  self._mcp_thread = None;  self._mcp_running = False

        # ── build UI ──────────────────────────────────────────────────────
        main = QVBoxLayout(self)
        main.setContentsMargins(6, 4, 6, 4)
        main.setSpacing(4)

        title = QLabel("Microscope Toolset")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight:bold;font-size:12px;")
        main.addWidget(title)

        # ── Core group ────────────────────────────────────────────────────
        core_grp = QGroupBox("Microscope Core")
        core_grp.setStyleSheet("QGroupBox{font-size:11px;}")
        core_lay = QVBoxLayout(core_grp)
        core_lay.setContentsMargins(4, 6, 4, 4)
        core_lay.setSpacing(3)

        status_row = QHBoxLayout()
        status_row.setSpacing(5)
        self._core_dot = QLabel()
        self._core_dot.setFixedSize(12, 12)
        self._core_dot.setStyleSheet(_DOT_STOPPED)
        self._core_type_badge = QLabel("")
        self._core_type_badge.setVisible(False)
        self._core_addr_lbl = QLabel("No core loaded")
        self._core_addr_lbl.setStyleSheet("font-size:10px;color:#555;")
        status_row.addWidget(self._core_dot)
        status_row.addWidget(self._core_type_badge)
        status_row.addWidget(self._core_addr_lbl)
        status_row.addStretch()
        core_lay.addLayout(status_row)

        remote_row = QHBoxLayout()
        remote_row.setSpacing(4)
        rl = QLabel("Remote:")
        rl.setStyleSheet("font-size:10px;")
        rl.setFixedWidth(50)
        self._remote_url_edit = QLineEdit()
        self._remote_url_edit.setPlaceholderText("http://127.0.0.1:5601")
        self._remote_url_edit.setStyleSheet("font-size:10px;")
        self._remote_connect_btn = QPushButton("Connect")
        self._remote_connect_btn.setFixedWidth(72)
        self._remote_connect_btn.setStyleSheet(_BTN_GREEN)
        remote_row.addWidget(rl)
        remote_row.addWidget(self._remote_url_edit)
        remote_row.addWidget(self._remote_connect_btn)
        core_lay.addLayout(remote_row)

        proxy_row = QHBoxLayout()
        proxy_row.setSpacing(4)
        pl = QLabel("Proxy port:")
        pl.setStyleSheet("font-size:10px;")
        pl.setFixedWidth(65)
        self._proxy_port_edit = QLineEdit()
        self._proxy_port_edit.setPlaceholderText("5601")
        self._proxy_port_edit.setStyleSheet("font-size:10px;")
        self._proxy_port_edit.setValidator(QIntValidator(1, 65535))
        proxy_row.addWidget(pl)
        proxy_row.addWidget(self._proxy_port_edit)
        proxy_row.addStretch()
        core_lay.addLayout(proxy_row)
        main.addWidget(core_grp)

        # ── Service panels ────────────────────────────────────────────────
        self._es_panel  = ServicePanel("Elasticsearch", "Start",   "Stop")
        self._pg_panel  = ServicePanel("PostgreSQL",    "Connect", "Disconnect")
        self._mcp_panel = ServicePanel("MCP Server",    "Start",   "Stop")
        self._mcp_panel.btn.setEnabled(False)   # enabled once core is ready
        main.addWidget(self._es_panel)
        main.addWidget(self._pg_panel)

        # MCP host / port config row
        mcp_cfg_row = QHBoxLayout()
        mcp_cfg_row.setSpacing(4)
        mcp_lbl = QLabel("MCP:")
        mcp_lbl.setStyleSheet("font-size:10px;")
        mcp_lbl.setFixedWidth(30)
        self._mcp_host_edit = QLineEdit()
        self._mcp_host_edit.setPlaceholderText("127.0.0.1")
        self._mcp_host_edit.setStyleSheet("font-size:10px;")
        colon_lbl = QLabel(":")
        colon_lbl.setStyleSheet("font-size:10px;")
        colon_lbl.setFixedWidth(8)
        self._mcp_port_edit = QLineEdit()
        self._mcp_port_edit.setPlaceholderText("5500")
        self._mcp_port_edit.setMaximumWidth(55)
        self._mcp_port_edit.setStyleSheet("font-size:10px;")
        self._mcp_port_edit.setValidator(QIntValidator(1, 65535))
        mcp_cfg_row.addWidget(mcp_lbl)
        mcp_cfg_row.addWidget(self._mcp_host_edit)
        mcp_cfg_row.addWidget(colon_lbl)
        mcp_cfg_row.addWidget(self._mcp_port_edit)
        main.addLayout(mcp_cfg_row)

        main.addWidget(self._mcp_panel)

        # ── Status bar ────────────────────────────────────────────────────
        self._status_lbl = QLabel("Adding napari-micromanager…")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size:10px;color:#666;padding:2px;")
        self._status_lbl.setWordWrap(True)
        main.addWidget(self._status_lbl)

        # ── Wire signals ──────────────────────────────────────────────────
        self._es_panel.btn.clicked.connect(self._toggle_elasticsearch)
        self._pg_panel.btn.clicked.connect(self._toggle_postgresql)
        self._mcp_panel.btn.clicked.connect(self._toggle_mcp_server)
        self._remote_connect_btn.clicked.connect(self._toggle_remote_core)

        # ── Restore persisted settings ────────────────────────────────────
        self._remote_url_edit.setText(self._settings.value("remote_url", ""))
        self._proxy_port_edit.setText(self._settings.value("proxy_port", "5601"))
        self._mcp_host_edit.setText(self._settings.value("mcp_host", "127.0.0.1"))
        self._mcp_port_edit.setText(self._settings.value("mcp_port", "5500"))

        # ── Add napari-micromanager on next tick ──────────────────────────
        QTimer.singleShot(500, self._add_napari_micromanager)
        if self._auto_config is not None:
            QTimer.singleShot(1500, self._auto_load_config)

    # ── Elasticsearch ───────────────────────────────────────────────────────

    def _toggle_elasticsearch(self):
        if self._es_running:
            self._stop_elasticsearch()
        else:
            self._start_elasticsearch()

    def _start_elasticsearch(self):
        self._es_panel.set_busy("starting…")
        self._set_status("Starting Elasticsearch…")
        self._es_thread = QThread()
        self._es_worker = ElasticsearchWorker()
        self._es_worker.moveToThread(self._es_thread)
        self._es_thread.started.connect(self._es_worker.run)
        self._es_worker.started.connect(self._on_es_started)
        self._es_worker.error.connect(self._on_es_error)
        self._es_thread.start()

    @pyqtSlot(str)
    def _on_es_started(self, url: str):
        self._es_running = True
        self._es_panel.set_connected(url)
        self._set_status("Elasticsearch ready")
        if self._es_thread:
            self._es_thread.quit()

    @pyqtSlot(str)
    def _on_es_error(self, msg: str):
        self._es_running = False
        self._es_panel.set_error(msg)
        self._set_status(f"Elasticsearch error: {msg}")
        if self._es_thread:
            self._es_thread.quit()

    def _stop_elasticsearch(self):
        self._es_panel.set_busy("stopping…")
        if self._es_worker is not None:
            self._es_worker.stopped.connect(self._on_es_stopped)
            self._es_worker.stop_es()
        else:
            self._on_es_stopped()

    @pyqtSlot()
    def _on_es_stopped(self):
        self._es_running = False
        self._es_panel.set_stopped()
        self._set_status("Elasticsearch stopped")

    # ── PostgreSQL ──────────────────────────────────────────────────────────

    def _toggle_postgresql(self):
        if self._pg_running:
            self._stop_postgresql()
        else:
            self._start_postgresql()

    def _start_postgresql(self):
        self._pg_panel.set_busy("connecting…")
        self._set_status("Connecting to PostgreSQL…")
        self._pg_thread = QThread()
        self._pg_worker = PostgreSQLWorker()
        self._pg_worker.moveToThread(self._pg_thread)
        self._pg_thread.started.connect(self._pg_worker.run)
        self._pg_worker.connected.connect(self._on_pg_connected)
        self._pg_worker.error.connect(self._on_pg_error)
        self._pg_thread.start()

    @pyqtSlot(str)
    def _on_pg_connected(self, url: str):
        self._pg_running = True
        self._pg_panel.set_connected(url)
        self._set_status("PostgreSQL connected")
        if self._pg_thread:
            self._pg_thread.quit()

    @pyqtSlot(str)
    def _on_pg_error(self, msg: str):
        self._pg_running = False
        self._pg_panel.set_error(msg)
        self._set_status(f"PostgreSQL error: {msg}")
        if self._pg_thread:
            self._pg_thread.quit()

    def _stop_postgresql(self):
        self._pg_panel.set_busy("disconnecting…")
        if self._pg_worker is not None:
            self._pg_worker.disconnected.connect(self._on_pg_disconnected)
            self._pg_worker.disconnect_pg()
        else:
            self._on_pg_disconnected()

    @pyqtSlot()
    def _on_pg_disconnected(self):
        self._pg_running = False
        self._pg_panel.set_stopped()
        self._set_status("PostgreSQL disconnected")

    # ── MCP Server ──────────────────────────────────────────────────────────

    def _toggle_mcp_server(self):
        if self._mcp_running:
            self._stop_mcp_server()
        else:
            self._start_mcp_server()

    def _start_mcp_server(self):
        if self._mmc is None:
            self._set_status("Load a .cfg file or connect a remote core first")
            return
        host = self._mcp_host_edit.text().strip() or "127.0.0.1"
        port_text = self._mcp_port_edit.text().strip()
        if not port_text.isdigit() or not (1 <= int(port_text) <= 65535):
            self._set_status("Invalid MCP port — enter a number between 1 and 65535")
            return
        port = int(port_text)
        self._settings.setValue("mcp_host", host)
        self._settings.setValue("mcp_port", str(port))
        self._mcp_host_edit.setEnabled(False)
        self._mcp_port_edit.setEnabled(False)
        self._mcp_panel.set_busy("starting…")
        self._set_status("Starting MCP server…")
        self._mcp_thread = QThread()
        self._mcp_worker = MCPServerWorker(
            mmc=self._mmc,
            viewer=self.viewer,
            viewer_proxy=self._viewer_proxy,
            host=host,
            port=port,
        )
        self._mcp_worker.moveToThread(self._mcp_thread)
        self._mcp_thread.started.connect(self._mcp_worker.run)
        self._mcp_worker.started.connect(self._on_mcp_started)
        self._mcp_worker.error.connect(self._on_mcp_error)
        self._mcp_thread.start()

    @pyqtSlot(str)
    def _on_mcp_started(self, url: str):
        self._mcp_running = True
        self._mcp_panel.set_connected(url)
        self._set_status("MCP server ready! Connect from Claude Code.")
        if self._mcp_thread:
            self._mcp_thread.quit()

    @pyqtSlot(str)
    def _on_mcp_error(self, msg: str):
        self._mcp_running = False
        self._mcp_panel.set_error(msg)
        self._set_status(f"MCP error: {msg}")
        if self._mcp_thread:
            self._mcp_thread.quit()

    def _stop_mcp_server(self):
        self._mcp_panel.set_busy("stopping…")
        if self._mcp_worker is not None:
            self._mcp_worker.stopped.connect(self._on_mcp_stopped)
            self._mcp_worker.stop_mcp()
        else:
            self._on_mcp_stopped()

    @pyqtSlot()
    def _on_mcp_stopped(self):
        self._mcp_running = False
        self._mcp_panel.set_stopped()
        self._mcp_host_edit.setEnabled(True)
        self._mcp_port_edit.setEnabled(True)
        if self._cfg_pending_restart:
            self._cfg_pending_restart = False
            self._start_mcp_server()
        else:
            self._set_status("MCP server stopped")

    # ── Remote core ─────────────────────────────────────────────────────────

    def _toggle_remote_core(self):
        if self._remote_connected:
            self._disconnect_remote_core()
        else:
            self._connect_remote_core()

    @staticmethod
    def _query_core_type(url: str) -> str:
        """GET /info from the proxy server; returns the core class name or 'RemoteCore'."""
        import urllib.request, json as _json
        try:
            with urllib.request.urlopen(f"{url}/info", timeout=3) as resp:
                return _json.loads(resp.read()).get("core_type", "RemoteCore")
        except Exception:
            return "RemoteCore"

    def _connect_remote_core(self):
        url = self._remote_url_edit.text().strip() or "http://127.0.0.1:5601"
        self._settings.setValue("remote_url", url)
        self._set_status(f"Connecting to {url}…")
        try:
            from pymmcore_proxy import connect
            remote_core = connect(url)
            self._in_user_load = True
            try:
                if self._mmwin is not None:
                    self._mmwin.set_core(remote_core)
            finally:
                self._in_user_load = False
            self._mmc = remote_core
            self._remote_connected = True
            self._remote_url = url
            core_type = self._query_core_type(url)
            from urllib.parse import urlparse
            p = urlparse(url)
            host_port = f"{p.hostname}:{p.port}"
            self._core_dot.setStyleSheet(_DOT_OK)
            self._set_core_badge(core_type, host_port)
            self._remote_connect_btn.setText("Disconnect")
            self._remote_connect_btn.setStyleSheet(_BTN_RED)
            self._mcp_panel.btn.setEnabled(True)
            self._set_status(f"Remote core connected ({core_type}) — start MCP server")
            logger.info(f"Connected to {core_type} at {url}")
        except Exception as e:
            logger.exception(f"Remote core connection failed: {e}")
            self._core_dot.setStyleSheet(_DOT_ERROR)
            self._clear_core_badge(f"Error: {str(e)[:40]}")
            self._set_status(f"Remote connection failed: {e}")

    def _disconnect_remote_core(self):
        self._mmc = None
        self._remote_connected = False
        self._remote_url = ""
        self._core_dot.setStyleSheet(_DOT_STOPPED)
        self._clear_core_badge()
        self._remote_connect_btn.setText("Connect")
        self._remote_connect_btn.setStyleSheet(_BTN_GREEN)
        self._mcp_panel.btn.setEnabled(False)
        self._set_status("Remote core disconnected")
        logger.info("Remote core disconnected")

    # ── napari-micromanager ─────────────────────────────────────────────────

    def _add_napari_micromanager(self):
        import traceback
        try:
            if self._mmwin is not None:
                logger.info("napari-micromanager already present — skipping")
                self._set_status("Load a .cfg file to initialize the core")
                return

            logger.info("Adding napari-micromanager plugin…")
            self.viewer.window.add_plugin_dock_widget(plugin_name="napari-micromanager")

            from napari_micromanager.main_window import get_main_window
            self._mmwin = get_main_window()
            self._install_load_hook(self._mmwin.core)

            # Patch set_core so the hook survives a CMMCorePlus ↔ UniMMCore swap.
            _orig_set_core = self._mmwin.set_core

            def _patched_set_core(core):
                _orig_set_core(core)
                self._install_load_hook(self._mmwin.core)

            self._mmwin.set_core = _patched_set_core

            self._set_status("Load a .cfg file to initialize the core")
            logger.info("napari-micromanager plugin added")

        except Exception as e:
            logger.error(f"Failed to add napari-micromanager: {e}\n" + traceback.format_exc())
            self._set_status(f"Error adding napari-micromanager: {e}")

    def _install_load_hook(self, core):
        """Wrap core.loadSystemConfiguration to intercept user-initiated cfg loads."""
        _nm_load = core.loadSystemConfiguration

        def _capturing_load(path):
            # Skip if we are already inside a user load (avoids re-entry from
            # napari-micromanager calling loadSystemConfiguration on the new
            # remote core during set_core).
            if self._in_user_load:
                _nm_load(path)
                return
            # Warn and block when a non-local remote core is active.
            # Loading a cfg would silently disconnect from the remote and
            # start a new local proxy — almost certainly not what the user wants.
            if self._remote_connected and self._is_truly_remote(self._remote_url):
                QMessageBox.warning(
                    self,
                    "Remote core is active",
                    f"You are connected to a remote core at <b>{self._remote_url}</b>.<br><br>"
                    "Loading a local .cfg file would disconnect you from the remote "
                    "and start a new local proxy.<br><br>"
                    "Disconnect the remote core first if you want to load a local configuration.",
                )
                logger.warning("Blocked cfg load: truly-remote core is active (%s)", self._remote_url)
                return
            self._in_user_load = True
            try:
                self._last_cfg_path = str(path)
                cfg_type = _classify_cfg(str(path))
                logger.info(f"cfg classification: {cfg_type!r} for {path}")
                if cfg_type == "mixed":
                    logger.error("Mixed C++/Python cfg not supported")
                    self._set_status("Error: mixed C++/Python (#py) cfg not supported yet")
                    return
                self._start_proxy_worker(str(path))
            finally:
                self._in_user_load = False

        core.loadSystemConfiguration = _capturing_load

    def _start_proxy_worker(self, cfg_path: str):
        port_text = self._proxy_port_edit.text().strip()
        proxy_port = int(port_text) if port_text.isdigit() and 1 <= int(port_text) <= 65535 else 5601
        self._settings.setValue("proxy_port", str(proxy_port))
        self._set_status("Starting microscope proxy server…")
        thread = QThread()
        worker = CoreProxyWorker(cfg_path=cfg_path, port=proxy_port)
        worker.moveToThread(thread)
        worker.server_ready.connect(self._on_proxy_ready)
        worker.server_error.connect(self._on_proxy_error)
        thread.started.connect(worker.run)
        self._proxy_thread = thread
        self._proxy_worker = worker
        thread.start()

    @pyqtSlot(str)
    def _on_proxy_ready(self, url: str):
        from pymmcore_proxy import connect
        logger.info(f"Proxy ready at {url} — connecting core")
        self._set_status("Proxy ready — connecting core…")
        remote_core = connect(url)
        self._in_user_load = True
        try:
            if self._mmwin is not None:
                self._mmwin.set_core(remote_core)
        finally:
            self._in_user_load = False
        self._mmc = remote_core
        core_type = self._query_core_type(url)
        from urllib.parse import urlparse
        p = urlparse(url)
        self._core_dot.setStyleSheet(_DOT_OK)
        self._set_core_badge(core_type, f"{p.hostname}:{p.port}")
        if self._mcp_running:
            # cfg reloaded while MCP was live — restart with new core
            self._cfg_pending_restart = True
            self._stop_mcp_server()
        else:
            self._mcp_panel.btn.setEnabled(True)
            self._set_status("Core ready — start MCP server to connect")

    @pyqtSlot(str)
    def _on_proxy_error(self, msg: str):
        logger.error(f"Proxy server failed: {msg}")
        self._core_dot.setStyleSheet(_DOT_ERROR)
        self._clear_core_badge(f"Error: {msg[:40]}")
        self._set_status(f"Proxy error: {msg}")

    def _auto_load_config(self):
        if self._auto_config is None:
            return
        if self._mmwin is None:
            # Plugin not ready yet — retry
            QTimer.singleShot(500, self._auto_load_config)
            return
        logger.info(f"Auto-loading config: {self._auto_config}")
        self._set_status(f"Auto-loading {os.path.basename(self._auto_config)}…")
        self._mmwin.core.loadSystemConfiguration(self._auto_config)

    # ── Core badge helpers ──────────────────────────────────────────────────

    _BADGE_STYLES = {
        "CMMCorePlus": ("CMM+", "background:#1565C0;color:white;border-radius:7px;padding:1px 5px;font-size:10px;font-weight:bold;"),
        "UniMMCore":   ("Uni",  "background:#6A1B9A;color:white;border-radius:7px;padding:1px 5px;font-size:10px;font-weight:bold;"),
    }
    _BADGE_FALLBACK = ("RMC",  "background:#E65100;color:white;border-radius:7px;padding:1px 5px;font-size:10px;font-weight:bold;")

    def _set_core_badge(self, core_type: str, host_port: str) -> None:
        text, style = self._BADGE_STYLES.get(core_type, self._BADGE_FALLBACK)
        self._core_type_badge.setText(text)
        self._core_type_badge.setStyleSheet(style)
        self._core_type_badge.setVisible(True)
        self._core_addr_lbl.setText(host_port)
        self._core_addr_lbl.setStyleSheet("font-size:10px;color:#555;")

    def _clear_core_badge(self, msg: str = "No core loaded") -> None:
        self._core_type_badge.setVisible(False)
        self._core_addr_lbl.setText(msg)
        self._core_addr_lbl.setStyleSheet("font-size:10px;color:#555;")

    # ── Remote-locality helpers ─────────────────────────────────────────────

    # ── Remote-locality helpers ─────────────────────────────────────────────

    @staticmethod
    def _is_truly_remote(url: str) -> bool:
        """Return True when *url* points to a host other than localhost."""
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return host not in ("localhost", "127.0.0.1", "::1", "")

    # ── Utilities ───────────────────────────────────────────────────────────

    def _set_status(self, message: str):
        self._status_lbl.setText(message)
        base = "font-size:10px;padding:2px;"
        msg  = message.lower()
        if any(w in msg for w in ("ready", "connected")):
            self._status_lbl.setStyleSheet(f"{base}color:#4CAF50;font-weight:bold;")
        elif any(w in msg for w in ("error", "failed", "not supported")):
            self._status_lbl.setStyleSheet(f"{base}color:#f44336;font-weight:bold;")
        elif any(w in msg for w in ("starting", "stopping", "connecting", "proxy", "loading", "…")):
            self._status_lbl.setStyleSheet(f"{base}color:#FF9800;font-weight:bold;")
        elif "load a" in msg:
            self._status_lbl.setStyleSheet(f"{base}color:#2196F3;font-weight:bold;")
        else:
            self._status_lbl.setStyleSheet(f"{base}color:#666;")

    def closeEvent(self, event):
        self.hide()
        event.ignore()
