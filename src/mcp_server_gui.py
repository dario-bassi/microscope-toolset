import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any

import napari
from PyQt6.QtCore import QObject, QSettings, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.benchmarking import BenchmarkLogger
from src.databases.es_server import _start_server, _stop_server, wait_for_es
from src.mcp_microscopetoolset import (
    NapariViewerMC,
    create_mcp_server,
    get_user_information,
    initialize_agents,
)
from src.microscope import MicroscopeEventCache
from src.utils import CoreProxyWorker
from src.utils import classify_cfg as _classify_cfg

logger = logging.getLogger("MCPServer")


class StartupTimer:
    """Phase-by-phase wall-clock timer for one startup sequence.

    Usage::
        t = StartupTimer()
        t.mark("phase_a")
        ...
        t.mark("phase_b")
        t.report()   # logs all phases + per-step deltas
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._marks: list[tuple[str, float]] = []

    def mark(self, phase: str) -> float:
        """Record *phase* and return elapsed seconds since T0."""
        elapsed = time.perf_counter() - self._t0
        self._marks.append((phase, elapsed))
        logger.debug("[TIMING] %-42s  t=%.3fs", phase, elapsed)
        return elapsed

    def report(self) -> None:
        """Emit a single INFO-level summary of all phases."""
        if not self._marks:
            return
        lines = ["[TIMING] Startup summary (seconds from T0):"]
        prev = 0.0
        for phase, t in self._marks:
            lines.append(f"  {phase:<42s}  t={t:6.3f}s  (+{t - prev:.3f}s)")
            prev = t
        logger.info("\n".join(lines))


if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

# ── Shared style constants ──────────────────────────────────────────────────
_DOT_STOPPED = "background:#aaaaaa;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_DOT_BUSY = "background:#FF9800;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_DOT_OK = "background:#4CAF50;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_DOT_ERROR = "background:#f44336;border-radius:6px;min-width:12px;min-height:12px;max-width:12px;max-height:12px;"
_BTN_GREEN = (
    "QPushButton{background:#4CAF50;color:white;border-radius:4px;"
    "padding:2px 8px;font-size:11px;}"
    "QPushButton:hover{background:#45a049;}"
    "QPushButton:disabled{background:#cccccc;color:#666;}"
)
_BTN_RED = (
    "QPushButton{background:#f44336;color:white;border-radius:4px;"
    "padding:2px 8px;font-size:11px;}"
    "QPushButton:hover{background:#da190b;}"
    "QPushButton:disabled{background:#cccccc;color:#666;}"
)


# ── Reusable status panel ───────────────────────────────────────────────────


class ServicePanel(QFrame):
    """[dot] Name  url/msg  [Start | Stop]"""

    def __init__(
        self, name: str, start_label: str = "Start", stop_label: str = "Stop", parent=None
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("ServicePanel{border:1px solid #ddd;border-radius:6px;}")
        self._start_label = start_label
        self._stop_label = stop_label

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
        self.error = None
        self.done_event = threading.Event()
        self.execute_on_main_thread.connect(
            self._execute_viewer_method,
            type=Qt.ConnectionType.QueuedConnection,
        )

    @pyqtSlot(str, dict)
    def _execute_viewer_method(self, method_name, kwargs):
        try:
            self.result = getattr(self.viewer, method_name)(**kwargs)
            self.error = None
        except Exception as e:
            logger.error(f"Viewer method {method_name} failed: {e}")
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
            raise RuntimeError(f"Timeout waiting for {method_name} on main thread")
        if self.error:
            raise self.error
        return self.result


# ── Service workers ─────────────────────────────────────────────────────────


class ElasticsearchWorker(QObject):
    started = pyqtSignal(str)  # url
    error = pyqtSignal(str)
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

            exe = (
                f"{es_home}\\bin\\elasticsearch.bat"
                if sys.platform.startswith("win")
                else f"{es_home}/bin/elasticsearch"
            )
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
    connected = pyqtSignal(str)  # "host:port"
    error = pyqtSignal(str)
    disconnected = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._db_conn = None

    @pyqtSlot()
    def run(self):
        try:
            from src.postqrl import DBConnection

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


def _port_in_use(port: int) -> bool:
    """Return True if *port* is currently bound by any process.

    Uses a non-destructive socket probe — works on Windows, macOS, and Linux
    without requiring admin privileges or external tools.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _log_port_listeners(port: int) -> None:
    """Log which process (if any) is still listening on *port* after stop.

    Cross-platform:
      Windows  — netstat -ano | findstr :<port>
      macOS    — lsof -iTCP:<port> -sTCP:LISTEN
      Linux    — ss -tlnp sport = :<port>
    """
    system = platform.system()
    try:
        if system == "Windows":
            out = subprocess.check_output(["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL)
            lines = [ln for ln in out.splitlines() if f":{port}" in ln]
        elif system == "Darwin":
            out = subprocess.check_output(
                ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN"], text=True, stderr=subprocess.DEVNULL
            )
            lines = out.strip().splitlines()
        else:  # Linux
            out = subprocess.check_output(
                ["ss", "-tlnp", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL
            )
            lines = out.strip().splitlines()
        if lines:
            logger.warning("Port %d still has listeners:\n%s", port, "\n".join(lines))
        else:
            logger.info("Port %d is free.", port)
    except Exception as e:
        logger.debug("Could not check port %d listeners: %s", port, e)


class MCPServerWorker(QObject):
    started = pyqtSignal(str)  # "http://host:port"
    error = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        mmc: Any,
        viewer: Any,
        viewer_proxy: ThreadSafeViewerProxy,
        host: str = "127.0.0.1",
        port: int = 5500,
    ):
        super().__init__()
        self._mmc = mmc
        self._viewer = viewer
        self._viewer_proxy = viewer_proxy
        self._host = host
        self._port = port
        self._uvicorn_server = None
        self._fastmcp_thread = None
        self._loop = None  # asyncio event loop owned by _fastmcp_thread

    @pyqtSlot()
    def run(self):
        self._stop_uvicorn()

        def _init_mcp():
            import asyncio

            import uvicorn

            try:
                logger.info("Initializing agents…")
                agents = initialize_agents(mmc=self._mmc)

                viewer_instance = NapariViewerMC(self._viewer)
                event_cache = MicroscopeEventCache(self._mmc)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ui = get_user_information()
                is_trained = ui.get("benchmark_agent_enable", "") == "true"
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

                app = mcp_server.streamable_http_app()
                config = uvicorn.Config(
                    app,
                    host=self._host,
                    port=self._port,
                    log_level=mcp_server.settings.log_level.lower(),
                )
                self._uvicorn_server = uvicorn.Server(config)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                try:
                    url = f"http://{self._host}:{self._port}"
                    self.started.emit(url)
                    loop.run_until_complete(self._uvicorn_server.serve())
                except OSError as e:
                    if "10048" in str(e) or "address already in use" in str(e).lower():
                        logger.error(f"Port {self._port} still in use")
                        self.error.emit(f"Port {self._port} still in use — try restarting")
                    else:
                        raise
                finally:
                    loop.close()
                    self._loop = None

            except Exception as e:
                logger.exception(f"FastMCP error: {e}")
                self.error.emit(str(e))

        self._fastmcp_thread = threading.Thread(target=_init_mcp, daemon=True)
        self._fastmcp_thread.start()

    def _stop_uvicorn(self):
        server = self._uvicorn_server
        loop = self._loop
        if server is not None:
            # force_exit=True makes uvicorn cancel open connections immediately
            # instead of waiting for them to drain — essential for fast port release.
            server.should_exit = True
            server.force_exit = True
            if loop is not None and not loop.is_closed():
                try:
                    # Wake the sleeping asyncio event loop so it checks should_exit
                    # without waiting for the next 0.1 s tick.
                    loop.call_soon_threadsafe(lambda: None)
                except RuntimeError:
                    pass
        if self._fastmcp_thread is not None:
            self._fastmcp_thread.join(timeout=5)
            # The thread may still be alive finishing lifespan cleanup, but uvicorn
            # closes its listen socket early in shutdown — before the lifespan completes.
            # Use the port probe as the authoritative check, not thread liveness.
            if self._fastmcp_thread.is_alive():
                logger.debug(
                    "MCP thread still running lifespan teardown (port may already be free)"
                )
            self._fastmcp_thread = None
            self._uvicorn_server = None
            self._loop = None
            if _port_in_use(self._port):
                # Port is still bound — lifespan hasn't released it yet.
                # Log listeners so the user can identify the holding process.
                logger.warning("Port %d still bound after stop", self._port)
                _log_port_listeners(self._port)
            else:
                logger.info("Port %d released — safe to restart", self._port)

    def stop_mcp(self):
        """May be called from any thread."""
        self._stop_uvicorn()
        self.stopped.emit()


# ── Benchmark worker ────────────────────────────────────────────────────────


class BenchmarkWorker(QObject):
    """Start a test_server.py subprocess and wait until /health responds."""

    ready = pyqtSignal(str)  # base URL once server is up
    error = pyqtSignal(str)

    def __init__(self, test_name: str, host: str, port: int):
        super().__init__()
        self._test_name = test_name
        self._host = host
        self._port = port
        self._process = None

    @pyqtSlot()
    def run(self):
        import urllib.request as _req

        try:
            cmd = [
                sys.executable,
                "-m",
                "src.benchmarking.test_server",
                self._test_name,
                "--host",
                self._host,
                "--port",
                str(self._port),
            ]
            self._process = subprocess.Popen(cmd)

            url = f"http://{self._host}:{self._port}"
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    with _req.urlopen(f"{url}/health", timeout=1.0) as r:  # nosec B310
                        if r.status == 200:
                            self.ready.emit(url)
                            return
                except Exception:
                    pass
                time.sleep(0.5)

            self._kill()
            self.error.emit("Test server did not start within 30 s")
        except Exception as e:
            self._kill()
            self.error.emit(str(e))

    def stop(self):
        self._kill()

    def _kill(self):
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None


# ── Main widget ─────────────────────────────────────────────────────────────


class MCPServer(QWidget):
    def __init__(self, auto_config: str | None = None):
        super().__init__()
        self._startup_timer = StartupTimer()
        self._auto_config = auto_config
        self.viewer = napari.current_viewer()
        self._mmc = None

        self.setObjectName("MCPServer")
        self.setWindowTitle("MCP Server")
        self.setMaximumWidth(450)

        viewer_mc = NapariViewerMC(self.viewer)
        self._viewer_proxy = ThreadSafeViewerProxy(viewer_mc)

        # Persistent settings
        self._settings = QSettings("MicroscopeToolset", "MCPServer")

        # napari-micromanager state
        self._mmwin = None
        self._last_cfg_path = None
        self._in_user_load = False
        self._proxy_thread = None
        self._proxy_worker = None
        self._remote_connected = False
        self._remote_url = ""  # stored on connect; used for locality check
        self._cfg_pending_restart = False  # auto-restart MCP after cfg reload

        # service worker/thread pairs
        self._es_worker = None
        self._es_thread = None
        self._es_running = False
        self._pg_worker = None
        self._pg_thread = None
        self._pg_running = False
        self._mcp_worker = None
        self._mcp_thread = None
        self._mcp_running = False
        self._bench_worker = None
        self._bench_thread = None
        self._bench_running = False

        # experiment tracking
        self._tracking_active = False
        self._tracking_name = ""
        self._tracking_workspace = None  # Path to active workspace folder

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
        self._es_panel = ServicePanel("Elasticsearch", "Start", "Stop")
        self._pg_panel = ServicePanel("PostgreSQL", "Connect", "Disconnect")
        self._mcp_panel = ServicePanel("MCP Server", "Start", "Stop")
        self._mcp_panel.btn.setEnabled(False)  # enabled once core is ready
        main.addWidget(self._es_panel)
        main.addWidget(self._pg_panel)

        # Hide ES/PG panels when the services are not configured in the environment
        try:
            _es_home = get_user_information().get("elastic_search_path_home", "")
        except Exception:
            _es_home = ""
        if not _es_home:
            self._es_panel.setVisible(False)
        try:
            _pg_host = get_user_information().get("db_host", "") or os.getenv("DB_HOST", "")
        except Exception:
            _pg_host = os.getenv("DB_HOST", "")
        if not _pg_host:
            self._pg_panel.setVisible(False)

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

        # Group config row + service panel under a QGroupBox (matches core/bench/tracking border)
        mcp_grp = QGroupBox("MCP Server")
        mcp_grp.setStyleSheet("QGroupBox{font-size:11px;}")
        mcp_grp_lay = QVBoxLayout(mcp_grp)
        mcp_grp_lay.setContentsMargins(4, 6, 4, 4)
        mcp_grp_lay.setSpacing(3)
        mcp_grp_lay.addLayout(mcp_cfg_row)
        self._mcp_panel.setFrameShape(QFrame.Shape.NoFrame)
        self._mcp_panel.setStyleSheet("")  # border provided by mcp_grp
        mcp_grp_lay.addWidget(self._mcp_panel)
        main.addWidget(mcp_grp)

        # ── Benchmarking group ────────────────────────────────────────────
        bench_grp = QGroupBox("Benchmarking")
        bench_grp.setStyleSheet("QGroupBox{font-size:11px;}")
        bench_lay = QVBoxLayout(bench_grp)
        bench_lay.setContentsMargins(4, 6, 4, 4)
        bench_lay.setSpacing(3)

        bench_sel_row = QHBoxLayout()
        bench_sel_row.setSpacing(4)
        tl = QLabel("Test:")
        tl.setStyleSheet("font-size:10px;")
        tl.setFixedWidth(30)
        self._bench_combo = QComboBox()
        self._bench_combo.setStyleSheet("font-size:10px;")
        bench_port_lbl = QLabel("Port:")
        bench_port_lbl.setStyleSheet("font-size:10px;")
        bench_port_lbl.setFixedWidth(28)
        self._bench_port_edit = QLineEdit()
        self._bench_port_edit.setPlaceholderText("5602")
        self._bench_port_edit.setMaximumWidth(50)
        self._bench_port_edit.setStyleSheet("font-size:10px;")
        self._bench_port_edit.setValidator(QIntValidator(1, 65535))
        bench_sel_row.addWidget(tl)
        bench_sel_row.addWidget(self._bench_combo)
        bench_sel_row.addWidget(bench_port_lbl)
        bench_sel_row.addWidget(self._bench_port_edit)
        bench_lay.addLayout(bench_sel_row)

        self._bench_panel = ServicePanel("Test Server", "Launch", "Stop")
        self._bench_panel.setFrameShape(QFrame.Shape.NoFrame)
        self._bench_panel.setStyleSheet("")  # border provided by bench_grp
        bench_lay.addWidget(self._bench_panel)

        self._bench_info_lbl = QLabel("")
        self._bench_info_lbl.setStyleSheet(
            "font-size:10px;color:#555;padding:2px 4px;" "background:#f5f5f5;border-radius:3px;"
        )
        self._bench_info_lbl.setWordWrap(True)
        self._bench_info_lbl.setVisible(False)
        bench_lay.addWidget(self._bench_info_lbl)

        main.addWidget(bench_grp)

        # ── Experiment Tracking group ──────────────────────────────────────
        track_grp = QGroupBox("Experiment Tracking")
        track_grp.setStyleSheet("QGroupBox{font-size:11px;}")
        track_lay = QVBoxLayout(track_grp)
        track_lay.setContentsMargins(4, 6, 4, 4)
        track_lay.setSpacing(3)

        track_row = QHBoxLayout()
        track_row.setSpacing(4)
        self._track_dot = QLabel()
        self._track_dot.setFixedSize(12, 12)
        self._track_dot.setStyleSheet(_DOT_STOPPED)
        track_name_lbl = QLabel("Name:")
        track_name_lbl.setStyleSheet("font-size:10px;")
        track_name_lbl.setFixedWidth(35)
        self._track_name_edit = QLineEdit()
        self._track_name_edit.setPlaceholderText("experiment_name  (optional)")
        self._track_name_edit.setStyleSheet("font-size:10px;")
        self._track_start_btn = QPushButton("Start Tracking")
        self._track_start_btn.setStyleSheet(_BTN_GREEN)
        self._track_stop_btn = QPushButton("Stop & Save")
        self._track_stop_btn.setStyleSheet(_BTN_RED)
        self._track_stop_btn.setEnabled(False)
        track_row.addWidget(self._track_dot)
        track_row.addWidget(track_name_lbl)
        track_row.addWidget(self._track_name_edit)
        track_row.addWidget(self._track_start_btn)
        track_row.addWidget(self._track_stop_btn)
        track_lay.addLayout(track_row)

        info_row = QHBoxLayout()
        info_row.setSpacing(4)
        self._track_info_lbl = QLabel("")
        self._track_info_lbl.setStyleSheet(
            "font-size:10px;color:#555;padding:2px 4px;" "background:#f5f5f5;border-radius:3px;"
        )
        self._track_info_lbl.setWordWrap(True)
        self._track_info_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._track_open_btn = QPushButton("Open")
        self._track_open_btn.setFixedWidth(44)
        self._track_open_btn.setStyleSheet(
            "QPushButton{font-size:10px;padding:1px 4px;border-radius:3px;"
            "background:#e0e0e0;color:#333;}"
            "QPushButton:hover{background:#bdbdbd;}"
            "QPushButton:disabled{background:#f0f0f0;color:#aaa;}"
        )
        self._track_open_btn.setEnabled(False)
        info_row.addWidget(self._track_info_lbl)
        info_row.addWidget(self._track_open_btn)
        track_lay.addLayout(info_row)

        main.addWidget(track_grp)

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
        self._bench_panel.btn.clicked.connect(self._toggle_benchmark)
        self._track_start_btn.clicked.connect(self._start_tracking)
        self._track_stop_btn.clicked.connect(self._stop_tracking)
        self._track_open_btn.clicked.connect(self._open_workspace)

        # ── Restore persisted settings ────────────────────────────────────
        # For proxy host/port: cascade QSettings → .env → hardcoded default.
        # This lets the lab admin set PROXY_CORE_HOST/PORT in .env so the
        # fields are pre-filled on first run; a user who changes them manually
        # keeps their own value (stored in QSettings) on subsequent runs.
        try:
            _ui = get_user_information()
            _env_host = _ui.get("proxy_core_host") or "127.0.0.1"
            _env_port = _ui.get("proxy_core_port") or "5601"
        except Exception:
            _env_host, _env_port = "127.0.0.1", "5601"
        _env_url = f"http://{_env_host}:{_env_port}"

        self._remote_url_edit.setText(self._settings.value("remote_url") or _env_url)
        self._proxy_port_edit.setText(self._settings.value("proxy_port") or _env_port)
        self._mcp_host_edit.setText(self._settings.value("mcp_host", "127.0.0.1"))
        self._mcp_port_edit.setText(self._settings.value("mcp_port", "5500"))
        self._bench_port_edit.setText(self._settings.value("bench_port", "5602"))

        # ── Populate test catalog ─────────────────────────────────────────
        self._refresh_bench_tests()

        # ── Add napari-micromanager on next tick ──────────────────────────
        self._startup_timer.mark("widget_init_done")
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
        if _port_in_use(port):
            self._mcp_panel.set_error(f"Port {port} still in use — wait a moment and retry")
            self._set_status(
                f"Port {port} is still bound — previous server may still be shutting down"
            )
            _log_port_listeners(port)
            return
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
        # Release references so the old worker/thread are GC'd and cannot
        # interfere with the next start (avoids double-stop on restart).
        self._mcp_worker = None
        self._mcp_thread = None
        if self._cfg_pending_restart:
            self._cfg_pending_restart = False
            self._start_mcp_server()
        else:
            if self._mmc is None:
                self._mcp_panel.btn.setEnabled(False)
            self._set_status("MCP server stopped")

    # ── Benchmarking ────────────────────────────────────────────────────────

    def _refresh_bench_tests(self):
        """Populate the test combo from src/benchmarking/test_*/."""
        try:
            from src.benchmarking import list_tests

            tests = list_tests()
        except Exception:
            tests = []
        self._bench_combo.clear()
        if not tests:
            self._bench_combo.addItem("No tests found in src/benchmarking/test_*/")
            self._bench_combo.setEnabled(False)
            self._bench_panel.btn.setEnabled(False)
        else:
            _MAX = 45
            for t in tests:
                label = f"{t['name']}  —  {t['title']}" if t["title"] else t["name"]
                display = label if len(label) <= _MAX else label[: _MAX - 1] + "…"
                self._bench_combo.addItem(display, userData=t["name"])
                self._bench_combo.setItemData(
                    self._bench_combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole
                )
            self._bench_combo.setEnabled(True)
            self._bench_panel.btn.setEnabled(True)

    def _toggle_benchmark(self):
        if self._bench_running:
            self._stop_benchmark()
        else:
            self._launch_benchmark()

    def _launch_benchmark(self):
        idx = self._bench_combo.currentIndex()
        test_name = self._bench_combo.itemData(idx)
        if not test_name:
            self._set_status("Select a test first")
            return
        port_text = self._bench_port_edit.text().strip()
        port = int(port_text) if port_text.isdigit() and 1 <= int(port_text) <= 65535 else 5602
        self._settings.setValue("bench_port", str(port))
        self._bench_combo.setEnabled(False)
        self._bench_port_edit.setEnabled(False)
        self._bench_panel.set_busy("starting…")
        self._set_status(f"Launching test '{test_name}' on port {port}…")
        self._bench_thread = QThread()
        self._bench_worker = BenchmarkWorker(test_name=test_name, host="127.0.0.1", port=port)
        self._bench_worker.moveToThread(self._bench_thread)
        self._bench_thread.started.connect(self._bench_worker.run)
        self._bench_worker.ready.connect(self._on_bench_ready)
        self._bench_worker.error.connect(self._on_bench_error)
        self._bench_thread.start()

    @pyqtSlot(str)
    def _on_bench_ready(self, url: str):
        self._bench_running = True
        self._bench_panel.set_connected(url)
        if self._bench_thread:
            self._bench_thread.quit()
        # Auto-connect the agent to the test server as a remote core
        self._remote_url_edit.setText(url)
        self._connect_remote_core()
        self._fetch_bench_info(url)
        self._set_status(f"Test server ready — agent connected to {url}")

    def _fetch_bench_info(self, url: str):
        import json as _json
        import urllib.request

        try:
            with urllib.request.urlopen(f"{url}/test/info", timeout=3) as r:  # nosec B310
                info = _json.loads(r.read())
            title = info.get("title", "")
            channels = ", ".join(info.get("channels", []))
            desc = (info.get("description", "") or "").strip().split("\n")[0]
            text = title
            if channels:
                text += f"  |  Channels: {channels}"
            if desc:
                text += f"\n{desc}"
            self._bench_info_lbl.setText(text)
            self._bench_info_lbl.setVisible(True)
        except Exception:
            self._bench_info_lbl.setVisible(False)

    @pyqtSlot(str)
    def _on_bench_error(self, msg: str):
        self._bench_running = False
        self._bench_combo.setEnabled(True)
        self._bench_port_edit.setEnabled(True)
        self._bench_panel.set_error(msg)
        self._set_status(f"Test server error: {msg}")
        if self._bench_thread:
            self._bench_thread.quit()

    def _stop_benchmark(self):
        self._bench_panel.set_busy("stopping…")
        if self._bench_worker is not None:
            self._bench_worker.stop()
        self._on_bench_stopped()

    def _on_bench_stopped(self):
        self._bench_running = False
        self._bench_panel.set_stopped()
        self._bench_combo.setEnabled(True)
        self._bench_port_edit.setEnabled(True)
        self._bench_info_lbl.setVisible(False)
        if self._remote_connected:
            self._disconnect_remote_core()
        self._set_status("Test server stopped")

    # ── Experiment tracking ─────────────────────────────────────────────────

    def _start_tracking(self):
        from src.benchmarking import start_experiment

        name = self._track_name_edit.text().strip() or None
        try:
            exp_name, workspace = start_experiment(name)
            self._tracking_active = True
            self._tracking_name = exp_name
            self._tracking_workspace = workspace
            self._track_dot.setStyleSheet(_DOT_OK)
            self._track_name_edit.setEnabled(False)
            self._track_start_btn.setEnabled(False)
            self._track_stop_btn.setEnabled(True)
            self._track_info_lbl.setText(
                f"Tracking: {exp_name}  |  workspace: …/{workspace.parent.name}/{workspace.name}"
            )
            self._track_info_lbl.setVisible(True)
            self._track_open_btn.setEnabled(True)
            self._set_status(f"Experiment tracking started: {exp_name}")
            logger.info(f"Experiment tracking started: {exp_name}, workspace: {workspace}")
        except Exception as e:
            self._track_dot.setStyleSheet(_DOT_ERROR)
            self._set_status(f"Tracking start failed: {e}")
            logger.exception(f"Tracking start failed: {e}")

    def _stop_tracking(self):
        from src.benchmarking import end_experiment

        self._track_dot.setStyleSheet(_DOT_BUSY)
        self._track_stop_btn.setEnabled(False)
        try:
            exp_dir = end_experiment()
            self._tracking_active = False
            self._tracking_name = ""
            self._tracking_workspace = exp_dir / "workspace"
            self._track_dot.setStyleSheet(_DOT_STOPPED)
            self._track_name_edit.setEnabled(True)
            self._track_name_edit.clear()
            self._track_start_btn.setEnabled(True)
            self._track_info_lbl.setText(f"Saved: {exp_dir.name}")
            self._track_info_lbl.setVisible(True)
            self._track_open_btn.setEnabled(True)  # keep open enabled to browse saved experiment
            self._set_status(f"Experiment saved: {exp_dir.name}")
            logger.info(f"Experiment saved to: {exp_dir}")
        except FileNotFoundError as e:
            self._track_dot.setStyleSheet(_DOT_ERROR)
            self._track_stop_btn.setEnabled(True)
            self._set_status(f"Tracking stop failed: {e}")
            logger.error(f"Tracking stop failed: {e}")
        except Exception as e:
            self._track_dot.setStyleSheet(_DOT_ERROR)
            self._track_stop_btn.setEnabled(True)
            self._set_status(f"Tracking save failed: {e}")
            logger.exception(f"Tracking save failed: {e}")

    def _open_workspace(self):
        """Open the experiment workspace folder in the system file explorer."""
        target = self._tracking_workspace
        if target is None or not target.exists():
            self._set_status("Workspace folder not found")
            return
        import subprocess as _sp

        if sys.platform == "win32":
            _sp.Popen(["explorer", str(target)])
        elif sys.platform == "darwin":
            _sp.Popen(["open", str(target)])
        else:
            _sp.Popen(["xdg-open", str(target)])

    # ── Remote core ─────────────────────────────────────────────────────────

    def _toggle_remote_core(self):
        if self._remote_connected:
            self._disconnect_remote_core()
        else:
            self._connect_remote_core()

    @staticmethod
    def _query_core_type(url: str) -> str:
        """GET /info from the proxy server; returns the core class name or 'RemoteCore'."""
        import json as _json
        import urllib.request

        try:
            with urllib.request.urlopen(f"{url}/info", timeout=3) as resp:  # nosec B310
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
            # Widgets rebuilt by set_core connect to systemConfigurationLoaded to
            # populate themselves (channels, presets, shutters, stages, …).  The
            # remote server already has a config loaded, so that event will never
            # arrive via WebSocket — emit it manually so they refresh.
            try:
                remote_core.events.systemConfigurationLoaded.emit()
            except Exception:
                pass
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
            if self._mcp_running:
                self._cfg_pending_restart = True
                self._stop_mcp_server()
                self._set_status(f"Remote core connected ({core_type}) — restarting MCP server…")
            else:
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
        if self._mcp_running:
            self._stop_mcp_server()

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
            logger.info(
                f"[NMM] MainWindow found: {type(self._mmwin).__name__}, "
                f"initial core: {type(self._mmwin.core).__name__}"
            )

            # _install_load_hook works only for local cores (CMMCorePlus / UniMMCore).
            # RemoteMMCore.__getattribute__ checks _RPC_FORWARD_METHODS before the
            # instance dict, so `core.loadSystemConfiguration = fn` is silently ignored.
            # This hook therefore only covers the _auto_load_config path (initial core).
            # The UI Load button is intercepted by _patch_config_widget_load instead.
            self._install_load_hook(self._mmwin.core)
            self._patch_config_widget_load()

            # Patch set_core so both hooks survive every CMMCorePlus↔UniMMCore↔Remote swap.
            _orig_set_core = self._mmwin.set_core

            def _patched_set_core(core):
                logger.info(f"[NMM] set_core called with {type(core).__name__}")
                _orig_set_core(core)
                logger.info(
                    f"[NMM] set_core complete — reinstalling hooks on "
                    f"{type(self._mmwin.core).__name__}"
                )
                self._install_load_hook(self._mmwin.core)
                self._patch_config_widget_load()

            self._mmwin.set_core = _patched_set_core

            self._startup_timer.mark("napari_mm_plugin_added")
            self._set_status("Load a .cfg file to initialize the core")
            logger.info("napari-micromanager plugin added")

        except Exception as e:
            logger.error(f"Failed to add napari-micromanager: {e}\n" + traceback.format_exc())
            self._set_status(f"Error adding napari-micromanager: {e}")

    def _install_load_hook(self, core):
        """Intercept loadSystemConfiguration on LOCAL cores (CMMCorePlus / UniMMCore).

        This hook only works for local cores.  RemoteMMCore.__getattribute__ checks
        _RPC_FORWARD_METHODS before the instance dict, so setting an instance
        attribute is silently ignored — the RPC proxy is always returned instead.
        The UI Load button is intercepted by _patch_config_widget_load for that case.

        This hook covers _auto_load_config, which calls core.loadSystemConfiguration
        directly on the initial local CMMCorePlus before any proxy is running.
        """

        def _capturing_load(path):
            logger.info(
                f"[InstallHook] _capturing_load entered: path={path!r}, "
                f"_in_user_load={self._in_user_load}"
            )
            if self._in_user_load:
                logger.debug("[InstallHook] re-entry guard — skipping")
                return

            new_type = _classify_cfg(str(path))
            logger.info(f"[InstallHook] cfg type={new_type!r}")
            if new_type == "mixed":
                logger.error("[InstallHook] Mixed C++/Python cfg not supported")
                self._set_status("Error: mixed C++/Python (#py) cfg not supported yet")
                return

            current_type = (
                _classify_cfg(self._last_cfg_path)
                if self._proxy_worker is not None and self._last_cfg_path
                else None
            )
            logger.info(f"[InstallHook] current_type={current_type!r} → new_type={new_type!r}")
            self._last_cfg_path = str(path)
            self._start_proxy_worker(str(path))

        core.loadSystemConfiguration = _capturing_load
        logger.info(f"[InstallHook] Hook installed on {type(core).__name__} (id={id(core):#x})")

    def _patch_config_widget_load(self):
        """Intercept the ConfigurationWidget Load button by patching _load_cfg.

        Called after every set_core / _rebuild_toolbars to keep the patch current.

        This is the primary interception point for RemoteMMCore: instance-attribute
        assignments on the core are ignored (see _install_load_hook), but
        ConfigurationWidget is a plain Python class so method replacement works.

        Decision logic (mirrors _install_load_hook):
          • Same core type as running proxy → call original _load_cfg (RPC handles
            unload+load synchronously; WebSocket delivers systemConfigurationLoaded).
          • Core type change or first load → _start_proxy_worker for a fresh proxy.
        """
        try:
            from pymmcore_widgets import ConfigurationWidget
        except ImportError:
            logger.warning("[LoadHook] pymmcore_widgets.ConfigurationWidget not importable")
            return

        widgets = self._mmwin.findChildren(ConfigurationWidget)
        if not widgets:
            logger.warning("[LoadHook] No ConfigurationWidget found in napari-micromanager")
            return

        for widget in widgets:
            # Capture the original bound method BEFORE any replacement.
            # Qt signal connections store a snapshot of the bound method at connect()
            # time, not a live attribute lookup — so replacing widget._load_cfg
            # would never be called by the button.  Instead we disconnect the button
            # from the original handler and reconnect it to our interceptor.
            orig_load_cfg = widget._load_cfg

            def _intercepted(*, _w=widget, _orig=orig_load_cfg):
                path = _w.cfg_LineEdit.text().strip()
                logger.info(f"[LoadHook] Load button clicked — path={path!r}")

                if not path:
                    logger.debug("[LoadHook] Empty path — passing through to original")
                    _orig()
                    return

                if self._remote_connected and self._is_truly_remote(self._remote_url):
                    QMessageBox.warning(
                        self,
                        "Remote core is active",
                        f"You are connected to a remote core at <b>{self._remote_url}</b>.<br><br>"
                        "Loading a local .cfg file would disconnect you from the remote "
                        "and start a new local proxy.<br><br>"
                        "Disconnect the remote core first if you want to load a local configuration.",
                    )
                    logger.warning(
                        "[LoadHook] Blocked — truly-remote core active (%s)", self._remote_url
                    )
                    return

                new_type = _classify_cfg(path)
                logger.info(f"[LoadHook] cfg type={new_type!r}")
                if new_type == "mixed":
                    logger.error("[LoadHook] Mixed C++/Python cfg not supported")
                    self._set_status("Error: mixed C++/Python (#py) cfg not supported yet")
                    return

                current_type = (
                    _classify_cfg(self._last_cfg_path)
                    if self._proxy_worker is not None and self._last_cfg_path
                    else None
                )
                logger.info(
                    f"[LoadHook] current_type={current_type!r}  new_type={new_type!r}  "
                    f"proxy_running={self._proxy_worker is not None}"
                )
                self._last_cfg_path = path

                if current_type is not None and current_type == new_type:
                    logger.info(f"[LoadHook] Same type ({new_type}) — passing to proxy RPC")
                    _orig()
                    return

                logger.info(
                    f"[LoadHook] Type change ({current_type!r} → {new_type!r}) "
                    f"— starting new proxy"
                )
                self._start_proxy_worker(path)

            try:
                widget.load_cfg_Button.clicked.disconnect(orig_load_cfg)
            except Exception as e:
                logger.warning(f"[LoadHook] Could not disconnect original handler: {e}")
            widget.load_cfg_Button.clicked.connect(_intercepted)
            logger.info(
                f"[LoadHook] Reconnected load_cfg_Button on "
                f"ConfigurationWidget (id={id(widget):#x})"
            )

    def _start_proxy_worker(self, cfg_path: str):
        # Start a fresh proxy with the core class required by cfg_path.
        # Called only when the core type is changing (virtual↔real) or on first load.
        if self._proxy_worker is not None:
            self._core_dot.setStyleSheet(_DOT_BUSY)
            self._clear_core_badge("Switching core…")
            self._set_status("Switching core — stopping previous proxy…")
            self._proxy_worker.stop()  # fast: force_exit=True
            if self._proxy_thread is not None:
                self._proxy_thread.quit()
                self._proxy_thread.wait(2000)
            self._proxy_worker = None
            self._proxy_thread = None

        port_text = self._proxy_port_edit.text().strip()
        proxy_port = (
            int(port_text) if port_text.isdigit() and 1 <= int(port_text) <= 65535 else 5601
        )
        self._settings.setValue("proxy_port", str(proxy_port))

        # Wait for the OS to release the port before binding again.
        # On Windows TIME_WAIT can hold a port for a few hundred milliseconds
        # after the socket is closed; polling here prevents WinError 10048.
        if _port_in_use(proxy_port):
            import time as _time

            deadline = _time.monotonic() + 5.0
            self._set_status(f"Waiting for port {proxy_port} to be released…")
            while _time.monotonic() < deadline:
                _time.sleep(0.2)
                if not _port_in_use(proxy_port):
                    break
            else:
                self._core_dot.setStyleSheet(_DOT_ERROR)
                self._clear_core_badge(f"Port {proxy_port} still in use")
                self._set_status(
                    f"Port {proxy_port} is still bound after 5 s — try a different port"
                )
                _log_port_listeners(proxy_port)
                return

        logger.info(f"[ProxyWorker] Launching CoreProxyWorker: cfg={cfg_path!r} port={proxy_port}")
        self._startup_timer = StartupTimer()
        self._startup_timer.mark("proxy_worker_launch")
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
        self._startup_timer.mark("proxy_worker_thread_started")

    @pyqtSlot(str, str)
    def _on_proxy_ready(self, url: str, cfg_path: str):
        from pymmcore_proxy import connect

        logger.info(f"[ProxyReady] Server up at {url}")
        self._startup_timer.mark("health_passed_signal_received")
        self._set_status("Proxy ready — connecting core…")

        # Close the old RemoteMMCore before handing the new core to napari-micromanager.
        # Without this, napari-micromanager's cleanup calls isSequenceRunning() on the
        # dead old proxy, which times out and can raise, corrupting the core swap.
        if self._mmc is not None:
            logger.info("[ProxyReady] Closing old remote core")
            try:
                self._mmc.close()
            except Exception as e:
                logger.debug("[ProxyReady] Old core close raised (expected): %s", e)
            self._mmc = None

        logger.info(f"[ProxyReady] Calling connect({url})")
        remote_core = connect(url)
        self._startup_timer.mark("connect_done")
        logger.info(f"[ProxyReady] Connected — remote_core type: {type(remote_core).__name__}")

        self._in_user_load = True
        try:
            if self._mmwin is not None:
                logger.info("[ProxyReady] Calling set_core on napari-micromanager")
                try:
                    self._mmwin.set_core(remote_core)
                    logger.info("[ProxyReady] set_core complete")
                except Exception as e:
                    # Old-core cleanup inside set_core can raise if the proxy was killed;
                    # the new core is already set at this point so we log and continue.
                    logger.warning("[ProxyReady] set_core raised (old proxy cleanup): %s", e)
        finally:
            self._in_user_load = False
        self._mmc = remote_core
        self._startup_timer.mark("set_core_and_rebuild_done")

        # Load cfg through the raw RPC layer — bypasses _capturing_load and
        # napari's _auto_detect_load, which both sit on top of the RPC method.
        # _rpc() is a plain Python method on RemoteMMCore (not in _RPC_FORWARD_METHODS)
        # so __getattribute__ doesn't intercept it.
        logger.info(f"[ProxyReady] Loading cfg via RPC: {cfg_path!r}")
        try:
            remote_core._rpc("loadSystemConfiguration", cfg_path)
            logger.info("[ProxyReady] RPC loadSystemConfiguration complete")
        except Exception as e:
            logger.warning("[ProxyReady] RPC loadSystemConfiguration failed: %s", e)
        self._startup_timer.mark("rpc_load_system_cfg_done")

        # Fire systemConfigurationLoaded directly on the client signal bus.
        # The WebSocket listener starts async in a background thread and may not
        # be connected yet, so the server's WebSocket broadcast would be lost.
        # Emitting here reaches all already-connected widgets (ChannelWidget etc.)
        # synchronously, giving them a fresh RPC call to the now-configured proxy.
        logger.info("[ProxyReady] Emitting systemConfigurationLoaded on client")
        try:
            remote_core.events.systemConfigurationLoaded.emit()
            logger.info("[ProxyReady] systemConfigurationLoaded emitted")
        except Exception as e:
            logger.warning("[ProxyReady] systemConfigurationLoaded emit failed: %s", e)
        self._startup_timer.mark("system_config_loaded_emitted")
        self._startup_timer.report()

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
        self._startup_timer.mark("auto_load_config_triggered")
        self._set_status(f"Auto-loading {os.path.basename(self._auto_config)}…")
        self._mmwin.core.loadSystemConfiguration(self._auto_config)

    # ── Core badge helpers ──────────────────────────────────────────────────

    _BADGE_STYLES = {
        "CMMCorePlus": (
            "CMM+",
            "background:#1565C0;color:white;border-radius:7px;padding:1px 5px;font-size:10px;font-weight:bold;",
        ),
        "UniMMCore": (
            "Uni",
            "background:#6A1B9A;color:white;border-radius:7px;padding:1px 5px;font-size:10px;font-weight:bold;",
        ),
    }
    _BADGE_FALLBACK = (
        "RMC",
        "background:#E65100;color:white;border-radius:7px;padding:1px 5px;font-size:10px;font-weight:bold;",
    )

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
        msg = message.lower()
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
