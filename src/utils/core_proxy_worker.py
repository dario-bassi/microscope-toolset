"""CoreProxyWorker — starts a pymmcore-proxy server for any cfg file type.

Handles both virtual (Python #py devices → UniMMCore) and real
(C++ Device drivers → CMMCorePlus) configurations.  All heavy imports
are deferred to run() so the Qt main thread is never blocked.
"""
import logging
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.utils.cfg_utils import classify_cfg

logger = logging.getLogger("CoreProxyWorker")


class CoreProxyWorker(QObject):
    """Starts a pymmcore-proxy HTTP server for a given .cfg file.

    Classifies the cfg file to choose the right core type:
      - 'virtual'  →  UniMMCore  (Python #py devices, simulation)
      - 'real'     →  CMMCorePlus (C++ Device drivers, real hardware)
      - 'mixed'    →  not supported; emits server_error immediately

    Loads the cfg into a fresh core instance, starts pymmcore-proxy's
    uvicorn in a daemon thread, then polls GET /health until the server
    accepts connections.

    Signals:
        server_ready(url)  — 'http://host:port' once the server is up
        server_error(msg)  — human-readable error if startup fails
    """

    server_ready = pyqtSignal(str)
    server_error = pyqtSignal(str)

    _POLL_INTERVAL = 0.2   # seconds between health-check attempts
    _MAX_WAIT      = 15.0  # seconds before giving up

    def __init__(self, cfg_path: str, host: str = "127.0.0.1", port: int = 5601):
        super().__init__()
        self._cfg_path = cfg_path
        self._host = host
        self._port = port

    @pyqtSlot()
    def run(self):
        try:
            import urllib.request
            import logging.handlers as _lh
            from pymmcore_proxy import serve

            # On Windows, RotatingFileHandler.doRollover() calls os.rename() which
            # fails if another process (e.g. the ipykernel MCP sandbox) has the same
            # pymmcore-plus log file open.  Disabling rotation here avoids the rename.
            _pmm = logging.getLogger("pymmcore-plus")
            for _h in _pmm.handlers:
                if isinstance(_h, _lh.RotatingFileHandler):
                    _h.maxBytes = 0

            cfg_type = classify_cfg(self._cfg_path)
            logger.info(f"cfg type={cfg_type!r}  path={self._cfg_path!r}")

            if cfg_type == "mixed":
                self.server_error.emit(
                    "Mixed C++/Python (#py) cfg files are not supported yet. "
                    "Use a cfg with either all C++ Device lines or all #py pyDevice lines."
                )
                return

            # Force psygnal signals for the server core.
            # CMMCorePlus auto-selects Qt signals when a QApplication is running,
            # but Qt signals connected from non-Qt threads (like uvicorn's asyncio
            # thread pool) silently fail to deliver.  psygnal has no such restriction.
            import os
            _old_backend = os.environ.get("PYMM_SIGNALS_BACKEND")
            os.environ["PYMM_SIGNALS_BACKEND"] = "psygnal"
            try:
                if cfg_type == "virtual":
                    from pymmcore_plus.experimental.unicore import UniMMCore
                    core = UniMMCore()
                else:
                    from pymmcore_plus import CMMCorePlus
                    core = CMMCorePlus()
            finally:
                if _old_backend is None:
                    os.environ.pop("PYMM_SIGNALS_BACKEND", None)
                else:
                    os.environ["PYMM_SIGNALS_BACKEND"] = _old_backend

            logger.info(f"Loading cfg with {type(core).__name__}")
            core.loadSystemConfiguration(self._cfg_path)
            logger.info(f"cfg loaded — starting proxy on {self._host}:{self._port}")

            server_thread = threading.Thread(
                target=serve,
                kwargs={"core": core, "host": self._host, "port": self._port},
                daemon=True,
                name=f"CoreProxy-{self._port}",
            )
            server_thread.start()

            url        = f"http://{self._host}:{self._port}"
            health_url = f"{url}/health"
            deadline   = time.monotonic() + self._MAX_WAIT

            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(health_url, timeout=0.5) as resp:
                        if resp.status == 200:
                            logger.info(f"Server ready at {url}")
                            self.server_ready.emit(url)
                            return
                except Exception:
                    pass
                time.sleep(self._POLL_INTERVAL)

            self.server_error.emit(
                f"Proxy server did not become ready within "
                f"{self._MAX_WAIT:.0f}s (port {self._port})"
            )

        except Exception as e:
            logger.exception(f"CoreProxyWorker error: {e}")
            self.server_error.emit(str(e))
