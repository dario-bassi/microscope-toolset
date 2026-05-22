"""CoreProxyWorker — starts a pymmcore-proxy server for any cfg file type.

Handles both virtual (Python #py devices → UniMMCore) and real
(C++ Device drivers → CMMCorePlus) configurations.  All heavy imports
are deferred to run() so the Qt main thread is never blocked.
"""

import logging
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .cfg_utils import classify_cfg

logger = logging.getLogger("CoreProxyWorker")


def install_single_read_getimage_shim(core):
    """Make ``getImage()`` re-readable for single-read camera adapters.

    Some camera device adapters (notably certain PVCAM builds) hand back the
    snapped frame from ``GetImageBuffer()`` only ONCE per ``snapImage()`` — a
    second ``getImage()`` raises *"Camera image buffer read failed."* This breaks
    the napari snap path, which reads the buffer twice per snap: ``mmc.snap()``
    calls ``snapImage()`` then ``getImage()`` (read #2), while ``snapImage()``
    emits ``imageSnapped`` whose napari handler (`_core_link._image_snapped`)
    calls ``getImage()`` to display (read #1). MMStudio and single-read clients
    work because they read once; see image.sc forum topic 107892.

    This caches the frame on the first ``getImage()`` after each ``snapImage()``
    and serves later reads of the *same* snap from the cache, so the underlying
    (failing) second hardware read never happens.

    Safe for BOTH camera types:
      * Re-readable cameras — the cache just mirrors the real frame; behaviour is
        identical (repeated reads of one snap return the same pixels anyway).
      * Single-read cameras — the failing 2nd read is avoided.
    The cache is invalidated on every ``snapImage()`` and only served if it was
    populated since the last snap, so a stale frame can never be returned and a
    genuine first-read failure still raises (not masked). Live/sequence
    acquisition uses ``getLastImage``/``popNextImage`` (untouched).
    """
    orig_snap = core.snapImage
    orig_get = core.getImage
    lock = threading.Lock()
    cache: dict = {}  # (numChannel, fix) -> frame; cleared on each snapImage

    def _snap_image(*args, **kwargs):
        with lock:
            cache.clear()
        return orig_snap(*args, **kwargs)

    def _get_image(numChannel=None, *, fix=True):
        key = (numChannel, fix)
        with lock:
            if key in cache:
                return cache[key]  # read #2+ of this snap
            img = orig_get(numChannel, fix=fix)  # read #1 (propagates on failure)
            cache[key] = img
            return img

    core.snapImage = _snap_image
    core.getImage = _get_image
    logger.info(
        "[getImage-shim] installed single-read-camera workaround on %s "
        "(getImage cached per snapImage; see forum 107892)",
        type(core).__name__,
    )


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

    server_ready = pyqtSignal(str, str)  # (url, cfg_path)
    server_error = pyqtSignal(str)

    _POLL_INTERVAL = 0.2  # seconds between health-check attempts
    _MAX_WAIT = 15.0  # seconds before giving up

    def __init__(self, cfg_path: str, host: str = "127.0.0.1", port: int = 5601):
        super().__init__()
        self._cfg_path = cfg_path
        self._host = host
        self._port = port
        self._uvicorn_server = None  # set inside _server_thread before serving
        self._server_thread = None

    @pyqtSlot()
    def run(self):
        try:
            import logging.handlers as _lh
            import urllib.request

            import uvicorn
            from pymmcore_proxy import ProxyServer

            _t0 = time.perf_counter()

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
                    # Real hardware only: guard against single-read camera adapters
                    # (e.g. PVCAM) whose getImage() fails on the napari snap path's
                    # second read. No-op for re-readable cameras.
                    install_single_read_getimage_shim(core)
            finally:
                if _old_backend is None:
                    os.environ.pop("PYMM_SIGNALS_BACKEND", None)
                else:
                    os.environ["PYMM_SIGNALS_BACKEND"] = _old_backend

            logger.info("[TIMING] worker:core_created          +%.3fs", time.perf_counter() - _t0)
            logger.info(
                f"Starting proxy with empty {type(core).__name__} on {self._host}:{self._port}"
            )

            proxy = ProxyServer(core, port=self._port)
            config = uvicorn.Config(
                proxy.app,
                host=self._host,
                port=self._port,
                log_level="warning",
            )
            self._uvicorn_server = uvicorn.Server(config)
            logger.info("[TIMING] worker:proxy_configured       +%.3fs", time.perf_counter() - _t0)

            # Run uvicorn in a daemon thread so stop() can join it from any thread.
            self._server_thread = threading.Thread(
                target=self._uvicorn_server.run,
                daemon=True,
                name=f"CoreProxy-{self._port}",
            )
            self._server_thread.start()
            logger.info("[TIMING] worker:uvicorn_thread_started +%.3fs", time.perf_counter() - _t0)

            url = f"http://{self._host}:{self._port}"
            health_url = f"{url}/health"
            deadline = time.monotonic() + self._MAX_WAIT

            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(health_url, timeout=0.5) as resp:  # nosec B310
                        if resp.status == 200:
                            logger.info(
                                "[TIMING] worker:health_200             +%.3fs",
                                time.perf_counter() - _t0,
                            )
                            logger.info(f"Server ready at {url}")
                            self.server_ready.emit(url, self._cfg_path)
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

    def stop(self):
        """Stop the proxy server. Safe to call from any thread.

        Sets force_exit=True so the listen socket is released immediately
        without waiting for in-flight connections to drain.
        """
        server = self._uvicorn_server
        if server is not None:
            server.should_exit = True
            server.force_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
            self._server_thread = None
        self._uvicorn_server = None
        logger.info("CoreProxy stopped on port %d", self._port)
