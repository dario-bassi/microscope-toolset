"""Standalone benchmark test server.

Wraps a benchmark test (from src/benchmarking/test_N/) as a pymmcore-proxy
server running in its own process.  The agent connects to it exactly like any
other remote core; the test config and ground truth are never exposed.

Extra endpoint (beyond the standard ProxyServer routes):
    GET /test/info — title, description, channel names

Usage
-----
CLI:
    python -m src.benchmarking.test_server test_1
    python -m src.benchmarking.test_server test_1 --host 0.0.0.0 --port 5601
    python -m src.benchmarking.test_server --list

From Python (e.g. launched as a subprocess by the GUI):
    from .test_server import serve_test
    serve_test("test_1", host="127.0.0.1", port=5601)
"""

from __future__ import annotations

import argparse
import sys

from pymmcore_proxy import ProxyServer
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# TestProxyServer
# ---------------------------------------------------------------------------


class TestProxyServer(ProxyServer):
    """ProxyServer with an additional GET /test/info endpoint."""

    def __init__(self, core, test_info: dict, host: str = "127.0.0.1", port: int = 5601):
        self._test_info = test_info
        extra = [Route("/test/info", self._handle_test_info, methods=["GET"])]
        super().__init__(core, host=host, port=port, extra_routes=extra)

    async def _handle_test_info(self, request: Request) -> JSONResponse:
        return JSONResponse(self._test_info)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_public_info(test_name: str, module) -> dict:
    """Return only the information the agent is allowed to know.

    Exposes: title, description (module docstring), channel names.
    Does NOT expose: n_cells, cell_type, seed, ground truth, focal_plane.
    """
    cfg: dict = getattr(module, "TEST_CONFIG", {})
    channel_names = [ch["name"] for ch in cfg.get("channels", [])]
    description = (module.__doc__ or "").strip()
    return {
        "test_name": test_name,
        "title": cfg.get("title", test_name),
        "description": description,
        "channels": channel_names,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serve_test(test_name: str, host: str = "127.0.0.1", port: int = 5601) -> None:
    """Set up a benchmark test and run the proxy server (blocking).

    Steps:
    1. Load the test module to extract public info.
    2. Call run_test() — creates the simulation, sets GLOBAL_BRIDGE, generates cfg.
    3. Load the generated cfg into the right core type (UniMMCore for #py virtual
       configs, CMMCorePlus for real hardware configs).
    4. Start TestProxyServer (blocking until Ctrl-C or process kill).
    """
    import os

    from .test_runner import _load_test_module, run_test

    print(f"[test_server] Loading test '{test_name}' ...")
    module = _load_test_module(test_name)
    test_info = _extract_public_info(test_name, module)

    cfg_path = run_test(test_name)

    from utils.cfg_utils import classify_cfg

    cfg_type = classify_cfg(str(cfg_path))

    # Force psygnal signals — Qt signals silently fail in uvicorn's asyncio thread.
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

    core.loadSystemConfiguration(str(cfg_path))

    server = TestProxyServer(core, test_info=test_info, host=host, port=port)

    print(f"[test_server] Ready — serving '{test_name}' on http://{host}:{port}")
    print(f"[test_server] Channels : {test_info['channels']}")
    print("[test_server] GET /test/info for task description")
    print("[test_server] Press Ctrl-C to stop.")
    server.run()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a pymmcore-proxy test server for a benchmark test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.benchmarking.test_server test_1\n"
            "  python -m src.benchmarking.test_server test_1 --port 5601\n"
            "  python -m src.benchmarking.test_server --list\n"
        ),
    )
    parser.add_argument(
        "test_name",
        nargs="?",
        help="Test folder name, e.g. test_1",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5601,
        help="Port to listen on (default: 5601)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit",
    )
    args = parser.parse_args()

    if args.list:
        from .test_runner import print_tests

        print_tests()
        sys.exit(0)

    if not args.test_name:
        parser.print_help()
        sys.exit(1)

    try:
        serve_test(args.test_name, host=args.host, port=args.port)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[test_server] Stopped.")


if __name__ == "__main__":
    _main()
