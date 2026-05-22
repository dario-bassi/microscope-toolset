import argparse
import logging
import os
import sys
import threading
import time

logger = logging.getLogger("NapariMicroscopeTool")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)


class _UvicornLifespanCancelFilter(logging.Filter):
    """Drop uvicorn's benign 'lifespan was cancelled' traceback.

    When we stop a uvicorn server with ``force_exit=True`` (needed to release
    the port immediately for a cfg switch or an MCP reconnect — see
    CoreProxyWorker.stop / MCPServerWorker._stop_uvicorn), uvicorn skips the
    graceful ``lifespan.shutdown`` and the lifespan task is cancelled as the
    loop tears down. uvicorn logs that ``asyncio.CancelledError`` on the
    ``uvicorn.error`` logger as a pre-formatted traceback *message* (exc_info
    is None), so it can only be matched by message content. It is cosmetic —
    the server is already stopping. A real lifespan *startup* failure raises a
    different exception (not CancelledError) and is left untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not ("CancelledError" in msg and "lifespan" in msg)


def _install_uvicorn_log_filter() -> None:
    """Attach the lifespan-cancel filter to the global ``uvicorn.error`` logger.

    Installed once at startup. It survives uvicorn's own ``logging.dictConfig``
    (which runs per-server with ``disable_existing_loggers=False`` and does not
    clear pre-existing filters), and the logger is process-global so this covers
    both the proxy (5601) and MCP (5500) servers.
    """
    uvicorn_error = logging.getLogger("uvicorn.error")
    if not any(isinstance(f, _UvicornLifespanCancelFilter) for f in uvicorn_error.filters):
        uvicorn_error.addFilter(_UvicornLifespanCancelFilter())


def main():
    _install_uvicorn_log_filter()
    parser = argparse.ArgumentParser(description="Microscope Toolset - napari + MCP server")
    parser.add_argument(
        "--review",
        type=str,
        default=False,
        help="Path to the .jsonl file to review the full conversation of the experiment with the Agent.",
    )
    parser.add_argument(
        "--log", type=str, default=None, help="Path to pymmcore-plus.log. Auto-detected if omitted."
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Test name to run (e.g. test_1). Omit the name to list available tests.",
    )
    args, _unknown = parser.parse_known_args()

    if args.review:
        from benchmarking import (
            default_log_path,
            launch_dashboard,
            merge_logs,
            parse_log_file,
            read_file,
        )

        try:
            messages, stats = read_file(args.review)
            logger.info(
                f"Loaded {len(messages)} messages | "
                f"turns: user={stats.num_user_turns} agent={stats.num_assistant_turns} | "
                f"tool calls: {stats.num_tool_calls} | cost: ${stats.estimated_cost_usd:.4f}"
            )

            log_path = args.log or default_log_path()
            if log_path:
                try:
                    log_entries = parse_log_file(log_path)
                    messages = merge_logs(messages, log_entries)
                    n_blocks = sum(
                        1 for m in messages if hasattr(m, "entries") and hasattr(m, "timestamp")
                    )
                    logger.info(f"Merged {n_blocks} hardware log block(s) from {log_path}")
                except Exception as e:
                    logger.warning(f"Could not load hardware log: {e}")
            else:
                logger.info("No pymmcore-plus.log found — skipping hardware log merge")

            launch_dashboard(messages, stats)
        except FileNotFoundError as e:
            logger.error(str(e))
        except Exception as e:
            logger.info(e)
    else:
        if args.test is not None and args.test == "":
            from benchmarking import print_tests

            print_tests()
            sys.exit(0)
        elif args.test:
            from benchmarking import run_test

            logger.info(f"Setting up test: {args.test}")
            auto_config = str(run_test(args.test))
            logger.info(f"Test cfg: {auto_config}")
        else:
            auto_config = None

        try:
            import napari

            from mcp_server_gui import MCPServer

            logger.info("Start napari window")
            viewer = napari.Viewer()
            logger.info(viewer.window)
            logger.info("start mcp server widget")

            main_window = MCPServer(auto_config=auto_config)
            viewer.window.add_dock_widget(
                widget=main_window, name="MCP Server", area="top", allowed_areas=["right"]
            )

            try:
                napari.run()
            finally:
                # Closing napari does not stop the MCP/proxy servers on its own;
                # tear them down here so they don't linger holding their ports
                # (an open MCP/SSE client otherwise wedges the asyncio shutdown).
                logger.info("Napari finished — shutting down servers")
                _shutdown_and_exit(main_window)
        except Exception as e:
            logger.info(f"Error starting napari: {e}")


def _force_kill() -> None:
    """Terminate this process *now*, skipping the orderly teardown that hangs.

    ``os._exit`` / ``ExitProcess`` run every loaded DLL's
    ``DLL_PROCESS_DETACH`` handler — and a real microscope driver DLL
    (Nikon Ti / Photometrics / Mosaic3) deadlocks there, so the in-process
    exit never completes. (Empirically only an *external* ``TerminateProcess``,
    e.g. Task Manager's End Task, could kill it.) ``TerminateProcess`` on our
    own process handle is that same forceful kill from the inside: the kernel
    tears the process down without running DLL detach. Falls back to
    ``os._exit`` on non-Windows or if the call unexpectedly returns.
    """
    logging.shutdown()
    if sys.platform == "win32":
        try:
            import ctypes

            # NB: declare the signature explicitly. Letting ctypes default the
            # return of GetCurrentProcess() to a 32-bit int truncates the -1
            # pseudo-handle to 0xFFFFFFFF (an invalid handle) and the call
            # silently no-ops. Pass the full-width -1 pseudo-handle directly.
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.TerminateProcess.restype = ctypes.c_int
            k32.TerminateProcess(ctypes.c_void_p(-1), 0)
        except Exception:
            pass
    os._exit(0)


def _shutdown_and_exit(main_window) -> None:
    """Stop the servers + release hardware, then let the process exit on its own.

    ``MCPServer.shutdown()`` unloads the devices gracefully (on the still-live
    proxy) and stops the servers, which lets the interpreter finalize normally.
    We deliberately do NOT force-kill on the happy path: a clean exit should
    happen by itself, and not masking it means a residual device/handle leak
    surfaces as a delayed exit instead of being silently papered over.

    The daemon watchdog is the safety net. If the process is still alive after
    the timeout, the normal exit is wedged — something did not release cleanly —
    so it logs that loudly (the leak signal you can grep for) and then
    force-terminates via ``_force_kill`` so you are never left with a stuck
    process. Net effect: clean exit when possible, a logged signal + forced
    exit when not.
    """
    _WATCHDOG_S = 20.0

    def _watchdog() -> None:
        time.sleep(_WATCHDOG_S)
        logger.warning(
            "Process still alive ~%.0fs after shutdown — the normal exit is "
            "WEDGED, so a device handle or driver thread did not release "
            "cleanly (see the last teardown lines above for how far it got). "
            "Force-terminating now. If this recurs, the graceful unload is not "
            "fully releasing the hardware.",
            _WATCHDOG_S,
        )
        _force_kill()

    threading.Thread(target=_watchdog, daemon=True, name="exit-watchdog").start()
    try:
        main_window.shutdown()
    except Exception:
        logger.exception("shutdown() raised during exit")
    logger.info(
        "Shutdown complete — exiting normally (no 'WEDGED' warning after this "
        "line means the process exited cleanly without a forced kill)"
    )
    # Return and let the interpreter finalize on its own. The watchdog above is
    # the only thing that force-kills, and only if that normal exit wedges.


if __name__ == "__main__":
    main()
