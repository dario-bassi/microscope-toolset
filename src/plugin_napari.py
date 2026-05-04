import argparse
import logging
import sys

#  logger
logger = logging.getLogger("NapariMicroscopeTool")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microscope Toolset - napari + MCP server")
    parser.add_argument(
        "--review", type=str, default=False,
        help="Path to the .jsonl file to review the full conversation of the experiment with the Agent."
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to pymmcore-plus.log. Auto-detected if omitted."
    )
    parser.add_argument(
        "--test", type=str, nargs="?", const="", default=None,
        help="Test name to run (e.g. test_1). Omit the name to list available tests."
    )
    args, _unknown = parser.parse_known_args()

    if args.review:
        from src.benchmarking.review_conversation import (
            read_file, parse_log_file, merge_logs, default_log_path,
        )
        from src.benchmarking.dashboard import launch_dashboard
        try:
            messages, stats = read_file(args.review)
            logger.info(
                f"Loaded {len(messages)} messages | "
                f"turns: user={stats.num_user_turns} agent={stats.num_assistant_turns} | "
                f"tool calls: {stats.num_tool_calls} | cost: ${stats.estimated_cost_usd:.4f}"
            )

            # Merge hardware logs
            log_path = args.log or default_log_path()
            if log_path:
                try:
                    log_entries = parse_log_file(log_path)
                    messages = merge_logs(messages, log_entries)
                    n_blocks = sum(1 for m in messages
                                   if hasattr(m, "entries") and hasattr(m, "timestamp"))
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
        #return
    else:
        # Resolve test / auto_config before importing napari
        if args.test is not None and args.test == "":
            from src.benchmarking.test_runner import print_tests
            print_tests()
            sys.exit(0)
        elif args.test:
            from src.benchmarking.test_runner import run_test
            logger.info(f"Setting up test: {args.test}")
            auto_config = str(run_test(args.test))
            logger.info(f"Test cfg: {auto_config}")
        else:
            auto_config = None

        try:
            import napari
            from src.mcp_server_gui import MCPServer

            logger.info("Start napari window")
            viewer = napari.Viewer()
            logger.info(viewer.window)
            logger.info("start mcp server widget")

            main_window = MCPServer(auto_config=auto_config)
            viewer.window.add_dock_widget(widget=main_window, name="MCP Server", area="top", allowed_areas=["right"])

            napari.run()

            logger.info("Napari finished")
        except Exception as e:
            logger.info(f"Error starting napari: {e}")
