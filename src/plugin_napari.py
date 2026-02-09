import argparse
import napari
from src.mcp_server_gui import MCPServer
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
    parser = argparse.ArgumentParser(description="Microscope Toolset – napari + MCP server")
    parser.add_argument(
        "--auto-start", action="store_true", default=False,
        help="Automatically start the MCP server and load the config file (no manual clicks needed)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/virtual_optogenetic.cfg",
        help="Path to the .cfg file to load (default: configs/virtual_optogenetic.cfg). Only used with --auto-start."
    )
    args, _unknown = parser.parse_known_args()

    try:
        logger.info("Start napari window")
        viewer = napari.Viewer()
        logger.info(viewer.window)
        logger.info("start mcp server widget")

        auto_config = args.config if args.auto_start else None
        main_window = MCPServer(auto_config=auto_config)
        viewer.window.add_dock_widget(widget=main_window, name="MCP Server", area="top", allowed_areas=["right"])

        napari.run()

        logger.info("Napari finished")
    except Exception as e:
        logger.info(f"Error starting napari: {e}")
