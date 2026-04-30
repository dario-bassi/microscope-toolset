from typing import Any, Annotated, Literal
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import ImageContent, TextContent
from pydantic import Field, BeforeValidator, PlainSerializer, WithJsonSchema
from src.local.prepare_code import prepare_code
import logging
import sys
import json
import datetime
import numpy as np
import asyncio
import base64
import time
import os
from io import BytesIO
from pydantic import BaseModel
from src.local.gatekeeper_core import GatekeeperCore
from dotenv import load_dotenv
from src.benchmarking.benchmark_logger import BenchmarkLogger

#  logger
logger = logging.getLogger("ServerSetup")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

# Global benchmark logger (set from outside when benchmarking)
benchmark_logger: BenchmarkLogger | None = None

# This ensures the LLM sees a standard list of numbers
NDArray = Annotated[
    np.ndarray,
    BeforeValidator(lambda v: np.array(v)),
    PlainSerializer(lambda v: v.tolist()),
    WithJsonSchema({
        "type": "array",
        "description": "Numerical array of any shape (1D, 2D, 3D, etc.) as nested lists."
    })
]


def create_mcp_server(
        database_agent,
        microscope_status,
        executor,
        viewer,
        event_cache,
        viewer_proxy=None,
        benchmark_logger_instance: BenchmarkLogger | None = None
) -> FastMCP:
    # Server definition
    mcp = FastMCP(
        name="Microscope Toolset",
        host="127.0.0.1",
        port=5500,
        streamable_http_path="/mcp",
        log_level="INFO"
    )

    # If a raw mmc instance is provided inside microscope_status, wrap it so tools run safely
    try:
        raw_mmc = getattr(microscope_status, "mmc", None)
        if raw_mmc is not None and not isinstance(raw_mmc, GatekeeperCore):
            microscope_status.mmc = GatekeeperCore(raw_mmc)
    except Exception:
        pass

    # Set up benchmark logger if provided
    global benchmark_logger
    benchmark_logger = benchmark_logger_instance

    # Load .env for configuration
    load_dotenv()

    @mcp.tool(
        name="pymmcore_api_database",
        description="Search the pymmcore-plus API documentation database using hybrid BM25 + KNN retrieval with cross-encoder re-ranking. Returns the top 25 most relevant chunks."
    )
    def pymmcore_api_database(
            user_query: str = Field(..., description="The user original question")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            if database_agent is None:
                result = {"user_query": user_query, "error": "Database agent not available (Elasticsearch not configured)"}
                return result

            # reformulate user query
            reformulated_question = database_agent.rephrase_query(user_query)

            # check if rephrase failed
            if isinstance(reformulated_question, dict) and reformulated_question.get('intent') == 'error':
                result = {
                    "user_query": user_query,
                    "error": reformulated_question.get('message', 'Failed to reformulate query')
                }
                return result

            # extract reformulated_query string from dict
            reformulated_query_str = reformulated_question.get("reformulated_query", user_query) if isinstance(reformulated_question, dict) else reformulated_question

            result = database_agent.api_pymmcore_context(user_query, reformulated_query_str)
            return result
        except Exception as e:
            logger.error(f"Error in pymmcore_api_database: {e}", exc_info=True)
            result = {
                "user_query": user_query,
                "error": f"Error retrieving information from databases: {str(e)}"
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="pymmcore_api_database",
                    input_params={"user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )
    @mcp.tool(
        name="micromanager_device_database",
        description="Search the Micro-Manager device documentation database using hybrid BM25 + KNN retrieval with cross-encoder re-ranking. Returns the top 25 most relevant chunks."
    )
    def micromanager_device_database(
            user_query: str = Field(..., description="The user original question")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            if database_agent is None:
                result = {"user_query": user_query, "error": "Database agent not available (Elasticsearch not configured)"}
                return result

            # reformulate user query
            reformulated_question = database_agent.rephrase_query(user_query)

            # check if rephrase failed
            if isinstance(reformulated_question, dict) and reformulated_question.get('intent') == 'error':
                result = {
                    "user_query": user_query,
                    "error": reformulated_question.get('message', 'Failed to reformulate query')
                }
                return result

            # extract reformulated_query string from dict
            reformulated_query_str = reformulated_question.get("reformulated_query", user_query) if isinstance(reformulated_question, dict) else reformulated_question

            result = database_agent.devices_micromanager_context(user_query, reformulated_query_str)
            return result
        except Exception as e:
            logger.error(f"Error in micromanager_device_database: {e}", exc_info=True)
            result = {
                "user_query": user_query,
                "error": f"Error retrieving information from databases: {str(e)}"
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="micromanager_device_database",
                    input_params={"user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )
    @mcp.tool(
        name="pdfs_publication_database",
        description="Search scientific publications using hybrid BM25 + KNN retrieval with cross-encoder re-ranking. Returns the top 25 most relevant chunks."
    )
    def pdfs_publication_database(
            user_query: str = Field(..., description="The user original question")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            if database_agent is None:
                result = {"user_query": user_query, "error": "Database agent not available (Elasticsearch not configured)"}
                return result

            # reformulate user query
            reformulated_question = database_agent.rephrase_query(user_query)

            # check if rephrase failed
            if isinstance(reformulated_question, dict) and reformulated_question.get('intent') == 'error':
                result = {
                    "user_query": user_query,
                    "error": reformulated_question.get('message', 'Failed to reformulate query')
                }
                return result

            reformulated_result = reformulated_question.get("reformulated_query", user_query) if isinstance(reformulated_question, dict) else user_query

            result = database_agent.pdf_publication_context(user_query, reformulated_result)
            return result
        except Exception as e:
            logger.error(f"Error in pdfs_publication_database: {e}", exc_info=True)
            result = {
                "user_query": user_query,
                "error": f"Error retrieving information from databases: {str(e)}"
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="pdfs_publication_database",
                    input_params={"user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
         name="reformulate_user_query",
         description="Rephrase a user question into an optimized search query for database retrieval via BM25 text matching and embedding vectors."
     )
    def reformulate_user_query(
             user_question: str = Field(..., description="The user original question"),
             user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
     ) -> dict[str, Any]:
         start_time = time.time()
         result = None
         try:
             if database_agent is None:
                 result = {"user_query": user_question, "error": "Database agent not available (Elasticsearch not configured)"}
                 return result
             # add check that structured response is getting the correct answer
             result = database_agent.rephrase_query(user_question)
             return result
         except Exception as e:
            logger.error(f"Error in reformulate_user_query: {e}", exc_info=True)
            result = {"user_question": user_question, "error": f"Error reformulating query: {str(e)}"}
            return result
         finally:
             execution_time_ms = (time.time() - start_time) * 1000
             if benchmark_logger and user_query:
                 benchmark_logger.set_query(user_query)
             if benchmark_logger and result is not None:
                 benchmark_logger.log_tool_call(
                     tool_name="reformulate_user_query",
                     input_params={"user_question": user_question, "user_query": user_query},
                     result=result,
                     execution_time_ms=execution_time_ms
                 )

    @mcp.tool(
        name="log_session",
        description=(
            "Save a session entry to the PostgreSQL logger database for future retrieval and learning. "
            "Log the user prompt, agent output, optional feedback, and a category label. "
            "Logged entries are embedded and can be retrieved in future sessions to improve responses. "
            "Returns a status indicating whether the log was saved (requires PostgreSQL to be configured)."
        )
    )
    def log_session(
        prompt: str = Field(..., description="A summary of the whole conversation: the user goal, steps taken, and final outcome."),
        output: str = Field(..., description="The agent's response, generated code, or result summary."),
        feedback: str = Field("", description="Optional feedback or notes about the session outcome (e.g. 'worked', 'failed', 'needed adjustment')."),
        category: str = Field("", description="Optional category label for the session (e.g. 'acquisition', 'analysis', 'troubleshooting', 'question')."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            if database_agent is None:
                result = {"status": "skipped", "reason": "Database agent not available (Elasticsearch not configured)"}
                return result
            data = {"prompt": prompt, "output": output, "feedback": feedback, "category": category}
            saved = database_agent.add_log(data)
            if saved:
                result = {"status": "saved", "prompt": prompt, "category": category}
            else:
                result = {"status": "skipped", "reason": "PostgreSQL logger not configured"}
            return result
        except Exception as e:
            logger.error(f"Error in log_session: {e}", exc_info=True)
            result = {"status": "error", "error": str(e)}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="log_session",
                    input_params={"prompt": prompt[:100], "category": category, "user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
        name="retrieve_session_logs",
        description=(
            "Retrieve past logged sessions from the PostgreSQL logger that are semantically similar to the current query. "
            "Use this at the start of a task to check if a similar experiment or analysis was done before "
            "and what approach worked. Returns the most similar past sessions ranked by embedding distance. "
            "Requires PostgreSQL to be configured; returns an empty list otherwise."
        )
    )
    def retrieve_session_logs(
        query: str = Field(..., description="The current task or question — used to find semantically similar past sessions."),
        k: int = Field(5, description="Number of past sessions to retrieve (default 5)."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            if database_agent is None:
                result = {"status": "skipped", "reason": "Database agent not available (Elasticsearch not configured)", "sessions": []}
                return result
            sessions = database_agent.retrieve_session_logs(query=query, k=k)
            result = {"status": "success", "count": len(sessions), "sessions": sessions}
            return result
        except Exception as e:
            logger.error(f"Error in retrieve_session_logs: {e}", exc_info=True)
            result = {"status": "error", "error": str(e), "sessions": []}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="retrieve_session_logs",
                    input_params={"query": query, "k": k, "user_query": user_query},
                    result={"status": result.get("status"), "count": result.get("count")},
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
        name="get_microscope_settings",
        description="Return the current microscope state: device property schemas, current values, and configuration groups."
    )
    def get_microscope_settings(
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            # Get Properties of the microscope
            logger.info("Getting microscope properties...")
            microscope_properties_response = microscope_status.get_properties()
            logger.info(f"Properties retrieved: {type(microscope_properties_response)}")

            # Get current settings
            logger.info("Getting microscope current status...")
            microscope_status_response = microscope_status.get_current_status()
            logger.info(f"Status retrieved: {type(microscope_status_response)}")

            # Get configuration settings
            logger.info("Getting microscope available configs...")
            config_settings = microscope_status.get_available_configs()
            logger.info(f"Configs retrieved: {type(config_settings)}")

            result = {
                "properties_schema": microscope_properties_response,
                "current_properties_status": microscope_status_response,
                "configuration_groups_settings": config_settings
            }
            logger.info({
                "tool": "get_microscope_settings",
                "properties_schema": microscope_properties_response,
                "current_properties_status": microscope_status_response,
                "configuration_groups_settings": config_settings
            })
            return result
        except Exception as e:
            logger.error(f"Error in get_microscope_settings: {e}", exc_info=True)
            result = {
                "error": f"Failed to get microscope settings: {str(e)}",
                "properties_schema": {},
                "current_properties_status": {},
                "configuration_groups_settings": {}
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="get_microscope_settings",
                    input_params={"user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
        name="answer_no_coding_query",
        description="Flag that the current request can be answered without code execution."
    )
    def answer_no_coding_query(
            user_query: str = Field(..., description="The user original query")
    ):
        start_time = time.time()
        result = {
            "user_query": user_query,
            "no_coding_query": True
        }
        execution_time_ms = (time.time() - start_time) * 1000
        if benchmark_logger and user_query:
            benchmark_logger.set_query(user_query)
        if benchmark_logger:
            benchmark_logger.log_tool_call(
                tool_name="answer_no_coding_query",
                input_params={"user_query": user_query},
                result=result,
                execution_time_ms=execution_time_ms
            )
        return result
    def _log_run(code, output, error, execution_mode, user_query, strategy):
        """Append a run record to microscope_toolset_runs.jsonl"""
        try:
            record = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "code": code,
                "output": output,
                "error": error,
                "execution_mode": execution_mode,
                "user_query": user_query,
                "strategy": strategy,
            }
            with open("microscope_toolset_runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as log_err:
            logger.warning(f"Failed to write run log: {log_err}")

    @mcp.tool(
        name="execute_python_code",
        description=(
            "Execute Python code on the microscope. The code runs in a namespace with a pre-configured `mmc` instance (CMMCorePlus/UniMMCore). Returns execution output or error details. "
            "IMPORTANT: Choose execution_mode carefully — using the wrong mode is a common source of bugs. "
            "Use 'live' whenever your code does multiple hardware operations that depend on each other (e.g., move stage then snap, or any loop with snap+analyze+move). "
            "Use 'buffered' only for simple single-shot operations or pure analysis of already-captured data. "
            "FEEDBACK WORKFLOWS (tracking, adaptive acquisition, timelapse with analysis): Use `run_mda_with_feedback(events, on_frame)` — a pre-configured helper available in the namespace. "
            "It runs pymmcore-plus MDA with a generator of MDAEvent objects, calling on_frame(image, event, metadata) synchronously after each frame. "
            "The generator can read shared state updated by on_frame to adapt subsequent events (e.g., re-center on a moving cell). "
            "This is the recommended approach over manual time.sleep() loops — it uses the microscope's hardware timing and handles all napari compatibility automatically. "
            "Requires execution_mode='live'. "
            "SMART ACQUISITION HELPERS (also available in namespace, require execution_mode='live'): "
            "- `center_on_cell(pixel_size_um=0.25, threshold_sigma=2.5, min_peak_above_bg=30.0, max_iterations=2)` → snap, find brightest region, re-center stage on it iteratively. Returns dict with image, centered (bool), peak, offset_um, centroid_px. "
            "- `find_bright_centroid(image, threshold_sigma=2.5)` → returns (cy, cx, area_px, peak) of bright region centroid. "
            "- `detect_cells(image, threshold_sigma=2.5, min_area_px=50, pixel_size_um=1.0, fill_holes=True, global_stats=None)` → returns list of cell dicts with centroid_px, area_um2, peak, bbox. Use fill_holes=True for brightfield. For multi-frame stacks, compute global (mean, std) once and pass as global_stats to avoid per-frame threshold drift. "
            "SLM / TARGETED STIMULATION: For optogenetics, FRAP, photoactivation — build pixel-accurate masks from segmentation (NOT bounding boxes). "
            "The SLM mask is 512x512 uint8 in viewport/camera space. Set via `mmc.setSLMDevice('SLM'); mmc.setSLMImage('SLM', mask); mmc.displaySLMImage('SLM')`. "
            "For dynamic experiments, update the mask each frame in the on_frame callback using `mmc.setSLMImage('SLM', new_mask); mmc.displaySLMImage('SLM')`. Always use the standard API — never use virtual-microscope internals like bridge.set_slm_mask(). "
            "Save mask stacks as 3D TIFFs alongside timelapses so the user can verify targeting."
        )
    )
    def execute_python_code(
            code: str = Field(..., description=(
                "Python code to execute. Constraints: "
                "- Use `mmc` (pre-configured CMMCorePlus instance) for hardware calls; do NOT re-instantiate it. "
                "- `run_mda_with_feedback(events, on_frame)` is available for MDA-based feedback workflows. "
                "- `center_on_cell(**kw)`, `find_bright_centroid(image)`, `detect_cells(image)` are available for smart acquisition. "
                "  events: Iterable[MDAEvent] (list or generator). on_frame: callback(image, event, metadata). "
                "  Returns list of (image, event, metadata) if on_frame is None. "
                "  MDAEvent fields: x_pos, y_pos (stage um), exposure (ms), min_start_time (s from MDA start), "
                "  index (dict e.g. {'t': 0, 'p': 0}), metadata (dict e.g. {'cell_id': 0}). "
                "  Generator pattern: yield MDAEvent(...) in a loop; on_frame updates shared state; generator reads it for next yield. "
                "- Do NOT access `viewer` directly (runs in daemon thread, no Qt/OpenGL access); use viewer_* tools instead. "
                "- Print results so they appear in the output. "
                "- After mmc.setXYPosition(), always call mmc.waitForDevice(mmc.getXYStageDevice()) before snapping. "
                "- To save multi-dimensional data (timelapse, multi-position), save as TIFF with tifffile.imwrite() then use viewer_add_image to display."
            )),
            execution_mode: Annotated[Literal["buffered", "live"], Field(description=(
                "Execution mode — CRITICAL choice: "
                "'buffered': hardware calls are intercepted and replayed atomically after code finishes. Redundant calls are deduplicated. "
                "WARNING: In buffered mode, all snapImage() calls produce the SAME image because they execute at the same moment. "
                "Use buffered ONLY for single-snap analysis or non-hardware code. "
                "'live': direct hardware access, each call executes immediately. "
                "REQUIRED for: any workflow involving move-then-snap, timelapse, tracking, multi-position imaging, "
                "run_mda_with_feedback(), or any code where hardware state must change between operations."
            ))] = "buffered",
            user_query: str = Field("", description="(Optional) The original user query, used for logging only."),
            strategy: str = Field("", description="(Optional) The strategy used, for logging only."),
            ) -> dict[str, Any]:
        """
        Prepares and executes Python code using the Execute agent.
        Returns a dictionary with 'output' (the execution result) and 'error' (if any).
        """
        start_time = time.time()
        result = None
        try:
            prepare_code_to_run = prepare_code(code)
            execution_output = executor.run_code_new(prepare_code_to_run, execution_mode)
            if "Error" in execution_output:
                logger.error({"tool": "execute_python_code", "code": code, "error": execution_output})
                _log_run(code, None, execution_output, execution_mode, user_query, strategy)
                result = {"code": code, "error": execution_output}
                return result
            elif 'viewer' in execution_output:
                err_msg = "Code references 'viewer' or 'napari.current_viewer()'. These are blocked because MCP tools run on a daemon thread — accessing the napari GUI directly will crash it. Use viewer_* MCP tools instead, or get_layer_data to export layer data to a TIFF file."
                logger.info({"tool": "execute_python_code", "code": code, "error": err_msg})
                _log_run(code, None, err_msg, execution_mode, user_query, strategy)
                result = {"code": code, "error": err_msg}
                return result
            else:
                logger.info({"tool": "execute_python_code", "code": code, "output": execution_output})
                _log_run(code, execution_output, None, execution_mode, user_query, strategy)
                result = {"code": code, "output": execution_output}
                return result
        except Exception as e:
            err_msg = f"Code preparation/execution failed: {e}"
            logger.error({"tool": "execute_python_code", "code": code, "error": err_msg})
            _log_run(code, None, err_msg, execution_mode, user_query, strategy)
            result = {"code": code, "error": err_msg}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="execute_python_code",
                    input_params={"code": code[:100] + "..." if len(code) > 100 else code, "execution_mode": execution_mode, "user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    # ------------------------------------------#
    # Simple microscope tools
    # ------------------------------------------#
    def _get_raw_mmc():
        """Get the raw mmc instance (CMMCorePlus or UniMMCore) from the executor namespace."""
        mmc_wrapper = executor.namespace["mmc"]
        return mmc_wrapper._mmc if hasattr(mmc_wrapper, "_mmc") else mmc_wrapper

    @mcp.tool(
        name="snap_image",
        description=(
            "Snap a single image with the current camera settings. Returns image metadata only (shape, dtype, min, max, mean) — NOT the pixel data itself. "
            "The image is also displayed automatically in the napari-micromanager live view. "
            "To access actual pixel data for analysis, use execute_python_code with mmc.snapImage() + mmc.getImage() instead."
        )
    )
    def snap_image(
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            raw_mmc = _get_raw_mmc()
            try:
                raw_mmc.snapImage()
                img = raw_mmc.getImage()
            except RuntimeError:
                # Fallback: some cameras (e.g. Photometrics PVCAM) fail on
                # snapImage() but work with single-frame sequence acquisition.
                logger.warning("snapImage() failed, falling back to single-frame sequence acquisition")
                raw_mmc.clearCircularBuffer()
                raw_mmc.startSequenceAcquisition(1, 0, True)
                timeout_ms = raw_mmc.getExposure() + 5000
                start_wait = time.time()
                while raw_mmc.isSequenceRunning():
                    time.sleep(0.05)
                    if (time.time() - start_wait) * 1000 > timeout_ms:
                        raw_mmc.stopSequenceAcquisition()
                        raise RuntimeError("Sequence acquisition timed out")
                if raw_mmc.getRemainingImageCount() < 1:
                    raise RuntimeError("No image returned from sequence acquisition fallback")
                img = raw_mmc.popNextImage()
            result = {
                "status": "success",
                "shape": list(img.shape),
                "dtype": str(img.dtype),
                "min": float(np.min(img)),
                "max": float(np.max(img)),
                "mean": float(np.mean(img)),
            }
            return result
        except Exception as e:
            result = {"status": "error", "message": str(e)}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="snap_image",
                    input_params={"user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
        name="move_stage",
        description=(
            "Move the XY stage to an absolute or relative position. Returns the final stage position after the move. "
            "Coordinate system: stage position defines the center of the camera viewport in world coordinates (micrometers). "
            "The mapping between pixels and world coordinates: world = stage + (pixel - 256) * pixel_size_um. "
            "You must determine pixel_size_um for the current setup (e.g., via calibration or get_microscope_settings). "
            "To center a detected object at pixel (px, py): compute world_x = stage_x + (px - 256) * pixel_size_um, then move directly to (world_x, world_y). "
            "This tool automatically waits for the stage to finish moving before returning."
        )
    )
    def move_stage(
        x: float = Field(..., description="X coordinate (absolute) or X displacement (relative), in micrometers. Stage X maps to image columns."),
        y: float = Field(..., description="Y coordinate (absolute) or Y displacement (relative), in micrometers. Stage Y maps to image rows."),
        relative: bool = Field(False, description="If True, move relative to current position. If False, move to absolute coordinates."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            raw_mmc = _get_raw_mmc()
            if relative:
                raw_mmc.setRelativeXYPosition(x, y)
            else:
                raw_mmc.setXYPosition(x, y)
            raw_mmc.waitForDevice(raw_mmc.getXYStageDevice())
            final_x, final_y = raw_mmc.getXPosition(), raw_mmc.getYPosition()
            result = {"status": "success", "x": final_x, "y": final_y}
            return result
        except Exception as e:
            result = {"status": "error", "message": str(e)}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="move_stage",
                    input_params={"x": x, "y": y, "relative": relative, "user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
        name="set_objective",
        description="Switch the objective lens by setting the state label on the Objective device. Returns the new current objective."
    )
    def set_objective(
        label: str = Field(..., description="Objective label to switch to (e.g. 'Nikon 10X S Fluor', '20x'). Use get_microscope_settings to discover available labels."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            raw_mmc = _get_raw_mmc()
            # Find the Objective state device
            obj_device = None
            for dev in raw_mmc.getLoadedDevices():
                dev_type = raw_mmc.getDeviceType(dev)
                # DeviceType 6 = StateDevice
                if "objective" in dev or (hasattr(dev_type, 'value') and dev_type.value == 6 and "objective" in dev):
                    obj_device = dev
                    break
            if obj_device is None:
                # Fallback: try common names
                for name in ["Objective", "ObjectiveTurret", "Nosepiece"]:
                    try:
                        raw_mmc.getDeviceType(name)
                        obj_device = name
                        break
                    except Exception:
                        continue
            if obj_device is None:
                result = {"status": "error", "message": "Could not find an Objective device. Use get_microscope_settings to check available devices."}
                return result
            raw_mmc.setProperty(obj_device, "Label", label)
            raw_mmc.waitForDevice(obj_device)
            current = raw_mmc.getProperty(obj_device, "Label")
            result = {"status": "success", "device": obj_device, "objective": current}
            return result
        except Exception as e:
            result = {"status": "error", "message": str(e)}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="set_objective",
                    input_params={"label": label, "user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
        name="get_stage_position",
        description=(
            "Return the current X, Y, and Z stage positions in micrometers. "
            "The stage position defines the center of the camera viewport in world coordinates. "
            "An object at pixel (px, py) in an image is at world position (stage_x + (px - 256) * pixel_size_um, stage_y + (py - 256) * pixel_size_um), "
            "where pixel_size_um depends on the current objective and camera configuration."
        )
    )
    def get_stage_position(
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        start_time = time.time()
        result = None
        try:
            raw_mmc = _get_raw_mmc()
            x = raw_mmc.getXPosition()
            y = raw_mmc.getYPosition()
            try:
                z = raw_mmc.getZPosition() if hasattr(raw_mmc, 'getZPosition') else raw_mmc.getPosition()
            except Exception:
                z = None
            result = {"status": "success", "x": x, "y": y, "z": z}
            return result
        except Exception as e:
            result = {"status": "error", "message": str(e)}
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="get_stage_position",
                    input_params={"user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )

    @mcp.tool(
            name="get_microscope_events",
            description="Retrieve recent microscope activity events to check current state and discover user actions. Returns a chronological list of events (image captures, property changes, exposure adjustments, etc.) with timestamps. Use this to: verify that commanded actions completed successfully, discover manual user interactions with the GUI, check current microscope state, or debug timing issues. Each event includes type, timestamp, and relevant data."
    )
    def get_microscope_events(
        limit: int = Field(100, description="Maximum number of recent events to retrieve (default 100)."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        """Get recent microscope events from the event cache"""
        start_time = time.time()
        result = None
        try:
            events = event_cache.get_recent_events(limit=limit)
            result = {
                "status": "success",
                "event_count": len(events),
                "events": events
            }
            return result
        except Exception as e:
            result = {
                "status": "error",
                "message": str(e)
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="get_microscope_events",
                    input_params={"limit": limit, "user_query": user_query},
                    result={"status": result.get("status"), "event_count": result.get("event_count")},
                    execution_time_ms=execution_time_ms
                )
        
    @mcp.tool(
    name="get_last_microscope_event",
    description="Get the most recent microscope event. Useful for quick checks like 'did my last snap() succeed?' or 'what was the last property change?'"
)
    def get_last_microscope_event(
        event_type: str | None = Field(None, description="Filter by event type or None for any event."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        """Get the most recent event"""
        start_time = time.time()
        result = None
        try:
            event = event_cache.get_last_event(event_type=event_type)
            if event:
                result = {
                    "status": "success",
                    "event": event
                }
                return result
            else:
                result = {
                    "status": "success",
                    "event": None,
                    "message": "No events found"
                }
                return result
        except Exception as e:
            result = {
                "status": "error",
                "message": str(e)
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="get_last_microscope_event",
                    input_params={"event_type": event_type, "user_query": user_query},
                    result={"status": result.get("status")},
                    execution_time_ms=execution_time_ms
                )
            
    # New Tool added
    # ------------------------------------------#
    # Napari Viewer
    # ------------------------------------------#
    @mcp.tool(
        name="viewer_session_information",
        description="Retrieve detailed information about the current napari viewer session. Returns metadata about the napari-micromanager viewer state including window size, available layers, camera position, and current display settings. Use this to understand the current state of the microscopy viewer before making changes."
    )
    def viewer_session_information(
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Return information regarding the viewer session of napari micromanager
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('session_information')
        else:
            return viewer.session_information()
    
    # List of layer
    @mcp.tool(
        name="viewer_list_of_layers",
        description="Get a list of all layers currently loaded in the napari viewer with their properties (name, type, visibility, opacity, colormap). Use this to understand what image layers, label layers, and other data layers are present in the viewer and plan layer manipulation operations."
    )
    def viewer_list_of_layers(
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Return a list of layers with all information
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('list_of_layers')
        else:
            return viewer.list_of_layers()
    
    # screenshot
    @mcp.tool(
        name="viewer_screenshot", 
        description="Capture a screenshot of the napari viewer's current state. This renders all visible layers and returns the image as an array. Set canvas_only=false to include GUI elements like scale bars and labels, or canvas_only=true to capture only the image data. Use this to visually inspect napari UI images.",
    )
    def viewer_screenshot(
        canvas_only: bool = Field(..., description="If True, capture only the canvas (image data) without GUI elements. If False, include scale bars, labels, and other UI elements in the screenshot."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Return the ImageContent to pass the image data to the LLM
        """
        # Use viewer proxy if available to execute on main thread
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('screenshot', canvas_only=canvas_only)
        else:
            return viewer.screenshot(canvas_only=canvas_only)
    
    @mcp.tool(
        name="viewer_layer_screenshot", 
        description="Capture the image data of a specific layer from the napari viewer. Provide the exact layer name to isolate and render only that layer's data. Useful for examining individual microscopy channels, labeled regions, or segmentation masks without interference from other layers."
    )
    def viewer_layer_screenshot(
        layer_name: str = Field(..., description="The exact name of the layer to capture. Use viewer_list_of_layers to see available layer names."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Return the ImageContent of a specific layer to pass to the LLM
        """
        # Use viewer proxy if available to execute on main thread
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('layer_screenshot', layer_name=layer_name)
        else:
            return viewer.layer_screenshot(layer_name=layer_name)

    @mcp.tool(
        name="get_layer_data",
        description=(
            "Export raw pixel data from a napari layer to a TIFF file on disk. "
            "This is the safe data bridge between the viewer and execute_python_code — "
            "the exported TIFF can be loaded with tifffile.imread() inside executed code. "
            "Supports Image and Labels layers. Returns metadata including path, shape, dtype, "
            "and type-specific fields (num_labels/unique_labels for Labels; min/max/mean for Image). "
            "Use this instead of napari.current_viewer() which is unsafe from the daemon thread."
        )
    )
    def get_layer_data(
        layer_name: str = Field(..., description="Exact name of the layer to export. Use viewer_list_of_layers to see available names."),
        save_path: str | None = Field(None, description="Optional file path for the TIFF output. Defaults to /tmp/<sanitized_layer_name>.tif."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        """Export a layer's raw numpy data to a TIFF file."""
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('get_layer_data', layer_name=layer_name, save_path=save_path)
        else:
            return viewer.get_layer_data(layer_name=layer_name, save_path=save_path)

    @mcp.tool(
        name="view_image",
        description=(
            "Load an image from disk and return it as visual content that the agent can see directly. "
            "Supports TIFF (including multi-frame), PNG, and JPEG. Optionally overlay a segmentation mask "
            "as green contours. Use this to visually verify segmentation, count cells, assess image quality, "
            "or decide on analysis strategy. Unlike viewer_screenshot (which captures the napari canvas), "
            "this loads arbitrary image files from disk. "
            "Contrast is auto-scaled using percentiles (1st-99.9th) to handle low-contrast microscopy images "
            "and ignore hot pixels. For fine contrast control, use viewer_screenshot instead — the user can "
            "adjust contrast limits interactively in napari before you take the screenshot."
        ),
    )
    def view_image(
        image_path: str = Field(..., description="Path to the image file (TIFF, PNG, JPEG)."),
        overlay_path: str | None = Field(None, description="Optional path to a segmentation/label mask. Contours will be drawn as green outlines on the image."),
        frame_index: int = Field(0, description="For multi-frame TIFFs, which frame to display (0-indexed)."),
        query: str = Field("", description="Context about what to look for in the image. Returned alongside the image as text."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Load an image from disk, optionally composite a segmentation overlay,
        and return as ImageContent so the LLM can see it.
        """
        import cv2
        from PIL import Image as PILImage
        import tifffile

        # Load image
        path = str(image_path)
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
        else:
            img = np.array(PILImage.open(path))

        # Handle multi-dimensional arrays: select frame
        if img.ndim == 3 and img.shape[-1] not in (3, 4):
            # Shape is (T, H, W) — select frame
            idx = min(frame_index, img.shape[0] - 1)
            img = img[idx]
        elif img.ndim == 4:
            # Shape is (T, H, W, C) — select frame
            idx = min(frame_index, img.shape[0] - 1)
            img = img[idx]

        # Normalize to uint8 using percentile scaling to handle hot pixels
        # and low-contrast microscopy images (e.g., 16-bit with bg~200, fg~300,
        # or uint8 with narrow range like 4-72)
        fimg = img.astype(np.float32)
        p_low = np.percentile(fimg, 1)
        p_high = np.percentile(fimg, 99.9)
        if p_high > p_low:
            img = np.clip((fimg - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        elif fimg.max() > 0:
            img = np.clip(fimg / fimg.max() * 255, 0, 255).astype(np.uint8)
        else:
            img = np.zeros_like(fimg, dtype=np.uint8)

        # Convert grayscale to RGB for overlay drawing
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Overlay segmentation contours if provided
        if overlay_path is not None:
            overlay_str = str(overlay_path)
            if overlay_str.lower().endswith(('.tif', '.tiff')):
                mask = tifffile.imread(overlay_str)
            else:
                mask = np.array(PILImage.open(overlay_str))

            # Handle multi-frame mask
            if mask.ndim == 3 and mask.shape[-1] not in (3, 4):
                idx = min(frame_index, mask.shape[0] - 1)
                mask = mask[idx]
            elif mask.ndim == 4:
                idx = min(frame_index, mask.shape[0] - 1)
                mask = mask[idx]

            # Draw contours for each label
            binary = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        # Resize if any edge exceeds 1568px (API limit)
        h, w = img.shape[:2]
        max_edge = 1568
        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode as PNG base64
        pil_img = PILImage.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Build metadata text
        meta = {"image_path": path, "frame_index": frame_index, "shape": list(img.shape[:2])}
        if overlay_path is not None:
            meta["overlay_path"] = str(overlay_path)
        if query:
            meta["query"] = query

        return [
            TextContent(type="text", text=json.dumps(meta)),
            ImageContent(type="image", data=base64_img, mimeType="image/png"),
        ]

    # tools for open interact with napari viewer
    @mcp.tool(
        name="viewer_add_image",
        description=(
            "Display an image in the napari viewer as a new layer. Two input methods: "
            "(1) Provide a file 'path' to a TIFF/PNG/JPEG — recommended for large or multi-dimensional data generated by execute_python_code (save with tifffile.imwrite(), then pass the path here). "
            "(2) Provide 'data' as a numpy array directly — works for small arrays but impractical for large image stacks over MCP JSON. "
            "Supports extra dimensions: shape (T, Z, H, W) creates time+Z sliders, (N, H, W) creates a slider for dimension N. "
            "Use colormap, blending, and channel_axis for visualization control."
        )
    )
    def viewer_add_image(
        data: NDArray | list[NDArray] | None = Field(None, description="Image data as a numpy array (2D, 3D, or 4D). For large data, prefer saving to TIFF and using 'path' instead."),
        path: str | None = Field(None, description="File path to load (TIFF, PNG, JPEG, etc.). Recommended for multi-dimensional data from execute_python_code workflows."),
        name: str | None = Field(None, description="Optional name for the image layer. If not provided, the filename will be used."),
        colormap: str | None = Field(None, description="Colormap to apply to the image (e.g., 'gray', 'viridis', 'magma', 'red', 'green', 'blue'). Default is 'gray' for grayscale images."),
        blending: str | None = Field(None, description="Blending mode for layer compositing: 'translucent' (default), 'additive', or 'opaque'."),
        channel_axis: int | str | None = Field(None, description="Axis index for multi-channel images. If provided, channels will be split into separate layers."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Add an image layer from a file path
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_image', img_data=data, path=path, name=name, colormap=colormap, blending=blending, channel_axis=channel_axis)
        else:
            return viewer.add_image(img_data=data,path=path, name=name, colormap=colormap, blending=blending, channel_axis=channel_axis)
    
    @mcp.tool(
        name="viewer_add_labels",
        description="Add a segmentation/labels layer to the napari viewer. You can provide EITHER a file path to a labels image OR a numpy array with labeled regions directly. Each unique integer value represents a distinct region (e.g., individual cells, nucleus, organelles). Use this to display cell detection results, segmentation masks, or any labeled image analysis results with automatic color mapping for easy visualization of individual regions."
    )
    def viewer_add_labels(
        path: str | None = Field(None, description="File path to the labels image file (TIFF, PNG, etc.). Use this if loading from disk. Mutually exclusive with img_data."),
        img_data: NDArray | None = Field(None, description="Labeled mask as a 2D, 3D, or 4D integer array where each unique value represents a different region/object. Supports Z-stacks and time series. Mutually exclusive with path."),
        name: str | None = Field(None, description="Optional name for the labels layer in the viewer. If not provided, a default name will be generated."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Add a labels layer from a file
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_labels', path=path, img_data=img_data, name=name)
        else:
            return viewer.add_labels(path=path, img_data=img_data, name=name)
        
    @mcp.tool(
        name="viewer_add_points",
        description="Add a points layer to the napari viewer for marking locations of interest. Provide a list of coordinate pairs (2D) or triples (3D) representing point positions in pixel/voxel space. Optionally set the layer name and point size for visualization. Use this to annotate cell locations, mark regions of interest, indicate measurement points, or overlay coordinate data on microscopy images."
    )
    def viewer_add_points(
        points: list[list[float]] = Field(..., description="List of point coordinates. For 2D: [[y1, x1], [y2, x2], ...]. For 3D: [[z1, y1, x1], [z2, y2, x2], ...]. Coordinates are in pixel/voxel space."),
        name: str | None = Field(None, description="Optional name for the points layer."),
        size: int | str = Field(10, description="Display size (diameter) of the points in pixels."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Add a points layer
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_points', points=points, name=name, size=size)
        else:
            return viewer.add_points(points=points, name=name, size=size)
        
    
    @mcp.tool(
        name="viewer_remove_layer",
        description="Remove a layer from the napari viewer by its exact name. Use this to clean up the viewer workspace by deleting intermediate processing results, redundant layers, or layers that are no longer needed for analysis. Check the current layers with viewer_list_of_layers before removing."
    )
    def viewer_remove_layer(
        name: str = Field(..., description="The exact name of the layer to remove. Must match a layer name in the viewer."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Remove an existince layer
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('remove_layer', name=name)
        else:
            return viewer.remove_layer(name=name)
    

    @mcp.tool(
        name="viewer_set_layer_properties",
        description="Modify visual properties of a layer in the napari viewer. Adjust visibility (True/False), opacity (0-1, where 0 is transparent), colormap ('viridis', 'magma', 'red', etc.), blending mode ('additive', 'translucent'), contrast limits for brightness/contrast adjustment, gamma for exposure, and optionally rename the layer. Use this to improve visualization, highlight specific features, or enhance contrast for better image analysis."
    )
    def viewer_set_layer_properties(
        name: str = Field(..., description="The name of the layer to modify."),
        visible: bool | None = Field(None, description="Set layer visibility: True to show, False to hide."),
        opacity: float | None = Field(None, description="Layer opacity from 0 (transparent) to 1 (opaque)."),
        colormap: str | None = Field(None, description="Colormap name (e.g., 'gray', 'viridis', 'magma', 'red', 'green', 'blue')."),
        blending: str | None = Field(None, description="Blending mode: 'translucent', 'additive', or 'opaque'."),
        contrast_limits: list[float] | None = Field(None, description="Two-element list [min, max] for contrast/brightness adjustment."),
        gamma: float | str | None = Field(None, description="Gamma correction value for exposure adjustment (typically 0.5-2.0)."),
        new_name: str | None = Field(None, description="New name to rename the layer to."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Set common properties on a layer name
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_layer_properties', name=name, visible=visible, opacity=opacity, colormap=colormap, blending=blending, contrast_limits=contrast_limits, gamma=gamma, new_name=new_name)
        else:
            return viewer.set_layer_properties(name=name, visible=visible, opacity=opacity, colormap=colormap, blending=blending, contrast_limits=contrast_limits, gamma=gamma, new_name=new_name)
    @mcp.tool(
        name="viewer_reorder_layer",
        description="Change the stacking order (z-order) of layers in the napari viewer. Specify the layer name and either an absolute index, or position it before/after another named layer. Use this to control which layers appear on top when layers overlap, which affects visibility in multi-layer microscopy visualizations where layer stacking order matters for interpretation."
    )
    def viewer_reorder_layer(
        name: str = Field(..., description="Name of the layer to reorder."),
        index: int | str | None = Field(None, description="Absolute position index (0 = bottom). Mutually exclusive with before/after."),
        before: str | None = Field(None, description="Name of layer to position this layer before. Mutually exclusive with index/after."),
        after: str | None = Field(None, description="Name of layer to position this layer after. Mutually exclusive with index/before."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Reorder a layer by name
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('reorder_layer', name=name, index=index, before=before, after=after)
        else:
            return viewer.reorder_layer(name=name, index=index, before=before, after=after)
    
    @mcp.tool(
        name="viewer_set_active_layer",
        description="Select/activate a specific layer in the napari viewer by name. The active layer is highlighted in the layers panel and operations like drawing, annotation, or selection tools apply to this layer. Use this when you need to work with a specific layer or prepare a layer for editing."
    )
    def viewer_set_active_layer(
        name: str = Field(..., description="Name of the layer to activate/select."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Set the selected/active layer by name
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_active_layer', name=name)
        else:
            return viewer.set_active_layer(name=name)
    
    @mcp.tool(
        name="viewer_reset_view",
        description="Reset the camera view to fit all visible data layers optimally in the viewer window. This adjusts zoom and pan to show the entire image extent. Use this to get a complete overview of your data after zooming into specific regions, or to standardize the view between different analyses."
    )
    def viewer_reset_view(
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Reset the camera view to fit the data
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('reset_view')
        else:
            return viewer.reset_view()
    
    @mcp.tool(
        name="viewer_set_camera",
        description="Control the camera viewing parameters in the napari viewer. Set the center position to pan to a specific region, zoom level to magnify (larger = more zoom), and angle for 3D rotation (if working in 3D mode). Use this to navigate to regions of interest, zoom in on details, or create consistent viewing angles for image documentation."
    )
    def viewer_set_camera(
        center: list[float] | None = Field(None, description="Center position coordinates [y, x] for 2D or [z, y, x] for 3D to pan the camera to."),
        zoom: float | str | None = Field(None, description="Zoom level (larger values = more magnification). Typical range: 0.5 to 10+."),
        angle: float | str | None = Field(None, description="Rotation angle in degrees for 3D viewing mode."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Set the camera properties: center, zoom, angle
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_camera', center=center, zoom=zoom, angle=angle)
        else:
            return viewer.set_camera(center=center, zoom=zoom, angle=angle)
    
    @mcp.tool(
        name="viewer_set_ndisplay", 
        description="Switch the napari viewer between 2D and 3D display modes. Set ndisplay=2 for standard 2D microscopy slice viewing, or ndisplay=3 for 3D volumetric visualization when working with Z-stack or 3D image data. Use this to toggle between 2D slice inspection and 3D volume rendering."
    )
    def viewer_set_ndisplay(
        ndisplay: int | str = Field(..., description="Number of displayed dimensions: 2 for 2D view, 3 for 3D volumetric view."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Set number of displayed dimension (2 or 3)
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_ndisplay', ndisplay=ndisplay)
        else:
            return viewer.set_ndisplay(ndisplay=ndisplay)
    
    @mcp.tool(
        name="viewer_set_dims_current_step",
        description="Navigate through a specific dimension (axis) of multi-dimensional image data. Provide the axis name/index (e.g., 'Z' for Z-stack depth, 0, 1, 2, etc.) and the step value. Use this to browse through Z-slices in a Z-stack, time frames in a time-lapse, or channels in multi-channel images. This is equivalent to moving the slider for that dimension."
    )
    def viewer_set_dims_current_step(
        axis: int | str = Field(..., description="Axis identifier: integer index (0, 1, 2, ...) or axis name ('Z', 'T', 'C' for Z-stack, time, channel)."),
        value: int | str = Field(..., description="Step value (slice index) to navigate to along the specified axis. Must be within valid range for that dimension."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Set the current step (slider position for a specific axis)
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_dims_current_step', axis=axis, value=value)
        else:
            return viewer.set_dims_current_step(axis=axis, value=value)
    
    @mcp.tool(
        name="viewer_set_grid",
        description="Toggle the display of a pixel grid overlay in the napari viewer. Set enabled=true to show the grid (useful for precise pixel-level measurements and alignment), or enabled=false to hide it for a cleaner view. Use this to switch between detailed pixel-level work and overview visualization modes."
    )
    def viewer_set_grid(
        enabled: bool | str = Field(True, description="Enable (True) or disable (False) the pixel grid overlay."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ):
        """
        Enable or disable grid view
        """
        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('set_grid', enabled=enabled)
        else:
            return viewer.set_grid(enabled=enabled)
        

    @mcp.tool(
            name="viewer_add_tracks",
            description="Add a tracks layer to the napari viewer for visualizing object trajectories over time."
    )
    def viewer_add_tracks(
        track_data: NDArray = Field(description="""NxD+1 NumPy Array or list containig the coordinates of N vertices with a
                track ID and coordinats in D dimensions. The ordering of these dimensions is the same
                as the ordering of the dimensions for image layers. This array is always accessible through the
                layer.data property and will grow or shrink as new tracks are either added or deleted.
                The Tracks layer assumes the first column is the track_id, the second column is the time axis,
                and columns 3-5 are Z, Y, and X, respectively. Other feature can be added in other coloumns.
                Each row is one vertex in a track. All vertices with the same track_id are joined into a single track."""),
        features: dict[str, Any] | None = Field(None, description="Features table where each row corresponds to a point and each column is a feature."),
        tail_width: float | None = Field(None, description="Float value representing the width of the track tails in pixels."),
        tail_length: float | None = Field(None, description="Float value representing the length of the positive (backward in time) tails in units of time."),
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        """It add a Track layer to the layer List."""

        if viewer_proxy is not None:
            return viewer_proxy.call_on_main_thread('add_tracks', track_data=track_data, features=features, tail_width=tail_width, tail_length= tail_length)
        else:
            return viewer.add_tracks(data=track_data, features=features, tail_width=tail_width, tail_length=tail_length)
    
    # TODO: add timelapse_screenshot later

    
    @mcp.tool(
            name="request_user_clarification",
            description="Allow the MCP Client to interact with the user asking for clarification or some new input to answer the user's request."
    )
    async def request_user_clarification(
        message: str = Field(..., description="The clarification message to present to the user."),
        ctx: Context = None,
        user_query: str = Field("", description="(Optional) The original user query, used for logging only.")
    ) -> dict[str, Any]:
        """Request user clarification from MCP client"""
        start_time = time.time()
        result = None
        try:
            # clarification
            class elicitClarification(BaseModel):
                agent_message: str = Field(description="The original Main Agent message.")
                user_answer: str = Field(description="The user answer directed to the Main Agent.")

            result_elicit = await ctx.elicit(message, elicitClarification)

            if result_elicit.action == "accept":
                result = result_elicit.data
            elif result_elicit.action == "decline":
                result = {
                    "agent_message": message,
                    "user_answer": "Operation declined"
                }
            elif result_elicit.action == "cancel":
                result = {
                    "agent_message": message,
                    "user_answer": "Operation cancelled"
                }

            return result
        except Exception as e:
            logger.error(f"Error in request_user_clarification: {e}", exc_info=True)
            result = {
                "agent_message": message,
                "user_answer": f"Elicitation not supported by this MCP client: {str(e)}. "
                               "Note: ctx.elicit() is supported by some MCP clients (e.g., VS Code Copilot) but not all."
            }
            return result
        finally:
            execution_time_ms = (time.time() - start_time) * 1000
            if benchmark_logger and user_query:
                benchmark_logger.set_query(user_query)
            if benchmark_logger and result is not None:
                benchmark_logger.log_tool_call(
                    tool_name="request_user_clarification",
                    input_params={"message": message, "user_query": user_query},
                    result=result,
                    execution_time_ms=execution_time_ms
                )
    
    #@mcp.tool(
    #    name="tool_for_segmenting",
    #    description="Segment a cellular image from a microscope using Cellpose. This tools is ideal to segment" \
    #    "single/multi-cells images from microscope. They can have single or multi channel fluorescence" \
    #    "and can a"
    #)
    #def tool_for_segmenting(
    #    path: str | None = Field(None, description="File path to the image or list of images (.tif)."),
    #    img: NDArray | list[NDArray] | None = Field(None, description="It can be list of 2D/3D/4D images, or array of 2D/3D/4D images. Images must have 3 channels."),
    #    mask_name: str | None = Field(None, description="Name of the mask that will use to describe the mask."),
    #    batch_size: int = Field(8, description="Number of 256x256 patches to run simultaneously on the GPU (can make smaller or bigger depending on GPU memory usage). Defaults to 8"),
    #    resample: bool = Field(True, description="Run dynamics at original image size (will be slower but create more accurate boundaries)."),
    #    channels_axis: int | None = Field(None, description="Channel axis in element of list x, or of np.ndarray x. if None, channels dimension is attempted to be automatically determined. Defaults to None."), 
    #    z_axis: int | None = Field(None, description="Z axis in element of list x, or of np.ndarray x. if None, z dimension is attempted to be automatically determined. Defaults to None."),
    #    normalize: bool = Field(True, description="if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel; can also pass dictionary of parameters (all keys are optional, default values shown): " \
    #    "- ”lowhigh”=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored) " \
    #    "- ”sharpen”=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels " \
    #    "- ”normalize”=True ; run normalization (if False, all following parameters ignored) " \
    #    "- ”percentile”=None : pass in percentiles to use as list [perc_low, perc_high] " \
    #    "- ”tile_norm_blocksize”=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100) " \
    #    "- ”norm3D”=True ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode. " \
    #    "Defaults to True."),
    #    rescale: float | None = Field(None, description="Resize factor for each image, if None, set to 1.0; (only used if diameter is None). Defaults to None."),
    #    diameter: float | list[float] | None = Field(None, description="diameters are used to rescale the image to 30 pix cell diameter."),
    #    flow_threshold: float = Field(0.4, description="Flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4."),
    #    cellprob_threshold: float = Field(0.0, description="All pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0."),
    #    augment: bool = Field(False, description="Tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.")
    #) -> dict[str, Any]:
    #    
    #    return viewer.segment_image(path=path,img=img, mask_name=mask_name,
    #                                            batch_size=batch_size, resample=resample, channels_axis=channels_axis,
    #                                            z_axis=z_axis, normalize=normalize, rescale=rescale,
    #                                            diameter=diameter, flow_threshold=flow_threshold,
    #                                            cellprob_threshold=cellprob_threshold, augment=augment)


    return mcp


