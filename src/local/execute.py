import importlib
import importlib.util
import subprocess
import sys
from io import StringIO
from contextlib import redirect_stdout,redirect_stderr

from pymmcore_plus import CMMCorePlus
import logging
import ast

from src.local.gatekeeper_core import GatekeeperCore
from src.local.mda_helpers import run_mda_with_feedback
from src.local.microscopy_utils import (
    find_bright_centroid, center_on_cell, detect_cells,
)

#  logger
logger = logging.getLogger("Execute")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


class Execute:

    # Core classes the agent must not re-instantiate
    _BLOCKED_CONSTRUCTORS = {"CMMCorePlus", "UniMMCore"}
    # Hardware-config methods the agent must not call
    _BLOCKED_METHODS = {"loadSystemConfiguration", "loadConfig"}

    def __init__(self, mmc: CMMCorePlus, filename: str | None = None):
        self.namespace = {}

        if filename is not None:
            mmc.loadSystemConfiguration(fileName=filename)
            logger.info("configuration file of the microscope was loaded")

        self.namespace["mmc"] = GatekeeperCore(mmc)
        self.namespace["run_mda_with_feedback"] = lambda events, on_frame=None: run_mda_with_feedback(mmc, events, on_frame)
        self.namespace["center_on_cell"] = lambda **kw: center_on_cell(mmc, **kw)
        self.namespace["find_bright_centroid"] = find_bright_centroid
        self.namespace["detect_cells"] = detect_cells
        logger.info("mmc instance is loaded into the namespace")
        logger.info("Execute initialized")


    def _install_library(self, module: str):
        """Install missing packages using pip with better error handling"""
        try:
            # First check if module is already available
            spec = importlib.util.find_spec(module)
            if spec is not None:
                return True

            logger.info(f"Installing package: {module}")

            # Install the package
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", module
            ], capture_output=True, text=True, timeout=120)  # Add timeout

            if result.returncode == 0:
                logger.info(f"Successfully installed {module}")
                return True
            else:
                logger.error(f"Failed to install {module}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing {module}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Installation failed for {module}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing {module}: {e}")
            return False
        
    def _get_missing_imports(self, code: str) -> list[str]:
        """Parse AST and return top-level module names that are not currently installed."""
        missing = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split('.')[0]
                    if importlib.util.find_spec(mod) is None and mod not in missing:
                        missing.append(mod)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split('.')[0]
                if importlib.util.find_spec(mod) is None and mod not in missing:
                    missing.append(mod)
        return missing

    def _preimport_dependencies(self, code: str) -> list[str]:
        """Validate imports, auto-install missing packages, return list of any that still failed."""
        missing = self._get_missing_imports(code)
        failed = []
        for mod in missing:
            logger.info(f"Package '{mod}' not found — attempting install.")
            if not self._install_library(mod):
                failed.append(mod)
        # pre-import all top-level modules so they are in sys.modules before exec
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return failed
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split('.')[0]
                    if mod not in failed:
                        try:
                            importlib.import_module(mod)
                        except Exception:
                            pass
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split('.')[0]
                if mod not in failed:
                    try:
                        importlib.import_module(mod)
                    except Exception:
                        pass
        return failed

    

    def run_code_new(self, code: str, execution_mode: str = "buffered"):
        """Execute code after pre-importing deps. Use GatekeeperCore to buffer hardware calls and commit on succession"""

        # validate mode
        if execution_mode not in ["buffered", "live"]:
            return f"Invalid execution mode: {execution_mode}. Must be 'buffered' or 'live'. "

        # Check code before running it
        if not self.is_safe_viewer(code):
            return "viewer"

        safe, reason = self.is_safe_code(code)
        if not safe:
            return f"Safety Error: {reason}"

        # Validate and pre-import dependencies
        try:
            failed = self._preimport_dependencies(code)
        except Exception as e:
            return f"Dependency parsing error: {e}"
        if failed:
            return f"Missing packages that could not be installed: {', '.join(failed)}. Install them manually and retry."
        
        # Execute code in specif mode
        if execution_mode == "live":
            logger.info(f"Executing code in live mode")
            return self._run_code_live(code)
        else:
            logger.info(f"Exeecuting code in buffer mode")
            return self._run_code_buffered(code)
        

    def _run_code_buffered(self, code: str):
        """Execute code after pre-importing deps. Use GatekeeperCore to buffer hardware calls and commit on succession"""
        
        # Static analysis of mmc usage (convervatives because LLM Agent makes mistakes!)
        mmc_obj = self.namespace["mmc"]
        # Get current state before code run
        try:
            snapshot = mmc_obj.snapshot_state()
        except Exception as e:
            logger.warning(f"Could not snapshot state: {e}")
            snapshot = None

        # Execute user code once
        try:
            out_f = StringIO()
            err_f = StringIO()
            # code execution
            with redirect_stdout(out_f), redirect_stderr(err_f):
                exec(code, self.namespace)
            # reading output+errors
            stdout_text = out_f.getvalue().strip()
            stderr_text = err_f.getvalue().strip()
            read_output = stdout_text

            if stderr_text:
                read_output = (read_output + "\nWarnings/Errors: " + stderr_text).strip()
            # Commit buffer changes
            try:
                mmc_obj.commit(real_mmc=mmc_obj._mmc, check_snapshot=snapshot)
                logger.info("Code executed successfully and committed.")
            except Exception as e:
                logger.error(f"Commit failed: {e}")
                mmc_obj.clear_pending()
                return f"Commit failed: {e}"
                
            
            return read_output if read_output else "Code executed successfully (no output)"
        
        except ModuleNotFoundError as e:

            module_name = str(e).split("'")[1] if "'" in str(e) else str(e)
            logger.error(f"Module not found during execution (unexpected): {module_name}")
            mmc_obj.clear_pending()
            return f"Module not found during execution: {module_name}"
        
        except Exception as e:

            error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            mmc_obj.clear_pending()
            return error_msg
        
    def _run_code_live(self, code: str):
        """Execute code after pre-importing deps. Use GatekeeperCore to buffer hardware calls and commit on succession"""

        # Static analysis of mmc usage (convervatives because LLM Agent makes mistakes!)
        shadow_mmc_obj = self.namespace["mmc"]

        self.namespace["mmc"] = shadow_mmc_obj._mmc

        # Execute user code once
        try:
            out_f = StringIO()
            err_f = StringIO()
            # code execution
            with redirect_stdout(out_f), redirect_stderr(err_f):
                exec(code, self.namespace)
            # reading output+errors
            stdout_text = out_f.getvalue().strip()
            stderr_text = err_f.getvalue().strip()
            read_output = stdout_text

            if stderr_text:
                read_output = (read_output + "\nWarnings/Errors: " + stderr_text).strip()

            logger.info("Code executed successfully.")
            return read_output if read_output else "Code executed successfully (no output)"
        
        except ModuleNotFoundError as e:

            module_name = str(e).split("'")[1] if "'" in str(e) else str(e)
            logger.error(f"Module not found during execution (unexpected): {module_name}")
            return f"Module not found during execution: {module_name}"
        
        except Exception as e:

            error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
        finally:
            self.namespace["mmc"] = shadow_mmc_obj


    def is_safe_viewer(self, code: str):
        """
        Checks if the code references 'viewer' or 'napari.current_viewer()'.

        Both are blocked because MCP tools run on a daemon thread while
        napari/Qt GUI must be accessed from the main thread. All viewer access
        must go through the ThreadSafeViewerProxy (viewer_* MCP tools).
        To get layer pixel data, use get_layer_data → TIFF → tifffile.imread.
        """
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Block bare 'viewer' name
            if isinstance(node, ast.Name) and node.id == 'viewer':
                return False
            # Block napari.current_viewer() pattern
            if (isinstance(node, ast.Attribute)
                    and node.attr == 'current_viewer'
                    and isinstance(node.value, ast.Name)
                    and node.value.id == 'napari'):
                return False

        return True

    def is_safe_code(self, code: str) -> tuple[bool, str]:
        """
        Check for patterns that must never appear in agent-submitted code:
        - Re-instantiating CMMCorePlus / UniMMCore (use the pre-configured mmc)
        - Calling CMMCorePlus.instance() / UniMMCore.instance()
        - Calling loadSystemConfiguration() / loadConfig() (config is managed by the toolset)
        Returns (True, "") if safe, (False, reason) if blocked.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True, ""  # syntax errors are reported later

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func

            # Block CMMCorePlus() / UniMMCore() direct construction
            if isinstance(func, ast.Name) and func.id in self._BLOCKED_CONSTRUCTORS:
                return False, (
                    f"Instantiating {func.id}() is not allowed inside agent code. "
                    "Use the pre-configured `mmc` instance instead."
                )

            if isinstance(func, ast.Attribute):
                # Block CMMCorePlus.instance() / UniMMCore.instance()
                if (func.attr == "instance"
                        and isinstance(func.value, ast.Name)
                        and func.value.id in self._BLOCKED_CONSTRUCTORS):
                    return False, (
                        f"Calling {func.value.id}.instance() is not allowed. "
                        "Use the pre-configured `mmc` instance instead."
                    )
                # Block .loadSystemConfiguration() / .loadConfig()
                if func.attr in self._BLOCKED_METHODS:
                    return False, (
                        f"Calling .{func.attr}() is not allowed inside agent code. "
                        "Hardware configuration is managed by the toolset."
                    )

        return True, ""

