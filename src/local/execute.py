import ast
import importlib
import importlib.util
import logging
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from pymmcore_plus import CMMCorePlus

from .gatekeeper_core import GatekeeperCore
from .mda_helpers import run_mda_with_feedback
from .microscopy_utils import (
    center_on_cell,
    detect_cells,
    find_bright_centroid,
)

#  logger
logger = logging.getLogger("Execute")
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


class Execute:
    # Core classes the agent must not re-instantiate
    _BLOCKED_CONSTRUCTORS = {"CMMCorePlus", "UniMMCore"}
    # Hardware-config methods the agent must not call
    _BLOCKED_METHODS = {"loadSystemConfiguration", "loadConfig"}

    # Registry: top-level library name → list of guard callables (ast.AST → tuple[bool, str])
    # Guards are run only when the corresponding library is imported in the submitted code.
    _LIBRARY_GUARDS: dict[str, list] = {}

    @classmethod
    def register_library_guard(cls, library: str, guard_fn) -> None:
        """Register a library-specific guard. Call this from any module to extend checks."""
        cls._LIBRARY_GUARDS.setdefault(library, []).append(guard_fn)

    def __init__(self, mmc: CMMCorePlus, filename: str | None = None):
        self.namespace = {}

        if filename is not None:
            mmc.loadSystemConfiguration(fileName=filename)
            logger.info("configuration file of the microscope was loaded")

        self._populate_namespace(mmc)
        logger.info("mmc instance is loaded into the namespace")
        logger.info("Execute initialized")

    def _populate_namespace(self, mmc) -> None:
        """Fill (or replace) all mmc-dependent namespace entries."""
        self.namespace["mmc"] = GatekeeperCore(mmc)
        self.namespace["run_mda_with_feedback"] = (
            lambda events, on_frame=None: run_mda_with_feedback(mmc, events, on_frame)
        )
        self.namespace["center_on_cell"] = lambda **kw: center_on_cell(mmc, **kw)
        self.namespace["find_bright_centroid"] = find_bright_centroid
        self.namespace["detect_cells"] = detect_cells

    def _refresh_workspace(self) -> None:
        """Inject workspace_dir into the namespace from the active experiment marker.

        Called at the start of every run_code_new() so the variable is always
        current — even if the experiment was started after the executor was created.
        """
        try:
            import json as _json
            from pathlib import Path as _Path

            from benchmarking.experiment_saver import MARKER_FILE

            if MARKER_FILE.exists():
                _marker = _json.loads(MARKER_FILE.read_text(encoding="utf-8"))
                _ws = _marker.get("workspace_dir")
                self.namespace["workspace_dir"] = _Path(_ws) if _ws else None
            else:
                self.namespace["workspace_dir"] = None
        except Exception:
            self.namespace["workspace_dir"] = None

    def update_core(self, mmc) -> None:
        """Swap the active core in the namespace after a core switch.

        Replaces the GatekeeperCore wrapper and re-binds every lambda that
        captured the old mmc instance in its closure.
        """
        self._populate_namespace(mmc)
        logger.info("Execute namespace updated with new core (%s)", type(mmc).__name__)

    def _install_library(self, module: str):
        """Install missing packages using pip with better error handling"""
        try:
            # First check if module is already available
            spec = importlib.util.find_spec(module)
            if spec is not None:
                return True

            logger.info(f"Installing package: {module}")

            # Install the package
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", module],
                capture_output=True,
                text=True,
                timeout=120,
            )  # Add timeout

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
                    mod = alias.name.split(".")[0]
                    if importlib.util.find_spec(mod) is None and mod not in missing:
                        missing.append(mod)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split(".")[0]
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
                    mod = alias.name.split(".")[0]
                    if mod not in failed:
                        try:
                            importlib.import_module(mod)
                        except Exception:
                            pass
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split(".")[0]
                if mod not in failed:
                    try:
                        importlib.import_module(mod)
                    except Exception:
                        pass
        return failed

    def run_code_new(self, code: str, execution_mode: str = "buffered"):
        """Execute code after pre-importing deps. Use GatekeeperCore to buffer hardware calls and commit on succession"""

        # Refresh workspace_dir from the active experiment marker (if any)
        self._refresh_workspace()

        # validate mode
        if execution_mode not in ["buffered", "live"]:
            return f"Invalid execution mode: {execution_mode}. Must be 'buffered' or 'live'. "

        # Check code before running it
        if not self.is_safe_viewer(code):
            return "viewer"

        safe, reason = self.is_safe_code(code)
        if not safe:
            return f"Safety Error: {reason}"

        ok, reason = self._check_library_guards(code)
        if not ok:
            return f"Library Guard Error: {reason}"

        # Validate and pre-import dependencies
        try:
            failed = self._preimport_dependencies(code)
        except Exception as e:
            return f"Dependency parsing error: {e}"
        if failed:
            return f"Missing packages that could not be installed: {', '.join(failed)}. Install them manually and retry."

        # Execute code in specif mode
        if execution_mode == "live":
            logger.info("Executing code in live mode")
            return self._run_code_live(code)
        else:
            logger.info("Exeecuting code in buffer mode")
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
            teardowns = self._apply_runtime_guards(code)
            try:
                with redirect_stdout(out_f), redirect_stderr(err_f):
                    exec(code, self.namespace)  # nosec B102
            finally:
                self._remove_runtime_guards(teardowns)
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
            teardowns = self._apply_runtime_guards(code)
            try:
                with redirect_stdout(out_f), redirect_stderr(err_f):
                    exec(code, self.namespace)  # nosec B102
            finally:
                self._remove_runtime_guards(teardowns)
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
            if isinstance(node, ast.Name) and node.id == "viewer":
                return False
            # Block napari.current_viewer() pattern
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "current_viewer"
                and isinstance(node.value, ast.Name)
                and node.value.id == "napari"
            ):
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
                if (
                    func.attr == "instance"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in self._BLOCKED_CONSTRUCTORS
                ):
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

    def _get_imported_modules(self, code: str) -> set[str]:
        """Return the set of top-level module names imported anywhere in the code."""
        modules: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return modules
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        return modules

    def _check_library_guards(self, code: str) -> tuple[bool, str]:
        """Run every registered guard for libraries that appear in the code."""
        imported = self._get_imported_modules(code)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True, ""
        for lib, guards in self._LIBRARY_GUARDS.items():
            if lib in imported:
                for guard_fn in guards:
                    ok, reason = guard_fn(tree)
                    if not ok:
                        return False, f"[{lib}] {reason}"
        return True, ""

    # Registry: top-level library name → runtime installer callable (namespace → teardown | None)
    _RUNTIME_GUARDS: dict[str, list] = {}

    @classmethod
    def register_runtime_guard(cls, library: str, installer_fn) -> None:
        """Register a runtime guard installer for a library."""
        cls._RUNTIME_GUARDS.setdefault(library, []).append(installer_fn)

    def _apply_runtime_guards(self, code: str) -> list:
        """Install runtime guards for libraries present in the code. Returns teardown callables."""
        imported = self._get_imported_modules(code)
        teardowns = []
        for lib, installers in self._RUNTIME_GUARDS.items():
            if lib in imported:
                for installer_fn in installers:
                    td = installer_fn(self.namespace)
                    if td is not None:
                        teardowns.append(td)
        return teardowns

    @staticmethod
    def _remove_runtime_guards(teardowns: list) -> None:
        for td in teardowns:
            try:
                td()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Built-in library guards
# ---------------------------------------------------------------------------


def _cellpose_diameter_guard(tree: ast.AST) -> tuple[bool, str]:
    """Require an explicit 'diameter' kwarg in any cellpose call."""
    has_diameter = any(
        kw.arg == "diameter"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
    )
    if not has_diameter:
        return False, (
            "no 'diameter' keyword argument found. "
            "Cellpose defaults to 30 px cell diameter — set it explicitly to match your cells."
        )
    return True, ""


Execute.register_library_guard("cellpose", _cellpose_diameter_guard)


def _cellpose_channels_guard(tree: ast.AST) -> tuple[bool, str]:
    """Require an explicit 'channels' kwarg — default [0,0] fails on multichannel images."""
    has_channels = any(
        kw.arg == "channels"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
    )
    if not has_channels:
        return False, (
            "no 'channels' keyword argument found. "
            "Cellpose defaults to [0, 0] (grayscale) — set channels explicitly, "
            "e.g. [0, 0] for grayscale, [1, 2] for cytoplasm+nucleus."
        )
    return True, ""


def _try_numeric_literal(node: ast.expr) -> float | None:
    """Return the numeric value of a literal node (handles negatives), or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def _cellpose_flow_threshold_guard(tree: ast.AST) -> tuple[bool, str]:
    """Block flow_threshold literals outside [0.0, 3.0]."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "flow_threshold":
                continue
            val = _try_numeric_literal(kw.value)
            if val is not None and not (0.0 <= val <= 3.0):
                return False, (
                    f"flow_threshold={val} is outside the sane range [0.0, 3.0]. "
                    "Values > 1.0 accept noise as cells; values < 0.0 miss real cells."
                )
    return True, ""


def _cellpose_cellprob_threshold_guard(tree: ast.AST) -> tuple[bool, str]:
    """Block cellprob_threshold literals outside [-6.0, 6.0]."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "cellprob_threshold":
                continue
            val = _try_numeric_literal(kw.value)
            if val is not None and not (-6.0 <= val <= 6.0):
                return False, (
                    f"cellprob_threshold={val} is outside the sane range [-6.0, 6.0]. "
                    "Extreme values cause silent over- or under-segmentation."
                )
    return True, ""


Execute.register_library_guard("cellpose", _cellpose_channels_guard)
Execute.register_library_guard("cellpose", _cellpose_flow_threshold_guard)
Execute.register_library_guard("cellpose", _cellpose_cellprob_threshold_guard)


# ---------------------------------------------------------------------------
# Built-in runtime guards
# ---------------------------------------------------------------------------

_CELLPOSE_SIZE_THRESHOLD = 512


def _install_cellpose_size_guard(namespace: dict):
    """
    Monkey-patch Cellpose.eval to check image size at runtime.
    Images larger than 512×512 raise RuntimeError unless the user sets
    cellpose_allow_large_image = True in their code before the eval call.
    Returns a teardown callable that restores the original method.
    """
    try:
        import cellpose.models

        original_eval = cellpose.models.Cellpose.eval
    except (ImportError, AttributeError):
        return None

    def _guarded_eval(self_model, x, *args, **kwargs):
        import numpy as np

        img = np.asarray(x) if not isinstance(x, list) else np.asarray(x[0])
        if img.ndim >= 2:
            h, w = img.shape[0], img.shape[1]
            if (h > _CELLPOSE_SIZE_THRESHOLD or w > _CELLPOSE_SIZE_THRESHOLD) and not namespace.get(
                "cellpose_allow_large_image", False
            ):
                raise RuntimeError(
                    f"Image size {h}×{w} px exceeds the {_CELLPOSE_SIZE_THRESHOLD}×"
                    f"{_CELLPOSE_SIZE_THRESHOLD} px threshold — segmentation may take a very long time.\n"
                    f"  • Resize first, then map masks back to original coordinates:\n"
                    f"      img_small = cv2.resize(img, ({_CELLPOSE_SIZE_THRESHOLD}, {_CELLPOSE_SIZE_THRESHOLD}))\n"
                    f"      masks, flows, _ = model.eval(img_small, ...)\n"
                    f"      masks = cv2.resize(masks, ({w}, {h}), interpolation=cv2.INTER_NEAREST)\n"
                    f"    Use INTER_NEAREST — bilinear/bicubic blends label IDs and corrupts the mask.\n"
                    f"  • Or allow the original size:  set  cellpose_allow_large_image = True  before the eval call."
                )
        return original_eval(self_model, x, *args, **kwargs)

    cellpose.models.Cellpose.eval = _guarded_eval

    def _teardown():
        cellpose.models.Cellpose.eval = original_eval

    return _teardown


Execute.register_runtime_guard("cellpose", _install_cellpose_size_guard)
