import importlib
import importlib.util
import subprocess
import sys
from io import StringIO
from contextlib import redirect_stdout,redirect_stderr

from pymmcore_plus.experimental.unicore import UniMMCore

from src.virtual_microscope.initialize_virtual_microscope import initialize_virtual_microscope

from pymmcore_plus import CMMCorePlus
import logging

#  logger
logger = logging.getLogger("Execute")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(fh)


class Execute:

    def __init__(self, filename: str, mmc: CMMCorePlus | UniMMCore = None, microscope_type: str = "real", viewer=None):
        self.namespace = {}
        #self.namespace["mmc"] = mmc
        #logger.info("mmc instance is loaded into the namespace")
        if microscope_type == "real":
            if mmc is not None:
                self.namespace["mmc"] = mmc
                logger.info("mmc instance is loaded into the namespace")
                # Load config file
                mmc.loadSystemConfiguration(fileName=filename)
                logger.info("configuration file of the microscope was loaded")
            else:
                self.namespace["mmc"] = CMMCorePlus().instance()
                exec(f"mmc.loadSystemConfiguration(fileName='{filename}')", self.namespace)
        elif microscope_type == "virtual":
            logger.info("Initializing virtual microscope...")
            self.namespace["mmc"] = mmc
            initialize_virtual_microscope(core=mmc)

            logger.info("mmc instance is loaded into the namespace")

        # IMPORTANT: Do NOT add viewer to namespace!
        # Viewer methods must be called via viewer tools (viewer_screenshot, etc.) which use
        # the thread-safe proxy. Direct viewer calls from executed code will cause Qt/OpenGL
        # threading errors since this code runs in a daemon thread.
        # if viewer is not None:
        #     self.namespace["viewer"] = viewer
        #     logger.info("viewer instance is loaded into the namespace")

        logger.info(f"Execute initialized for {microscope_type} microscope")

    def set_viewer(self, viewer):
        """
        DEPRECATED: Do not add viewer to execution namespace.
        Viewer operations must be performed through thread-safe viewer tools
        (viewer_screenshot, viewer_add_labels, etc.) which use the proxy pattern
        to execute on the main Qt thread. Direct viewer access causes threading errors.
        """
        logger.warning("set_viewer() called but viewer is not added to namespace to prevent threading issues")
        pass
        return self


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

    def run_code(self, code: str):
        """Execute code with better error handling and output capture"""
        max_attempts = 3  # Prevent infinite loops
        attempts = 0
        read_output = ""

        while attempts < max_attempts:
            attempts += 1
            try:
                f = StringIO()
                with redirect_stdout(f):
                    # Also capture stderr
                    err_f = StringIO()
                    with redirect_stderr(err_f):
                        exec(code, self.namespace)

                read_output = f.getvalue().strip()
                stderr_output = err_f.getvalue().strip()

                if stderr_output:
                    read_output += f"\nWarnings/Errors: {stderr_output}"

                logger.info("Code executed successfully")
                return read_output if read_output else "Code executed successfully (no output)"

            except ModuleNotFoundError as e:
                module_name = str(e).split("'")[1] if "'" in str(e) else str(e)
                logger.info(f"Attempting to import missing module: {module_name}")

                try:
                    self.namespace[module_name] = importlib.import_module(module_name)
                    continue  # Retry execution
                except ImportError:
                    # Module not available, try to install
                    if self._install_library(module_name):
                        self.namespace[module_name] = importlib.import_module(module_name)
                        continue
                    else:
                        return f"Could not install required module: {module_name}"

            except ImportError as e:
                import_module = str(e).split("'")[1] if "'" in str(e) else str(e)
                logger.info(f"Attempting to install missing package: {import_module}")

                if self._install_library(import_module):
                    continue
                else:
                    return f"Could not install the module {import_module}"

            except Exception as e:
                error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return error_msg

        return f"Code execution failed after {max_attempts} attempts"
