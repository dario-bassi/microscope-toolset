import importlib
import importlib.util
import subprocess
import sys
from io import StringIO
from contextlib import redirect_stdout,redirect_stderr

from pymmcore_plus.experimental.unicore import UniMMCore

from src.virtual_microscope.initialize_virtual_microscope import initialize_virtual_microscope, initialize_virtual_microscope_from_configuration

from pymmcore_plus import CMMCorePlus
import logging
import ast

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

    def __init__(self, filename: str, mmc: CMMCorePlus | UniMMCore = None, microscope_type: str = "real"):
        self.namespace = {}
        
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
            if filename is None:
                initialize_virtual_microscope(core=mmc)
            elif isinstance(filename, str) and filename != "":
                initialize_virtual_microscope_from_configuration(core=mmc)
                mmc.loadSystemConfiguration(fileName=filename)
            else:
                raise ValueError(f"The file configuration {filename} doesn't exists. Please checks the name.")

            logger.info("mmc instance is loaded into the namespace")

        logger.info(f"Execute initialized for {microscope_type} microscope")


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

        # Check code before running it
        if not self.is_safe_viewer(code):
            return "viewer"
        
        
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
    

    def is_safe_viewer(self, code: str):
        """
        Checks if in the code there is viewer
        """
        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Check if code contains 'viewer'
            if isinstance(node, ast.Name) and node.id == 'viewer':
                return False
            
        return True

