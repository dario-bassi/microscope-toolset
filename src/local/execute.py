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

from src.local.gatekeeper_core import GatekeeperCore

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
                # Load config file
                mmc.loadSystemConfiguration(fileName=filename)
                logger.info("configuration file of the microscope was loaded")
                self.namespace["mmc"] = GatekeeperCore(mmc)#mmc
                logger.info("mmc instance is loaded into the namespace")
            else:
                real_mmc = CMMCorePlus().instance()
                real_mmc.loadSystemConfiguration(fileName=filename)
                self.namespace["mmc"] = GatekeeperCore(real_mmc)#CMMCorePlus().instance()
                #exec(f"mmc.loadSystemConfiguration(fileName='{filename}')", self.namespace)
        elif microscope_type == "virtual":
            logger.info("Initializing virtual microscope...")          
            if filename is None:
                initialize_virtual_microscope(core=mmc)
            elif isinstance(filename, str) and filename != "":
                initialize_virtual_microscope_from_configuration(core=mmc)
                mmc.loadSystemConfiguration(fileName=filename)
            else:
                raise ValueError(f"The file configuration {filename} doesn't exists. Please checks the name.")

            logger.info("mmc instance is loaded into the namespace")

            self.namespace["mmc"] = GatekeeperCore(mmc)#mmc

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
        
    def _preimport_dependencies(self, code: str):
        """Parse AST for import an ensure modules are available (install if needed)."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    mod = n.name.split('.')[0]
                    if importlib.util.find_spec(mod) is None:
                        if not self._install_library(mod):
                            raise ModuleNotFoundError(mod)
                    importlib.import_module(mod)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split('.')[0]
                if importlib.util.find_spec(mod) is None:
                    if not self._install_library(mod):
                        raise ModuleNotFoundError(node)
                importlib.import_module(mod)

    
    def _find_mmc_calls(self, code: str):
        """Return list of method names called on the 'mmc' name in the code string."""
        tree = ast.parse(code)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'mmc':
                    calls.append(func.attr)

        return calls
    


    def _classify_mmc_method(self, method_name: str) -> str:
        """
        Clasify pymmcore-plus methods
        - 'allowed' : safe getters/queries
         - 'buffer'  : idempotent setters / moves (buffer & replay with predicate)
         - 'special' : acquisitions / non-idempotent (snap/sequence) -> cache / require explicit commit
        Conservative default is 'buffer'.
        """
        if method_name.startswith(("get", "is")):
            return "allowed"
        special = {
            "snapImage", "getImage", "startSequenceAcquisition", "stopSequenceAcquisition",
            "snap", "getLastImage", "startContinuousSequenceAcquisition"
        }
        if method_name in special:
            return "special"
        
        if method_name.startswith(("set", "load", "enable", "setPosition", "setXYPosition", "setZPosition")):
            return "buffer"
        
        return "buffer"
    

    def run_code_new(self, code: str):
        """Execute code after pre-importing deps. Use GatekeeperCore to buffer hardware calls and commit on succession"""

        # Check code before running it
        if not self.is_safe_viewer(code):
            return "viewer"
        
        # Ensure dependencies present
        try:
            self._preimport_dependencies(code)
        except ModuleNotFoundError as e:
            return f"Dependency error: {e}"
        
        except Exception as e:
            return f"Dependency parsing error: {e}"
        
        # Static analysis of mmc usage (convervatives)
        mmc_obj = self.namespace.get("mmc")
        try:
            mmc_calls = self._find_mmc_calls(code)
            classifications = {m: self._classify_mmc_method(m) for m in mmc_calls}
            # If there are special non-idempotent calls, we allow them but they will be cached at commit time.
        except Exception:
            classifications = {}

        # Snapshot state if GatekeeperCore present
        snapshot = None
        if mmc_obj and hasattr(mmc_obj, "snapshot_state"):
            try:
                snapshot = mmc_obj.snapshot_state()
            except Exception:
                snapshot = None

        # Execute user code once
        try:
            out_f = StringIO()
            err_f = StringIO()
            with redirect_stdout(out_f), redirect_stderr(err_f):
                exec(code, self.namespace)
            
            stdout_text = out_f.getvalue().strip()
            stderr_text = err_f.getvalue().strip()
            read_output = stdout_text

            if stderr_text:
                read_output = (read_output + "\nWarnings/Errors: " + stderr_text).strip()

            # Commit buffered mmc calls if wrapper present
            if mmc_obj and hasattr(mmc_obj, "commit"):
                try:
                    real = getattr(mmc_obj, "_mmc", None)
                    mmc_obj.commit(real_mmc=real, check_snapshot=snapshot)
                except Exception as e:
                    return f"Commit failed: {e}"
            logger.info("Code executed successfully")
            return read_output if read_output else "Code executed successfully (no output)"
        except ModuleNotFoundError as e:
            module_name = str(e).split("'")[1] if "'" in str(e) else str(e)
            logger.error(f"Module not found during execution (unexpected): {module_name}")
            return f"Module not found during execution: {module_name}"
        except Exception as e:
            error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return error_msg


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

