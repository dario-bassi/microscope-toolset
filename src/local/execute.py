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

    

    def run_code_new(self, code: str, execution_mode: str = "buffered"):
        """Execute code after pre-importing deps. Use GatekeeperCore to buffer hardware calls and commit on succession"""

        # validate mode
        if execution_mode not in ["buffered", "live"]:
            return f"Invalid execution mode: {execution_mode}. Must be 'buffered' or 'live'. "

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


    def run_code_old(self, code: str):
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

