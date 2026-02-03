from pymmcore_plus import CMMCorePlus

from src.virtual_microscope.microscope_sim import MicroscopeSim
from src.virtual_microscope.microscope_sim_optimized import MicroscopeSimOptmized
from src.virtual_microscope.pymmcore_camera_sim import SimCameraDevice
from src.virtual_microscope.pymmcore_stage_sim import SimStageDevice
from src.virtual_microscope.pymmcore_state_device_sim import LEDDevice, FilterWheelDevice, ObjectiveDevice
from src.virtual_microscope.pymmcore_shutter_sim import SimShutterDevice
from src.virtual_microscope.pymmcore_slm_sim import SimSLMDevice
import src.virtual_microscope.simulation_bridge as bridge_module
from src.virtual_microscope.simulation_bridge import SimulationBridge
from src.virtual_microscope.pymmcore_z_stage_sim import SimZStageDevice
from pymmcore_plus.experimental.unicore.core._unicore import UniMMCore
import logging
import sys

#  logger
logger = logging.getLogger("Execute")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("microscope_toolset.log", encoding="utf-8")
fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(fh)



def initialize_virtual_microscope(core: UniMMCore) -> None:

    try:
        # access UniMMCore
        # development python devices
        logger.info("Initialized UniMMCore")
        #microscope_simulation = MicroscopeSim() # old simulation to uncomment!
        #microscope_simulation = MicroscopeSimOptmized(cell_type="normal", nb_cells=100)
        microscope_simulation = MicroscopeSimOptmized(cell_type='cycle', nb_cells=50)
        # Initialize global Singleton
        bridge_module.GLOBAL_BRIDGE = SimulationBridge(microscope_simulation)
        logger.info("Initialized MicroscopeSim")

        # -----------------------------------------
        # unload all devices
        core.unloadAllDevices()
        logger.info("Unloaded AllDevices")
        # -----------------------------------------------------
        # load device

        

        #core.loadPyDevice("Camera", SimCameraDevice(core=core, microscope_sim=microscope_simulation))
        core.loadPyDevice("Camera", SimCameraDevice())
        logger.info("Load Camera Device")
        #core.loadPyDevice("XYStage", SimStageDevice(microscope_sim=microscope_simulation))
        core.loadPyDevice("XYStage", SimStageDevice())
        logger.info("Load XYStage Device")
        core.loadPyDevice("ZStage", SimZStageDevice())
        logger.info("Load ZStage Device")
        #core.loadPyDevice("LED", SimStateDevice(label="LED",
        #                                        state_dict={0: "UV", 1: "BLUE", 2: "CYAN", 3: "GREEN", 4: "YELLOW",
        #                                                    5: "ORANGE", 6: "RED"},
        #                                        microscope_sim=microscope_simulation))
        core.loadPyDevice("LED", LEDDevice())
        logger.info("Load LED Device")
        #core.loadPyDevice("Filter Wheel", SimStateDevice(label="Filter Wheel",
        #                                                 state_dict={0: "Electra1(402/454)", 1: "SCFP2(434/474)",
        #                                                             2: "TagGFP2(483/506)", 3: "obeYFP(514/528)",
        #                                                             5: "mRFP1-Q667(549/570)", 6: "mScarlet3(569/582)",
        #                                                             7: "miRFP670(642/670)"},
        #                                                 microscope_sim=microscope_simulation))
        logger.info("Load Filter Wheel Device")
        core.loadPyDevice("Filter Wheel", FilterWheelDevice())
        core.loadPyDevice("Shutter", SimShutterDevice())
        logger.info("Shutter Device")
        core.loadPyDevice("Objective", ObjectiveDevice())
        logger.info("Load Objective Device")
        core.loadPyDevice("SLM", SimSLMDevice())
        logger.info("Load SLM Device")
        # ------------------------------------
        # initialize device
        core.initializeDevice("Camera")
        core.initializeDevice("XYStage")
        core.initializeDevice("Filter Wheel")
        core.initializeDevice("LED")
        core.initializeDevice("Shutter")
        core.initializeDevice("ZStage")
        core.initializeDevice("Objective")
        core.initializeDevice("SLM")
        logger.info("Initialized Device")

        # ---------------------------
        # Set initial value of some device
        # Set initial value of some device
        core.setCameraDevice("Camera")
        core.setXYStageDevice("XYStage")
        core.setFocusDevice("ZStage")
        core.setShutterDevice("Shutter")
        core.setState("LED", 0)
        core.setState("Filter Wheel", 0)
        core.setState("Objective", 0)
        core.setSLMDevice("SLM")

        # define configuration groups of the microscope
        # Once integrated, this part will be forwarded to the configuration file (.cfg)!
        core.defineConfigGroup("Fake")
        core.defineConfigGroup("Real")
        core.defineConfig("Fake", "nucleus-channel", "LED", "Label", "ORANGE")
        core.defineConfig("Fake", "nucleus-channel", "Filter Wheel", "Label", "mScarlet3(569/582)")
        core.defineConfig("Fake", "membrane-channel", "LED", "Label", "RED")
        core.defineConfig("Fake", "membrane-channel", "Filter Wheel", "Label", "miRFP670(642/670)")
        core.defineConfig("Fake", "all-channel", "LED", "Label", "CYAN")
        core.defineConfig("Fake", "all-channel", "Filter Wheel", "Label", "Electra1(402/454)")
        core.defineConfig("Real", "membrane-channel", "LED", "Label", "RED")
        core.defineConfig("Real", "membrane-channel", "Filter Wheel", "Label", "miRFP670(642/670)")

        core.setConfig("Fake", "all-channel")

        logger.info("Initialization Complete")

    except Exception as e:
        logger.error(f"Error: {e}")
        RuntimeError(f"Error: {e}")


def initialize_virtual_microscope_from_configuration(core: UniMMCore) -> None:

    try:
        # access UniMMCore
        # development python devices
        logger.info("Initialized UniMMCore")
        #microscope_simulation = MicroscopeSim() # old simulation to uncomment!
        #microscope_simulation = MicroscopeSimOptmized(cell_type="normal", nb_cells=100)
        microscope_simulation = MicroscopeSimOptmized(cell_type="cycle", nb_cells=10)
        # Initialize global Singleton
        bridge_module.GLOBAL_BRIDGE = SimulationBridge(microscope_simulation)
        logger.info("Initialized MicroscopeSim")

        logger.info("Initialization Complete")

    except Exception as e:
        logger.error(f"Error: {e}")
        RuntimeError(f"Error: {e}")