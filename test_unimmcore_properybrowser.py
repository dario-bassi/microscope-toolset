"""Test script to debug PropertyBrowser with UniMMCore."""

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.experimental.unicore import UniMMCore
from pymmcore_widgets import PropertyBrowser
from qtpy.QtWidgets import QApplication

def test_with_cmm_core():
    """Test PropertyBrowser with CMMCorePlus."""
    print("\n=== Testing with CMMCorePlus ===")
    mmc = CMMCorePlus.instance()
    mmc.loadSystemConfiguration()
    
    # Check what iterProperties returns
    props = list(mmc.iterProperties(as_object=True))
    print(f"Number of properties found by iterProperties(): {len(props)}")
    if props:
        print(f"First few properties: {[f'{p.device}-{p.name}' for p in props[:3]]}")
    
    # Check loaded devices
    devices = mmc.getLoadedDevices()
    print(f"Number of loaded devices: {len(devices)}")
    print(f"Loaded devices: {devices}")

def test_with_uni_mmcore():
    """Test PropertyBrowser with UniMMCore."""
    print("\n=== Testing with UniMMCore ===")
    mmc = UniMMCore()
    mmc.loadSystemConfiguration()
    
    # Check what iterProperties returns
    props = list(mmc.iterProperties(as_object=True))
    print(f"Number of properties found by iterProperties(): {len(props)}")
    if props:
        print(f"First few properties: {[f'{p.device}-{p.name}' for p in props[:3]]}")
    
    # Check loaded devices
    devices = mmc.getLoadedDevices()
    print(f"Number of loaded devices: {len(devices)}")
    print(f"Loaded devices: {devices}")

if __name__ == "__main__":
    try:
        test_with_cmm_core()
    except Exception as e:
        print(f"Error with CMMCorePlus: {e}")
    
    try:
        test_with_uni_mmcore()
    except Exception as e:
        print(f"Error with UniMMCore: {e}")
    
    # Now test PropertyBrowser with both
    app = QApplication([])
    
    print("\n=== Testing PropertyBrowser with CMMCorePlus ===")
    try:
        mmc_cmm = CMMCorePlus.instance()
        mmc_cmm.loadSystemConfiguration()
        pb_cmm = PropertyBrowser(mmcore=mmc_cmm)
        print(f"PropertyBrowser with CMMCorePlus - table rows: {pb_cmm._prop_table.rowCount()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Testing PropertyBrowser with UniMMCore ===")
    try:
        mmc_uni = UniMMCore()
        mmc_uni.loadSystemConfiguration()
        pb_uni = PropertyBrowser(mmcore=mmc_uni)
        print(f"PropertyBrowser with UniMMCore - table rows: {pb_uni._prop_table.rowCount()}")
    except Exception as e:
        print(f"Error: {e}")
