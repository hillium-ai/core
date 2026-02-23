import pytest

def test_vr_bridge_import():
    """Test that VR bridge can be imported"""
    try:
        import hillium_vr
        assert hillium_vr is not None
        print("VR Bridge imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import hillium_vr: {e}")


def test_vr_bridge_creation():
    """Test that VR bridge can be created"""
    try:
        import hillium_vr
        bridge = hillium_vr.VrBridge()
        assert bridge is not None
        print("VR Bridge created successfully")
    except Exception as e:
        pytest.fail(f"Failed to create VR Bridge: {e}")