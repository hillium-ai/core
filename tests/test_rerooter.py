import torch
from loqus_core.motor.rerooter import RerooterNetwork

def test_rerooter_network_creation():
    """Test that RerooterNetwork can be instantiated"""
    network = RerooterNetwork()
    assert network is not None
    print("✅ RerooterNetwork instantiation test passed")


def test_rerooter_network_forward_pass():
    """Test that RerooterNetwork can run a forward pass"""
    network = RerooterNetwork()
    
    # Create dummy inputs
    batch_size = 1
    start = torch.randn(batch_size, 2)
    goal = torch.randn(batch_size, 2)
    local_map = torch.randn(batch_size, 1, 32, 32)
    
    # Run forward pass
    policy_logits, value = network(start, goal, local_map)
    
    # Check output shapes
    assert policy_logits.shape == (batch_size, 8)
    assert value.shape == (batch_size, 1)
    print("✅ RerooterNetwork forward pass test passed")


def test_rerooter_network_export():
    """Test that RerooterNetwork can export to TorchScript"""
    network = RerooterNetwork()
    
    # Export to TorchScript
    try:
        network.export_to_torchscript('test_export.pt')
        print("✅ RerooterNetwork export test passed")
    except Exception as e:
        print(f"❌ RerooterNetwork export failed: {e}")
        raise


def test_rerooter_network_torchscript_loading():
    """Test that exported TorchScript model can be loaded"""
    # First export
    network = RerooterNetwork()
    network.export_to_torchscript('test_export.pt')
    
    # Try to load it back
    try:
        loaded_model = torch.jit.load('test_export.pt')
        print("✅ RerooterNetwork TorchScript loading test passed")
    except Exception as e:
        print(f"❌ RerooterNetwork TorchScript loading failed: {e}")
        raise

if __name__ == "__main__":
    test_rerooter_network_creation()
    test_rerooter_network_forward_pass()
    test_rerooter_network_export()
    test_rerooter_network_torchscript_loading()
    print("All tests passed!")