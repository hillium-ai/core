import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loqus_core.motor.rerooter import RerooterNetwork

def test_rerooter_network():
    # Create model
    model = RerooterNetwork()
    
    # Create dummy inputs
    start = torch.randn(2)
    goal = torch.randn(2)
    local_map = torch.randn(8, 8)  # 8x8 grid
    
    # Test forward pass
    policy_logits, value = model(start, goal, local_map)
    
    # Check shapes
    assert policy_logits.shape == (8,), f"Expected policy_logits shape (8,), got {policy_logits.shape}"
    assert value.shape == (1,), f"Expected value shape (1,), got {value.shape}"
    
    print("RerooterNetwork test passed!")
    

def test_torchscript_export():
    # Test that we can load the exported model
    try:
        model = torch.jit.load("rerooter.pt")
        print("TorchScript model loaded successfully")
        
        # Test with dummy inputs
        start = torch.randn(2)
        goal = torch.randn(2)
        local_map = torch.randn(8, 8)
        
        # Forward pass
        policy_logits, value = model(start, goal, local_map)
        print(f"TorchScript forward pass successful. Policy logits shape: {policy_logits.shape}, Value shape: {value.shape}")
        
    except Exception as e:
        print(f"Error loading TorchScript model: {e}")
        raise

if __name__ == "__main__":
    test_rerooter_network()
    test_torchscript_export()
    print("All tests passed!")