import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from loqus_core.motor.rerooter_network import RerooterNetwork

def test_rerooter_network():
    """Test that RerooterNetwork can be instantiated and run"""
    # Create network instance
    network = RerooterNetwork(map_size=32, hidden_dim=128)
    
    # Create dummy inputs
    batch_size = 1
    start = torch.randn(batch_size, 2)
    goal = torch.randn(batch_size, 2)
    local_map = torch.randn(batch_size, 1, 32, 32)
    
    # Run forward pass
    policy_logits, value = network(start, goal, local_map)
    
    # Check output shapes
    assert policy_logits.shape == (batch_size, 8), f"Expected policy_logits shape (1, 8), got {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Expected value shape (1, 1), got {value.shape}"
    
    # Test forward_scriptable method
    policy_logits_script, value_script = network.forward_scriptable(start, goal, local_map)
    
    assert policy_logits_script.shape == (batch_size, 8), f"Expected scripted policy_logits shape (1, 8), got {policy_logits_script.shape}"
    assert value_script.shape == (batch_size, 1), f"Expected scripted value shape (1, 1), got {value_script.shape}"
    
    print("RerooterNetwork test passed!")
    
    # Export to TorchScript
    scripted_network = torch.jit.script(network)
    print("TorchScript export successful!")
    
if __name__ == "__main__":
    test_rerooter_network()