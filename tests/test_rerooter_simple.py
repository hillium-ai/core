import torch
from loqus_core.motor.rerooter import RerooterNetwork


def test_simple_forward_pass():
    # Create a simple network
    network = RerooterNetwork()
    
    # Create dummy inputs
    start = torch.tensor([0.5, 0.5])
    goal = torch.tensor([0.8, 0.2])
    local_map = torch.rand(64, 64)
    
    # Run forward pass
    policy_logits, value = network(start, goal, local_map)
    
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Check output shapes
    assert policy_logits.shape == (8,)
    assert value.shape == (1,)
    
    print("Simple test passed!")

if __name__ == "__main__":
    test_simple_forward_pass()
