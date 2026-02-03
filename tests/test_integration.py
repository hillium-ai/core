import torch
from loqus_core.motor.rerooter import RerooterNetwork


def test_integration():
    # Test that we can create and use the network
    network = RerooterNetwork()
    
    # Test with dummy data
    start = torch.tensor([0.5, 0.5])
    goal = torch.tensor([0.8, 0.2])
    local_map = torch.rand(64, 64)
    
    # Forward pass
    policy_logits, value = network(start, goal, local_map)
    
    # Check that outputs are reasonable
    assert policy_logits.shape == (8,)
    assert value.shape == (1,)
    
    # Test JIT export
    scripted = torch.jit.script(network)
    policy_logits2, value2 = scripted(start, goal, local_map)
    
    assert policy_logits2.shape == (8,)
    assert value2.shape == (1,)
    
    print("Integration test passed!")

if __name__ == "__main__":
    test_integration()
