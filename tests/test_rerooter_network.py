import torch
import unittest
from loqus_core.motor.rerooter_network import RerooterNetwork


class TestRerooterNetwork(unittest.TestCase):
    def setUp(self):
        self.network = RerooterNetwork(map_size=32, hidden_dim=128)
        
    def test_network_initialization(self):
        """Test that the network initializes correctly"""
        self.assertIsInstance(self.network, torch.nn.Module)
        self.assertEqual(self.network.map_size, 32)
        self.assertEqual(self.network.hidden_dim, 128)
        
    def test_forward_pass(self):
        """Test the forward pass with valid inputs"""
        # Create test inputs
        start = torch.randn(1, 2)
        goal = torch.randn(1, 2)
        local_map = torch.randn(1, 1, 32, 32)
        
        # Forward pass
        policy_logits, value = self.network(start, goal, local_map)
        
        # Check output shapes
        self.assertEqual(policy_logits.shape, (1, 8))
        self.assertEqual(value.shape, (1, 1))
        
    def test_forward_scriptable(self):
        """Test the forward_scriptable method"""
        start = torch.randn(1, 2)
        goal = torch.randn(1, 2)
        local_map = torch.randn(1, 1, 32, 32)
        
        # Test scriptable forward
        policy_logits, value = self.network.forward_scriptable(start, goal, local_map)
        
        self.assertEqual(policy_logits.shape, (1, 8))
        self.assertEqual(value.shape, (1, 1))
        
    def test_export_to_torchscript(self):
        """Test exporting to TorchScript"""
        # Test export
        try:
            scripted = self.network.export_to_torchscript("test_export.pt")
            self.assertIsNotNone(scripted)
        except Exception as e:
            self.fail(f"Export failed with error: {e}")
            
    def test_batch_forward_pass(self):
        """Test forward pass with batched inputs"""
        batch_size = 4
        start = torch.randn(batch_size, 2)
        goal = torch.randn(batch_size, 2)
        local_map = torch.randn(batch_size, 1, 32, 32)
        
        policy_logits, value = self.network(start, goal, local_map)
        
        self.assertEqual(policy_logits.shape, (batch_size, 8))
        self.assertEqual(value.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()
