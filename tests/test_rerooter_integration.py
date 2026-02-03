import torch
import unittest
from loqus_core.motor.rerooter_network import RerooterNetwork


class TestRerooterIntegration(unittest.TestCase):
    def setUp(self):
        self.network = RerooterNetwork(map_size=32, hidden_dim=128)
        
    def test_complete_workflow(self):
        """Test complete workflow from initialization to export"""
        # Test initialization
        self.assertIsInstance(self.network, torch.nn.Module)
        
        # Test forward pass
        start = torch.randn(1, 2)
        goal = torch.randn(1, 2)
        local_map = torch.randn(1, 1, 32, 32)
        
        policy_logits, value = self.network(start, goal, local_map)
        
        # Verify outputs
        self.assertEqual(policy_logits.shape, (1, 8))
        self.assertEqual(value.shape, (1, 1))
        
        # Test scriptable forward
        policy_logits_script, value_script = self.network.forward_scriptable(start, goal, local_map)
        self.assertEqual(policy_logits_script.shape, (1, 8))
        self.assertEqual(value_script.shape, (1, 1))
        
        # Test export functionality
        try:
            self.network.export_to_torchscript("test_rerooter_export.pt")
            # Verify file was created
            import os
            self.assertTrue(os.path.exists("test_rerooter_export.pt"))
            # Clean up
            os.remove("test_rerooter_export.pt")
        except Exception as e:
            self.fail(f"Export test failed: {e}")
            
    def test_batch_processing(self):
        """Test processing multiple inputs at once"""
        batch_size = 4
        start = torch.randn(batch_size, 2)
        goal = torch.randn(batch_size, 2)
        local_map = torch.randn(batch_size, 1, 32, 32)
        
        policy_logits, value = self.network(start, goal, local_map)
        
        self.assertEqual(policy_logits.shape, (batch_size, 8))
        self.assertEqual(value.shape, (batch_size, 1))
        
    def test_input_validation(self):
        """Test that inputs are properly validated"""
        start = torch.randn(1, 2)
        goal = torch.randn(1, 2)
        local_map = torch.randn(1, 1, 32, 32)
        
        # Test normal case
        policy_logits, value = self.network(start, goal, local_map)
        self.assertEqual(policy_logits.shape, (1, 8))
        
        # Test with different map sizes
        local_map_large = torch.randn(1, 1, 64, 64)
        policy_logits_large, value_large = self.network(start, goal, local_map_large)
        self.assertEqual(policy_logits_large.shape, (1, 8))


if __name__ == '__main__':
    unittest.main()
