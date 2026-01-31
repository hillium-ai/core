import unittest

# Import the PowerInferBackend
from loqus_core.inference import PowerInferBackend, get_backend


class TestPowerInferBackendSimple(unittest.TestCase):
    
    def test_backend_creation(self):
        """Test that PowerInferBackend can be created."""
        backend = PowerInferBackend()
        self.assertIsInstance(backend, PowerInferBackend)
        
    def test_get_backend_factory(self):
        """Test that get_backend returns PowerInferBackend for powerinfer type."""
        backend = get_backend("powerinfer")
        self.assertIsInstance(backend, PowerInferBackend)
        
    def test_basic_methods_exist(self):
        """Test that basic methods exist."""
        backend = PowerInferBackend()
        
        # Check that methods exist
        self.assertTrue(hasattr(backend, 'load_model'))
        self.assertTrue(hasattr(backend, 'generate'))
        self.assertTrue(hasattr(backend, 'unload'))
        self.assertTrue(hasattr(backend, 'is_loaded'))
        
    def test_is_loaded_initially_false(self):
        """Test that backend is not loaded initially."""
        backend = PowerInferBackend()
        self.assertFalse(backend.is_loaded())


if __name__ == '__main__':
    unittest.main()