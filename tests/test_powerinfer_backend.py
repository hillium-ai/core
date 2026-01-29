import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loqus_core.inference.backend import PowerInferBackend
from loqus_core.inference.backend import GenerateParams, GenerateResult


class TestPowerInferBackend:
    """
    Test cases for PowerInferBackend.
    """
    
    def test_backend_initialization(self):
        """
        Test that PowerInferBackend can be initialized.
        """
        backend = PowerInferBackend()
        assert backend is not None
        
    def test_backend_is_loaded(self):
        """
        Test that backend reports correctly when not loaded.
        """
        backend = PowerInferBackend()
        assert backend.is_loaded() is False
        
    def test_backend_load_model_raises_runtime_error(self):
        """
        Test that load_model raises RuntimeError when library is not available.
        """
        # Mock the powerinfer_lib to None to simulate library not being available
        with patch('loqus_core.inference.powerinfer_backend.powerinfer_lib', None):
            backend = PowerInferBackend()
            
            with pytest.raises(RuntimeError) as exc_info:
                backend.load_model("/fake/path", {})
            
            assert "PowerInfer library not available" in str(exc_info.value)
        
    def test_backend_generate_raises_runtime_error(self):
        """
        Test that generate raises RuntimeError when library is not available.
        """
        # Mock the powerinfer_lib to None to simulate library not being available
        with patch('loqus_core.inference.powerinfer_backend.powerinfer_lib', None):
            backend = PowerInferBackend()
            
            with pytest.raises(RuntimeError) as exc_info:
                backend.generate("test prompt", GenerateParams())
            
            assert "PowerInfer library not available" in str(exc_info.value)
        
    def test_backend_unload_works(self):
        """
        Test that unload works without errors.
        """
        backend = PowerInferBackend()
        # Should not raise any exception
        backend.unload()
        assert backend.is_loaded() is False
        
    def test_backend_interface_compliance(self):
        """
        Test that PowerInferBackend implements the interface correctly.
        """
        backend = PowerInferBackend()
        
        # Test that it has all required methods
        assert hasattr(backend, 'load_model')
        assert hasattr(backend, 'generate')
        assert hasattr(backend, 'unload')
        assert hasattr(backend, 'is_loaded')
        
        # Test that methods are callable
        assert callable(backend.load_model)
        assert callable(backend.generate)
        assert callable(backend.unload)
        assert callable(backend.is_loaded)
        
    def test_backend_factory_returns_powerinfer(self):
        """
        Test that the factory returns PowerInferBackend for powerinfer backend.
        """
        from loqus_core.inference.backend import get_backend
        
        backend = get_backend("powerinfer")
        assert isinstance(backend, PowerInferBackend)
        
    def test_backend_factory_handles_invalid_backend(self):
        """
        Test that the factory raises ValueError for invalid backend.
        """
        from loqus_core.inference.backend import get_backend
        
        with pytest.raises(ValueError):
            get_backend("invalid_backend")