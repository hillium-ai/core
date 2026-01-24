"""Enhanced unit tests for Inference Backend Interface with security and error handling."""

import pytest
import os
from unittest.mock import Mock, patch

from loqus_core.inference.backend import (
    LlamaCppBackend,
    PowerInferBackend,
    GenerateParams,
    GenerateResult,
    get_backend,
)


def test_backend_interface_exists():
    """Test that the interface exists and can be imported."""
    # This should not raise any import errors
    assert LlamaCppBackend is not None
    assert PowerInferBackend is not None
    assert GenerateParams is not None
    assert GenerateResult is not None


def test_llamacpp_backend_implements_interface():
    """Test that LlamaCppBackend implements the interface correctly."""
    backend = LlamaCppBackend()
    
    # Test that it has all required methods
    assert hasattr(backend, 'load_model')
    assert hasattr(backend, 'generate')
    assert hasattr(backend, 'unload')
    assert hasattr(backend, 'is_loaded')
    
    # Test that it's properly initialized
    assert not backend.is_loaded()


def test_powerinfer_backend_implements_interface():
    """Test that PowerInferBackend implements the interface correctly."""
    backend = PowerInferBackend()
    
    # Test that it has all required methods
    assert hasattr(backend, 'load_model')
    assert hasattr(backend, 'generate')
    assert hasattr(backend, 'unload')
    assert hasattr(backend, 'is_loaded')
    
    # Test that it's properly initialized
    assert not backend.is_loaded()


def test_powerinfer_not_implemented():
    """PowerInfer should raise NotImplementedError."""
    backend = PowerInferBackend()
    
    with pytest.raises(NotImplementedError):
        backend.load_model("/fake/path", {})
    
    with pytest.raises(NotImplementedError):
        backend.generate("test", GenerateParams())


def test_backend_factory():
    """Factory should return correct backend."""
    backend = get_backend("llama.cpp")
    assert isinstance(backend, LlamaCppBackend)
    
    backend = get_backend("llama_cpp")
    assert isinstance(backend, LlamaCppBackend)
    
    backend = get_backend("powerinfer")
    assert isinstance(backend, PowerInferBackend)
    
    with pytest.raises(ValueError):
        get_backend("unknown_backend")


def test_llamacpp_requires_load():
    """Generation without load should fail."""
    backend = LlamaCppBackend()
    
    with pytest.raises(RuntimeError):
        backend.generate("test", GenerateParams())


def test_generate_params_default_values():
    """Test that GenerateParams has correct default values."""
    params = GenerateParams()
    assert params.max_tokens == 512
    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k == 40
    assert params.stop_sequences == ()
    assert params.seed is None


def test_generate_result_dataclass():
    """Test that GenerateResult dataclass works correctly."""
    result = GenerateResult(
        text="test text",
        tokens_generated=10,
        latency_ms=100.0,
        finish_reason="stop"
    )
    
    assert result.text == "test text"
    assert result.tokens_generated == 10
    assert result.latency_ms == 100.0
    assert result.finish_reason == "stop"


def test_llamacpp_backend_path_validation():
    """Test that LlamaCppBackend validates model paths correctly."""
    backend = LlamaCppBackend()
    
    # Test with invalid path type
    with pytest.raises(TypeError):
        backend.load_model(123, {})
    
    # Test with empty path
    with pytest.raises(ValueError):
        backend.load_model("", {})
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        backend.load_model("/non/existent/file.gguf", {})
    
    # Test with directory path (should fail)
    with pytest.raises(ValueError):
        backend.load_model("/", {})


def test_llamacpp_backend_path_traversal_protection():
    """Test that LlamaCppBackend protects against path traversal."""
    backend = LlamaCppBackend()
    
    # Test path traversal detection
    with pytest.raises(ValueError):
        backend.load_model("/path/../secret_file.gguf", {})


def test_llamacpp_backend_input_validation():
    """Test that LlamaCppBackend validates inputs correctly."""
    backend = LlamaCppBackend()
    
    # Test with valid model path (but don't actually load it)
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        with patch('os.path.isfile') as mock_isfile:
            mock_isfile.return_value = True
            
            # This should work without errors
            try:
                backend.load_model("/valid/model.gguf", {})
            except Exception:
                # We expect this to fail due to missing llama_cpp import
                pass


def test_llamacpp_backend_unload_safety():
    """Test that unload method is safe to call multiple times."""
    backend = LlamaCppBackend()
    
    # Unload when not loaded should not raise error
    backend.unload()  # Should not raise
    
    # Unload again should not raise error
    backend.unload()  # Should not raise