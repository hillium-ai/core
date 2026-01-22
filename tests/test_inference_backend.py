import pytest
from unittest.mock import Mock, patch
from loqus_core.inference.backend import (
    InferenceBackend,
    LlamaCppBackend,
    PowerInferBackend,
    GenerateParams
)


def test_inference_backend_interface():
    """Test that InferenceBackend is an abstract base class."""
    # This should raise TypeError because we're trying to instantiate an ABC
    with pytest.raises(TypeError):
        InferenceBackend()


def test_llama_cpp_backend_load_model_success():
    """Test successful model loading with mocked llama_cpp."""
    with patch('llama_cpp.Llama') as mock_llama:
        backend = LlamaCppBackend()
        
        # Mock the llama instance
        mock_instance = Mock()
        mock_llama.return_value = mock_instance
        
        # This should not raise an exception
        backend.load_model("/path/to/model.gguf", {"n_ctx": 2048})
        
        assert backend.is_loaded is True
        mock_llama.assert_called_once_with(
            model_path="/path/to/model.gguf",
            n_ctx=2048,
            n_threads=None,
            n_gpu_layers=0,
            verbose=False
        )


def test_llama_cpp_backend_load_model_invalid_path():
    """Test model loading with invalid path."""
    backend = LlamaCppBackend()
    
    # Test with empty path
    with pytest.raises(ValueError, match="Model path must be a non-empty string"):
        backend.load_model("", {})
    
    # Test with None path
    with pytest.raises(ValueError, match="Model path must be a non-empty string"):
        backend.load_model(None, {})


def test_llama_cpp_backend_generate_success():
    """Test successful generation with mocked llama_cpp."""
    with patch('llama_cpp.Llama') as mock_llama:
        backend = LlamaCppBackend()
        
        # Mock the llama instance and its response
        mock_instance = Mock()
        mock_instance.return_value = {
            "choices": [{"text": "Generated response"}]
        }
        mock_llama.return_value = mock_instance
        
        # Load model first
        backend.load_model("/path/to/model.gguf", {})
        
        # Generate response
        params = GenerateParams(max_tokens=100, temperature=0.7)
        result = backend.generate("Test prompt", params)
        
        assert result == "Generated response"


def test_llama_cpp_backend_generate_not_loaded():
    """Test generation when model is not loaded."""
    backend = LlamaCppBackend()
    
    params = GenerateParams()
    with pytest.raises(RuntimeError, match="Model must be loaded before generation"):
        backend.generate("Test prompt", params)


def test_llama_cpp_backend_generate_invalid_prompt():
    """Test generation with invalid prompt."""
    backend = LlamaCppBackend()
    
    # Load model first
    with patch('llama_cpp.Llama') as mock_llama:
        mock_instance = Mock()
        mock_llama.return_value = mock_instance
        backend.load_model("/path/to/model.gguf", {})
        
        # Test with empty prompt
        params = GenerateParams()
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            backend.generate("", params)
        
        # Test with None prompt
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            backend.generate(None, params)


def test_llama_cpp_backend_unload_success():
    """Test successful model unloading."""
    backend = LlamaCppBackend()
    
    # Load model first
    with patch('llama_cpp.Llama') as mock_llama:
        mock_instance = Mock()
        mock_llama.return_value = mock_instance
        backend.load_model("/path/to/model.gguf", {})
        
        # Unload should not raise an exception
        backend.unload()
        
        assert backend.is_loaded is False


def test_power_infer_backend_not_implemented():
    """Test that PowerInferBackend raises NotImplementedError."""
    backend = PowerInferBackend()
    
    # Test all methods raise NotImplementedError
    with pytest.raises(NotImplementedError, match="PowerInfer backend is not yet implemented"):
        backend.load_model("/path/to/model.gguf", {})
    
    with pytest.raises(NotImplementedError, match="PowerInfer backend is not yet implemented"):
        backend.generate("Test prompt", GenerateParams())
    
    with pytest.raises(NotImplementedError, match="PowerInfer backend is not yet implemented"):
        backend.unload()


def test_generate_params_default_values():
    """Test that GenerateParams has correct default values."""
    params = GenerateParams()
    
    assert params.max_tokens == 100
    assert params.temperature == 0.7
    assert params.top_p == 0.9


def test_generate_params_custom_values():
    """Test that GenerateParams accepts custom values."""
    params = GenerateParams(max_tokens=200, temperature=0.5, top_p=0.8)
    
    assert params.max_tokens == 200
    assert params.temperature == 0.5
    assert params.top_p == 0.8