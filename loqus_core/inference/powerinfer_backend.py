import ctypes
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .backend import InferenceBackend, GenerateParams, GenerateResult

# Load the Rust library
try:
    # This will be updated to load the actual compiled library
    lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "libpowerinfer_rs.so")
    if not os.path.exists(lib_path):
        # Try debug build
        lib_path = os.path.join(os.path.dirname(__file__), "..", "..", "target", "debug", "libpowerinfer_rs.so")
    
    powerinfer_lib = ctypes.CDLL(lib_path)
    
    # Define function signatures
    powerinfer_lib.powerinfer_load_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    powerinfer_lib.powerinfer_load_model.restype = ctypes.c_void_p
    
    powerinfer_lib.powerinfer_destroy_model.argtypes = [ctypes.c_void_p]
    powerinfer_lib.powerinfer_destroy_model.restype = None
    
    powerinfer_lib.powerinfer_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    powerinfer_lib.powerinfer_generate.restype = ctypes.c_char_p
    
    powerinfer_lib.powerinfer_is_loaded.argtypes = [ctypes.c_void_p]
    powerinfer_lib.powerinfer_is_loaded.restype = ctypes.c_bool
    
except (OSError, ImportError) as e:
    # Fallback to NotImplementedError if library not available
    print(f"Warning: PowerInfer library not available: {e}")
    powerinfer_lib = None


class PowerInferBackend(InferenceBackend):
    """
    PowerInfer backend implementation using Rust FFI.
    
    This backend provides hybrid CPU/GPU sparse inference for HilliumOS.
    """
    
    def __init__(self):
        self._model_handle = None
        self._is_loaded = False
        
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """
        Load a model from disk using PowerInfer backend.
        
        Args:
            path: Path to model file (GGUF format)
            config: Backend-specific configuration
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        if powerinfer_lib is None:
            raise RuntimeError("PowerInfer library not available")
        
        # Validate path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if not os.path.isfile(path):
            raise ValueError(f"Model path is not a file: {path}")
        
        # Convert path to bytes
        c_path = path.encode('utf-8')
        
        # Convert config to JSON string
        import json
        config_json = json.dumps(config)
        c_config = config_json.encode('utf-8')
        
        # Call C function to load model
        model_handle = powerinfer_lib.powerinfer_load_model(c_path, c_config)
        
        if model_handle is None or model_handle == 0:
            raise RuntimeError("Failed to load model with PowerInfer backend")
        
        self._model_handle = model_handle
        self._is_loaded = True
        
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """
        Generate text from prompt using PowerInfer backend.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            GenerateResult with generated text and metadata
            
        Raises:
            RuntimeError: If model not loaded or generation fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if powerinfer_lib is None:
            raise RuntimeError("PowerInfer library not available")
        
        # Convert prompt to bytes
        c_prompt = prompt.encode('utf-8')
        
        # Convert params to JSON string
        import json
        params_dict = {
            'max_tokens': params.max_tokens,
            'temperature': params.temperature,
            'top_p': params.top_p,
            'top_k': params.top_k,
            'stop_sequences': params.stop_sequences,
            'seed': params.seed
        }
        params_json = json.dumps(params_dict)
        c_params = params_json.encode('utf-8')
        
        # Call C function to generate
        result_ptr = powerinfer_lib.powerinfer_generate(
            self._model_handle,
            c_prompt,
            c_params
        )
        
        if result_ptr is None:
            raise RuntimeError("Generation failed with PowerInfer backend")
        
        # Convert result back to Python
        result_str = ctypes.string_at(result_ptr).decode('utf-8')
        
        # Parse JSON result
        import json
        result_data = json.loads(result_str)
        
        return GenerateResult(
            text=result_data['text'],
            tokens_generated=result_data['tokens_generated'],
            latency_ms=result_data['latency_ms'],
            finish_reason=result_data['finish_reason']
        )
        
    def unload(self) -> None:
        """
        Unload model and release resources.
        """
        if powerinfer_lib is not None and self._model_handle is not None:
            powerinfer_lib.powerinfer_destroy_model(self._model_handle)
            self._model_handle = None
            self._is_loaded = False
        
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        return self._is_loaded
