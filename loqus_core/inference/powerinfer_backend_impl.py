import ctypes
import os
import json
from typing import Dict, Any

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
    
except Exception as e:
    raise RuntimeError(f"Failed to load PowerInfer library: {e}")


def load_model(path: str, config: Dict[str, Any]) -> ctypes.c_void_p:
    """Load a model and return a handle."""
    try:
        # Convert path to bytes
        path_bytes = path.encode('utf-8')
        
        # Convert config to JSON string
        config_json = json.dumps(config).encode('utf-8')
        
        # Call the Rust function
        model_handle = powerinfer_lib.powerinfer_load_model(path_bytes, config_json)
        
        if model_handle is None:
            raise RuntimeError("Failed to load model")
        
        return model_handle
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def generate(model_handle: ctypes.c_void_p, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate text using the model."""
    try:
        # Convert prompt to bytes
        prompt_bytes = prompt.encode('utf-8')
        
        # Convert params to JSON string
        params_json = json.dumps(params).encode('utf-8')
        
        # Call the Rust function
        result_bytes = powerinfer_lib.powerinfer_generate(model_handle, prompt_bytes, params_json)
        
        if result_bytes is None:
            raise RuntimeError("Generation failed")
        
        # Decode result
        result_str = result_bytes.decode('utf-8')
        
        # Parse JSON result
        return json.loads(result_str)
    except Exception as e:
        raise RuntimeError(f"Failed to generate: {e}")


def destroy_model(model_handle: ctypes.c_void_p) -> None:
    """Destroy the model."""
    try:
        powerinfer_lib.powerinfer_destroy_model(model_handle)
    except Exception as e:
        raise RuntimeError(f"Failed to destroy model: {e}")


def is_loaded(model_handle: ctypes.c_void_p) -> bool:
    """Check if model is loaded."""
    try:
        return powerinfer_lib.powerinfer_is_loaded(model_handle)
    except Exception as e:
        raise RuntimeError(f"Failed to check if model is loaded: {e}")