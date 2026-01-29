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


class PowerInferBackend:
    """
    Wrapper class for PowerInfer Rust FFI.
    """
    
    def __init__(self):
        self._model_handle = None
        
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """
        Load a model from disk.
        """
        # Validate path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if not os.path.isfile(path):
            raise ValueError(f"Model path is not a file: {path}")
        
        # Convert path to bytes
        c_path = path.encode('utf-8')
        
        # Convert config to JSON string
        config_json = json.dumps(config)
        c_config = config_json.encode('utf-8')
        
        # Call C function to load model
        model_handle = powerinfer_lib.powerinfer_load_model(c_path, c_config)
        
        if model_handle is None or model_handle == 0:
            raise RuntimeError("Failed to load model")
        
        self._model_handle = model_handle
        
    def generate(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate text using the loaded model.
        """
        if self._model_handle is None:
            raise RuntimeError("Model not loaded")
        
        # Convert prompt to bytes
        c_prompt = prompt.encode('utf-8')
        
        # Convert params to JSON string
        params_json = json.dumps(params)
        c_params = params_json.encode('utf-8')
        
        # Call C function to generate
        result_ptr = powerinfer_lib.powerinfer_generate(
            self._model_handle,
            c_prompt,
            c_params
        )
        
        if result_ptr is None:
            raise RuntimeError("Generation failed")
        
        # Decode result
        result_str = result_ptr.decode('utf-8')
        
        # Parse JSON result
        return json.loads(result_str)
        
    def unload(self) -> None:
        """
        Unload the model and release resources.
        """
        if self._model_handle is not None:
            powerinfer_lib.powerinfer_destroy_model(self._model_handle)
            self._model_handle = None