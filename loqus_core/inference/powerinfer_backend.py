import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from loqus_core.inference.backend import InferenceBackend, GenerateParams, GenerateResult

logger = logging.getLogger(__name__)


class PowerInferBackend(InferenceBackend):
    """
    PowerInfer backend implementation using Rust FFI.
    
    This backend provides hybrid CPU/GPU sparse inference for HilliumOS.
    """
    
    def __init__(self):
        self._model_handle = None
        self._is_loaded = False
        # Import the Rust FFI module
        try:
            import pyo3_powerinfer
            self._ffi_module = pyo3_powerinfer
        except ImportError:
            logger.error("pyo3_powerinfer module not found")
            self._ffi_module = None
            
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """
        Load a model from disk using PowerInfer backend.
        
        Args:
            path: Path to model file (GGUF format)
            config: Backend-specific configuration
            
        Raises:
            RuntimeError: If loading fails
        """
        if self._ffi_module is None:
            raise RuntimeError("PowerInfer backend not available - pyo3_powerinfer module not found")
        
        # Validate path
        if not isinstance(path, str):
            logger.error(f"Invalid path type: {type(path)}")
            raise TypeError("Model path must be a string")
        
        if not path:
            logger.error("Empty model path provided")
            raise ValueError("Model path cannot be empty")
        
        try:
            # Call the Rust FFI function to load model
            model_handle = self._ffi_module.powerinfer_load_model(path, config)
            
            if model_handle is None:
                raise RuntimeError("Failed to load model via PowerInfer backend")
            
            self._model_handle = model_handle
            self._is_loaded = True
            logger.info(f"Loaded model using PowerInfer backend: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """
        Generate text from prompt using PowerInfer backend.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            GenerateResult with generated text and metadata
            
        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded():
            logger.error("Attempted generation without loaded model")
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not isinstance(prompt, str):
            logger.error(f"Invalid prompt type: {type(prompt)}")
            raise TypeError("Prompt must be a string")
        
        if not prompt:
            logger.warning("Empty prompt provided")
            
        try:
            # Convert GenerateParams to dictionary for FFI
            params_dict = {
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "stop_sequences": params.stop_sequences,
                "seed": params.seed
            }
            
            # Call the Rust FFI function to generate
            result_json = self._ffi_module.powerinfer_generate(self._model_handle, prompt, params_dict)
            
            if result_json is None:
                raise RuntimeError("Generation failed via PowerInfer backend")
            
            # Parse the JSON result
            import json
            result_data = json.loads(result_json)
            
            return GenerateResult(
                text=result_data["text"],
                tokens_generated=result_data["tokens_generated"],
                latency_ms=result_data["latency_ms"],
                finish_reason=result_data["finish_reason"],
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def unload(self) -> None:
        """
        Unload model and release resources.
        """
        if self._model_handle is not None and self._ffi_module is not None:
            try:
                self._ffi_module.powerinfer_destroy_model(self._model_handle)
                self._model_handle = None
                self._is_loaded = False
                logger.info("PowerInfer model unloaded")
            except Exception as e:
                logger.error(f"Error during model unload: {e}")
                self._model_handle = None
                self._is_loaded = False
                raise RuntimeError(f"Error during model unload: {e}")
        else:
            self._model_handle = None
            self._is_loaded = False
            
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        if self._ffi_module is None or self._model_handle is None:
            return False
        
        try:
            return self._ffi_module.powerinfer_is_loaded(self._model_handle)
        except Exception:
            return False