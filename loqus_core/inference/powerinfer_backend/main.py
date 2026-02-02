import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PowerInferBackendError(Exception):
    """Custom exception for PowerInfer backend errors."""
    pass


class PowerInferBackend:
    """
    PowerInfer backend implementation using Rust FFI.
    
    This backend provides hybrid CPU/GPU sparse inference for HilliumOS.
    """
    
    def __init__(self):
        self._model_handle = None
        self._is_loaded = False
        self._is_available = False
        
        # Try to import PowerInfer FFI
        try:
            from .powerinfer_ffi import (
                powerinfer_load_model,
                powerinfer_generate,
                powerinfer_destroy_model,
                powerinfer_is_loaded,
                is_powerinfer_available
            )
            self._load_model_func = powerinfer_load_model
            self._generate_func = powerinfer_generate
            self._destroy_model_func = powerinfer_destroy_model
            self._is_loaded_func = powerinfer_is_loaded
            self._is_available = is_powerinfer_available()
            logger.info("PowerInfer backend initialized successfully")
        except ImportError as e:
            logger.warning(f"PowerInfer FFI module not found: {e}")
            logger.warning("Using mock mode for PowerInfer backend")
            self._is_available = False

    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """
        Load a model from disk using PowerInfer backend.
        
        Args:
            path: Path to model file (GGUF format)
            config: Backend-specific configuration
            
        Raises:
            PowerInferBackendError: If loading fails
        """
        if not self._is_available:
            logger.warning("PowerInfer backend not available - using mock mode")
            # In mock mode, we just set the state
            self._is_loaded = True
            return
        
        # Validate path
        if not isinstance(path, str):
            logger.error(f"Invalid path type: {type(path)}")
            raise TypeError("Model path must be a string")
        
        if not path:
            logger.error("Empty model path provided")
            raise ValueError("Model path cannot be empty")
        
        try:
            # Call the PowerInfer FFI function to load model
            model_handle = self._load_model_func(path, config)
            
            if model_handle is None:
                raise PowerInferBackendError("Failed to load model via PowerInfer backend")
            
            self._model_handle = model_handle
            self._is_loaded = True
            logger.info(f"Loaded model using PowerInfer backend: {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise PowerInferBackendError(f"Model loading failed: {e}")
    
    def generate(self, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate text from prompt using PowerInfer backend.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            Dict with generated text and metadata
            
        Raises:
            PowerInferBackendError: If generation fails
        """
        if not self.is_loaded():
            logger.error("Attempted generation without loaded model")
            raise PowerInferBackendError("Model not loaded. Call load_model() first.")
        
        if not isinstance(prompt, str):
            logger.error(f"Invalid prompt type: {type(prompt)}")
            raise TypeError("Prompt must be a string")
        
        if not prompt:
            logger.warning("Empty prompt provided")
            
        if not self._is_available:
            # Mock mode - return dummy result
            logger.info("Using mock mode for generation")
            return {
                "text": "[MOCK] Generated text for: " + prompt,
                "tokens_generated": len(prompt.split()),
                "latency_ms": 10.0,
                "finish_reason": "stop",
            }
        
        try:
            # Call the PowerInfer FFI function to generate
            result_json = self._generate_func(self._model_handle, prompt, params)
            
            if result_json is None:
                raise PowerInferBackendError("Generation failed via PowerInfer backend")
            
            # Parse the JSON result
            import json
            result_data = json.loads(result_json)
            
            return result_data
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise PowerInferBackendError(f"Generation failed: {e}")
    
    def unload(self) -> None:
        """
        Unload model and release resources.
        """
        if self._model_handle is not None and self._is_available:
            try:
                self._destroy_model_func(self._model_handle)
                self._model_handle = None
                self._is_loaded = False
                logger.info("PowerInfer model unloaded")
            except Exception as e:
                logger.error(f"Error during model unload: {e}")
                self._model_handle = None
                self._is_loaded = False
                raise PowerInferBackendError(f"Error during model unload: {e}")
        else:
            self._model_handle = None
            self._is_loaded = False
    
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        if not self._is_available:
            # Mock mode - rely on internal flag
            return self._is_loaded

        if self._model_handle is None:
            return False

        try:
            return self._is_loaded_func(self._model_handle)
        except Exception:
            return False

    def is_available(self) -> bool:
        """
        Check if PowerInfer backend is available.
        """
        return self._is_available
