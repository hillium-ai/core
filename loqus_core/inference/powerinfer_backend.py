import logging
from typing import Dict, Any
import os

# Import the wrapper class
try:
    from .powerinfer_backend_wrapper import PowerInferBackend as PowerInferBackendWrapper
    HAS_WRAPPER = True
except ImportError:
    HAS_WRAPPER = False
    PowerInferBackendWrapper = None

from .backend import InferenceBackend, GenerateParams, GenerateResult

logger = logging.getLogger(__name__)

class PowerInferBackend(InferenceBackend):
    """
    PowerInfer backend implementation using Rust FFI.
    
    This backend provides hybrid CPU/GPU sparse inference for HilliumOS.
    """
    
    def __init__(self):
        self._backend_wrapper = None
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
        if not HAS_WRAPPER:
            raise RuntimeError("PowerInfer backend not available: wrapper not found")
        
        # Input validation
        if not isinstance(path, str):
            logger.error(f"Invalid path type: {type(path)}")
            raise TypeError("Model path must be a string")
        
        if not path:
            logger.error("Empty model path provided")
            raise ValueError("Model path cannot be empty")
        
        # Validate path (prevent path traversal)
        try:
            # Check if path is absolute
            if not os.path.isabs(path):
                logger.warning(f"Model path is not absolute: {path}")

            # Security check: prevent path traversal
            # Check if path contains '..' before normalization
            if '..' in path:
                logger.error(f"Path traversal detected in model path: {path}")
                raise ValueError("Path traversal detected in model path")

            # Normalize the path after security checks
            normalized_path = os.path.normpath(path)
            
            # Check if file exists
            if not os.path.exists(normalized_path):
                logger.error(f"Model file does not exist: {normalized_path}")
                raise FileNotFoundError(f"Model file not found: {normalized_path}")

            # Check if it's a file (not a directory)
            if not os.path.isfile(normalized_path):
                logger.error(f"Model path is not a file: {normalized_path}")
                raise ValueError(f"Model path is not a file: {normalized_path}")

        except (FileNotFoundError, ValueError) as e:
            # Re-raise specific exceptions without wrapping
            raise e
        except Exception as e:
            logger.error(f"Model path validation failed: {e}")
            raise RuntimeError(f"Model path validation failed: {e}")
        
        try:
            # Initialize the wrapper
            self._backend_wrapper = PowerInferBackendWrapper()
            
            # Load the model using the wrapper
            self._backend_wrapper.load_model(normalized_path, config)
            self._is_loaded = True
            
            logger.info(f"Loaded PowerInfer model: {normalized_path}")
            
        except Exception as e:
            logger.error(f"Failed to load PowerInfer model: {e}")
            raise RuntimeError(f"PowerInfer model loading failed: {e}")
    
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
        # Input validation
        if not isinstance(prompt, str):
            logger.error(f"Invalid prompt type: {type(prompt)}")
            raise TypeError("Prompt must be a string")
        
        if not prompt:
            logger.warning("Empty prompt provided")
            
        if not self.is_loaded():
            logger.error("Attempted generation without loaded model")
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert GenerateParams to dict for the wrapper
            params_dict = {
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "stop_sequences": params.stop_sequences,
                "seed": params.seed
            }
            
            # Generate using the wrapper
            result = self._backend_wrapper.generate(prompt, params_dict)
            
            # Convert result to GenerateResult
            return GenerateResult(
                text=result["text"],
                tokens_generated=result["tokens_generated"],
                latency_ms=result["latency_ms"],
                finish_reason=result["finish_reason"]
            )
            
        except Exception as e:
            logger.error(f"PowerInfer generation failed: {e}")
            raise RuntimeError(f"PowerInfer generation failed: {e}")
    
    def unload(self) -> None:
        """
        Unload model and release resources.
        
        Safe to call multiple times.
        """
        if self._backend_wrapper is not None:
            try:
                self._backend_wrapper.unload()
                self._is_loaded = False
                logger.info("PowerInfer model unloaded")
            except Exception as e:
                logger.error(f"Error during PowerInfer model unload: {e}")
                self._is_loaded = False
                raise RuntimeError(f"Error during PowerInfer model unload: {e}")
        else:
            # Already unloaded
            self._is_loaded = False
    
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        return self._is_loaded