import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import pathlib

logger = logging.getLogger(__name__)


@dataclass
class GenerateParams:
    """Parameters for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: tuple = ()
    seed: Optional[int] = None


@dataclass  
class GenerateResult:
    """Result from text generation."""
    text: str
    tokens_generated: int
    latency_ms: float
    finish_reason: str  # "stop", "length", "error"


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All backends must implement these methods:
    - load_model(): Load model into memory
    - generate(): Generate text from prompt
    - unload(): Release model resources
    - is_loaded(): Check if model is loaded
    
    Example:
        backend = LlamaCppBackend()
        backend.load_model("/path/to/model.gguf", {})
        result = backend.generate("Hello", GenerateParams())
        backend.unload()
    """
    
    @abstractmethod
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """
        Load a model from disk.
        
        Args:
            path: Path to model file (GGUF format)
            config: Backend-specific configuration
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            GenerateResult with generated text and metadata
            
        Raises:
            RuntimeError: If model not loaded or generation fails
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload model and release resources.
        
        Safe to call multiple times.
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass


class LlamaCppBackend(InferenceBackend):
    """
    Default inference backend using llama.cpp.
    
    This is the production backend for HilliumOS MVP.
    """
    
    def __init__(self):
        self._model = None
        self._model_path: Optional[str] = None
        self._is_loaded = False
    
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """Load model using llama-cpp-python."""
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
            from llama_cpp import Llama
            
            self._model = Llama(
                model_path=normalized_path,
                n_ctx=config.get("n_ctx", 4096),
                n_gpu_layers=config.get("n_gpu_layers", -1),
                verbose=config.get("verbose", False),
            )
            self._model_path = normalized_path
            logger.info(f"Loaded model: {normalized_path}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed")
            raise RuntimeError("llama-cpp-python is required for LlamaCppBackend")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """Generate text using llama.cpp."""
        # Input validation
        if not isinstance(prompt, str):
            logger.error(f"Invalid prompt type: {type(prompt)}")
            raise TypeError("Prompt must be a string")
        
        if not prompt:
            logger.warning("Empty prompt provided")
            
        if not self.is_loaded():
            logger.error("Attempted generation without loaded model")
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import time
        start = time.perf_counter()
        
        try:
            output = self._model(
                prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                stop=list(params.stop_sequences) if params.stop_sequences else None,
            )
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            return GenerateResult(
                text=output["choices"][0]["text"],
                tokens_generated=output["usage"]["completion_tokens"],
                latency_ms=elapsed_ms,
                finish_reason=output["choices"][0]["finish_reason"],
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def unload(self) -> None:
        """Unload model."""
        if self._model is not None:
            try:
                del self._model
                self._model = None
                self._model_path = None
                logger.info("Model unloaded")
            except Exception as e:
                logger.error(f"Error during model unload: {e}")
                # Continue to clear references even if deletion fails
                self._model = None
                self._model_path = None
                raise RuntimeError(f"Error during model unload: {e}")
        
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        return self._is_loaded


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
            NotImplementedError: If not implemented
        """
        # This is a placeholder implementation
        # In a real implementation, this would load a model using Rust FFI
        raise NotImplementedError("PowerInfer backend is not fully implemented yet")
        
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """
        Generate text from prompt using PowerInfer backend.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            GenerateResult with generated text and metadata
            
        Raises:
            NotImplementedError: If not implemented
        """
        # This is a placeholder implementation
        # In a real implementation, this would generate text using Rust FFI
        raise NotImplementedError("PowerInfer backend is not fully implemented yet")
        
    def unload(self) -> None:
        """
        Unload model and release resources.
        """
        self._model_handle = None
        self._is_loaded = False
        logger.info("PowerInfer model unloaded")
        
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        return self._is_loaded


def get_backend(backend_type: str = "llama.cpp") -> InferenceBackend:
    """
    Factory function to get inference backend.
    
    Args:
        backend_type: "llama.cpp" or "powerinfer"
        
    Returns:
        InferenceBackend instance
        
    Raises:
        ValueError: If backend_type unknown
    """
    # Import here to avoid circular import
    if backend_type.lower() == "powerinfer":
        from .powerinfer_backend import PowerInferBackend
    else:
        from .backend import LlamaCppBackend
    
    backends = {
        "llama.cpp": LlamaCppBackend,
        "llama_cpp": LlamaCppBackend,
        "powerinfer": PowerInferBackend,
    }
    
    if backend_type.lower() not in backends:
        logger.error(f"Unknown backend: {backend_type}")
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type.lower()]()