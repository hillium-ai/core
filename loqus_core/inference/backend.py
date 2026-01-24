"""
Inference Backend Interface

Defines the contract for all inference backends in HilliumOS.
Enables hot-swapping between llama.cpp and PowerInfer.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

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
    
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """Load model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            self._model = Llama(
                model_path=path,
                n_ctx=config.get("n_ctx", 4096),
                n_gpu_layers=config.get("n_gpu_layers", -1),
                verbose=config.get("verbose", False),
            )
            self._model_path = path
            logger.info(f"Loaded model: {path}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed")
            raise RuntimeError("llama-cpp-python is required for LlamaCppBackend")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        """Generate text using llama.cpp."""
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
            del self._model
            self._model = None
            self._model_path = None
            logger.info("Model unloaded")
    
    def is_loaded(self) -> bool:
        return self._model is not None


class PowerInferBackend(InferenceBackend):
    """
    Future backend for sparse inference.
    
    NOT IMPLEMENTED for MVP. Returns NotImplementedError.
    See ADR-015 for architecture details.
    """
    
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        logger.warning("PowerInferBackend is not implemented for MVP")
        raise NotImplementedError(
            "PowerInfer integration is planned for post-MVP (v8.6+). "
            "Use LlamaCppBackend for current implementation."
        )
    
    def generate(self, prompt: str, params: GenerateParams) -> GenerateResult:
        raise NotImplementedError("PowerInferBackend not implemented")
    
    def unload(self) -> None:
        pass  # Nothing to unload
    
    def is_loaded(self) -> bool:
        return False


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
    backends = {
        "llama.cpp": LlamaCppBackend,
        "llama_cpp": LlamaCppBackend,
        "powerinfer": PowerInferBackend,
    }
    
    if backend_type.lower() not in backends:
        logger.error(f"Unknown backend: {backend_type}")
        raise ValueError(f"Unknown backend: {backend_type}. Available: {list(backends.keys())}")
    
    return backends[backend_type.lower()]()
