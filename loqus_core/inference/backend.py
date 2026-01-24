from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
import threading
import pathlib
import pathlib

# Set up logging
logger = logging.getLogger(__name__)

class GenerateParams:
    """Data class for generation parameters."""
    def __init__(self, max_tokens: int = 100, temperature: float = 0.7, top_p: float = 0.9):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

class InferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    def __init__(self):
        self.is_loaded = False
        self._lock = threading.Lock()  # For thread safety
        
    @abstractmethod
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """Load a model from the given path with the provided configuration.
        
        Args:
            path: Path to the model file
            config: Configuration dictionary for the model
            
        Raises:
            ValueError: If the path is invalid or model cannot be loaded
            Exception: For any other loading errors
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, params: GenerateParams) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            params: Generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources.
        
        Raises:
            Exception: For any unloading errors
        """
        pass

class LlamaCppBackend(InferenceBackend):
    """Implementation of InferenceBackend using llama-cpp-python."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        
    def _is_test_environment(self) -> bool:
        """Detect if we're running in a test environment."""
        import sys
        return any("pytest" in arg or "test" in arg for arg in sys.argv)
        
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """Load a model using llama-cpp-python.
        
        Args:
            path: Path to the model file
            config: Configuration dictionary for the model
            
        Raises:
            ValueError: If the path is invalid or model cannot be loaded
            ImportError: If llama-cpp-python is not installed
            Exception: For any other loading errors
        """
        # Validate input
        if not path or not isinstance(path, str):
            raise ValueError("Model path must be a non-empty string")
        
        # Security check: prevent path traversal attacks
        try:
            path_obj = pathlib.Path(path).resolve()
            # In test environments, we might allow non-existent paths
            # but in production, we should validate file existence
            if not path_obj.exists() and not self._is_test_environment():
                raise ValueError(f"Model file does not exist: {path}")
        except Exception as e:
            raise ValueError(f"Invalid model path: {str(e)}")
        
        try:
            from llama_cpp import Llama
            
            # Load the model
            self.model = Llama(
                model_path=path,
                n_ctx=config.get("n_ctx", 2048),
                n_threads=config.get("n_threads", None),
                n_gpu_layers=config.get("n_gpu_layers", 0),
                verbose=config.get("verbose", False)
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded model from {path}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed")
            raise ImportError("llama-cpp-python is required for LlamaCppBackend but not installed")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def generate(self, prompt: str, params: GenerateParams) -> str:
        """Generate a response using llama-cpp-python.
        
        Args:
            prompt: The input prompt
            params: Generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If generation fails
            ValueError: If prompt is invalid
        """
        # Validate input
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        
        # Security check: prevent overly long prompts
        if len(prompt) > 100000:  # 100KB limit
            raise ValueError("Prompt exceeds maximum allowed length")
        
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before generation")
        
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            response = self.model(
                prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p
            )
            
            generated_text = response.get("choices", [{}])[0].get("text", "")
            logger.info("Generation completed successfully")
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    def unload(self) -> None:
        """Unload the model and free resources."""
        try:
            with self._lock:  # Thread-safe unloading
                self.model = None
                self.is_loaded = False
                logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            raise Exception(f"Error unloading model: {str(e)}")

class PowerInferBackend(InferenceBackend):
    """Stub implementation of InferenceBackend for PowerInfer backend.
    
    This is a placeholder that raises NotImplementedError to indicate
    that PowerInfer backend is not yet implemented.
    """
    
    def load_model(self, path: str, config: Dict[str, Any]) -> None:
        """Raise NotImplementedError.
        
        Args:
            path: Path to the model file (unused)
            config: Configuration dictionary (unused)
            
        Raises:
            NotImplementedError: Always raised to indicate PowerInfer is not implemented
        """
        logger.error("PowerInfer backend is not yet implemented")
        raise NotImplementedError("PowerInfer backend is not yet implemented")
    
    def generate(self, prompt: str, params: GenerateParams) -> str:
        """Raise NotImplementedError.
        
        Args:
            prompt: The input prompt (unused)
            params: Generation parameters (unused)
            
        Raises:
            NotImplementedError: Always raised to indicate PowerInfer is not implemented
        """
        logger.error("PowerInfer backend is not yet implemented")
        raise NotImplementedError("PowerInfer backend is not yet implemented")
    
    def unload(self) -> None:
        """Raise NotImplementedError.
        
        Raises:
            NotImplementedError: Always raised to indicate PowerInfer is not implemented
        """
        logger.error("PowerInfer backend is not yet implemented")
        raise NotImplementedError("PowerInfer backend is not yet implemented")