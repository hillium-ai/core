class PowerInferBackend(InferenceBackend):
    """
    PowerInfer backend implementation using Rust FFI.
    
    This backend provides hybrid CPU/GPU sparse inference for HilliumOS.
    """
    
    def __init__(self):
        self._model_handle = None
        self._is_loaded = False
        self._ffi_module = None
        # Import the PowerInfer FFI module
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
        except ImportError:
            logger.warning("PowerInfer backend not available - using mock mode")
            # Create a mock implementation
            self._is_available = False
    
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
            logger.warning("PowerInfer backend not available - using mock mode")
            # In mock mode, we just set the state
            self._is_loaded = True
            return
        
        try:
            # Call the FFI function directly
            self._model_handle = self._load_model_func(path, config)
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
            
        if self._ffi_module is None:
            # Mock mode - return dummy result
            logger.info("Using mock mode for generation")
            return GenerateResult(
                text="[MOCK] Generated text for: " + prompt,
                tokens_generated=len(prompt.split()),
                latency_ms=10.0,
                finish_reason="stop",
            )
        
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
            
            # Call the FFI function directly
            result_json = self._generate_func(self._model_handle, prompt, params_dict)
            
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
        if self._ffi_module is not None and self._model_handle is not None:
            try:
                self._destroy_model_func(self._model_handle)
                self._model_handle = None
                self._is_loaded = False
                logger.info("PowerInfer model unloaded")
            except Exception as e:
                logger.error(f"Error during model unload: {e}")
                self._is_loaded = False
                raise RuntimeError(f"Error during model unload: {e}")
        else:
            self._is_loaded = False
    
    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        """
        if self._ffi_module is None:
            # Mock mode - rely on internal flag
            return self._is_loaded

        try:
            return self._is_loaded_func(self._model_handle)
        except Exception:
            return False