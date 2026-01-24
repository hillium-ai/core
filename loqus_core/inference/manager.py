# Native Model Manager for LoqusCore
# Manages lifecycle of LLMs using llama-cpp-python

import os
import time
import threading
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import platform

try:
    from llama_cpp import Llama
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    LLAMA_CPP_AVAILABLE = False
    Llama = None

# Import backend classes
try:
    from loqus_core.inference.backend import LlamaCppBackend, PowerInferBackend, InferenceBackend
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    LlamaCppBackend = None
    PowerInferBackend = None

class ModelStatus(Enum):
    """Model loading/unloading status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"

class ModelConfig:
    """Configuration for model loading"""
    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx

class NativeModelManager:
    """Manages native LLM inference using llama-cpp-python"""
    
    def __init__(self, backend_type: str = "llama.cpp"):
        self.backend_type = backend_type
        self.model = None
        self.status = ModelStatus.UNLOADED
        self.config = None
        self.hardware_info = self._detect_hardware()
        self._lock = threading.Lock()  # For thread safety
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Detect available hardware acceleration.
        
        Target platforms:
        - Production: Jetson Orin/Thor (Linux ARM64 + CUDA)
        - Development: macOS (Metal) or Hillium-Forge container (Linux ARM64, no GPU)
        - Forge emulation: Linux aarch64 container on Mac (CPU only, no CUDA/Metal)
        """
        system = platform.system()
        machine = platform.machine()  # 'aarch64' for Linux ARM, 'arm64' for Mac
        
        hardware = {
            "system": system,
            "machine": machine,
            "metal_available": False,
            "cuda_available": False,
            "accelerator": "cpu",
            "is_jetson": False,
            "is_forge_container": False
        }
        
        # Detect if running in Hillium-Forge container (Linux ARM64 on Mac)
        is_forge = self._detect_forge_container()
        hardware["is_forge_container"] = is_forge
        
        if system == "Linux":
            if is_forge:
                # Hillium-Forge: Linux ARM64 container on Mac - CPU only (no GPU passthrough)
                # This is for compilation/testing, not inference
                hardware["accelerator"] = "cpu"
            else:
                # Real Linux - check for Jetson or NVIDIA GPU
                is_jetson = self._detect_jetson()
                hardware["is_jetson"] = is_jetson
                
                if is_jetson or self._check_nvidia_gpu():
                    hardware["cuda_available"] = True
                    hardware["accelerator"] = "cuda"
                
        elif system == "Darwin":  # macOS (development only)
            # Apple Silicon has Metal
            if machine == "arm64":
                hardware["metal_available"] = True
                hardware["accelerator"] = "metal"
        
        return hardware
    
    def _detect_forge_container(self) -> bool:
        """Detect if running inside Hillium-Forge Docker container"""
        try:
            # Check for Docker container indicators
            if os.path.exists("/.dockerenv"):
                return True
            # Check hostname pattern
            hostname = os.environ.get("HOSTNAME", "")
            if hostname == "forge":
                return True
            # Check if running in /workspace (Forge mount point)
            cwd = os.getcwd()
            if cwd.startswith("/workspace"):
                return True
        except:
            pass
        return False
    
    def _detect_jetson(self) -> bool:
        """Detect if running on NVIDIA Jetson platform"""
        try:
            # Jetson devices have /etc/nv_tegra_release or tegra in device tree
            if os.path.exists("/etc/nv_tegra_release"):
                return True
            # Alternative: check /proc/device-tree/model
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    if "jetson" in model or "tegra" in model or "orin" in model:
                        return True
        except:
            pass
        return False
    
    def _check_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available via nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except:
            return False
    
    def load_model(self, config: ModelConfig) -> bool:
        """
        Load model into memory with hardware-appropriate settings.
        
        Hardware behavior:
        - Metal (Mac): n_gpu_layers=-1 (all layers on GPU)
        - CUDA (Jetson): n_gpu_layers=-1 (all layers on GPU)
        - CPU (Forge): n_gpu_layers=0 (all layers on CPU, slower but works)
        """
        if self.backend_type == "powerinfer":
            # For PowerInfer backend, we would initialize the PowerInfer backend here
            # This is a placeholder - actual implementation would depend on PowerInfer specifics
            print("⚠️  PowerInfer backend selected. This is a placeholder implementation.")
            # In a real implementation, we would initialize PowerInfer backend here
            # For now, we'll just return True to indicate success
            self.status = ModelStatus.READY
            return True
        
        if not BACKENDS_AVAILABLE:
            raise ImportError("Required backend classes not available")
        
        if self.status != ModelStatus.UNLOADED:
            return False
        
        self.status = ModelStatus.LOADING
        self.config = config
        
        try:
            start_time = time.time()
            accelerator = self.hardware_info["accelerator"]
            
            # Determine n_gpu_layers based on environment
            if accelerator == "metal":
                # macOS with Apple Silicon - use all layers on Metal GPU
                n_gpu_layers = -1
            elif accelerator == "cuda":
                # Jetson/Linux with NVIDIA GPU - use all layers on CUDA
                n_gpu_layers = -1
            else:
                # CPU only (Forge container or no GPU available)
                # Force CPU-only inference
                n_gpu_layers = 0
                
            # Allow config override if explicitly set to a specific value
            if config.n_gpu_layers >= 0:
                n_gpu_layers = config.n_gpu_layers
            
            # Warn if running heavy inference in Forge (not recommended)
            if self.hardware_info.get("is_forge_container", False):
                print("⚠️  WARNING: Running in Forge container (CPU-only). "
                      "Inference will be slow. Use for testing only.")
            
            # Use the new backend interface
            if self.backend_type == "llama.cpp":
                # Create LlamaCppBackend instance
                backend = LlamaCppBackend()
                
                # Prepare configuration for backend
                backend_config = {
                    "n_ctx": config.n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                    "verbose": False
                }
                
                # Load model using backend
                backend.load_model(config.model_path, backend_config)
                
                # Store reference to backend
                self.model = backend
            else:
                # Handle unknown backend type
                raise ValueError(f"Unsupported backend type: {self.backend_type}")
            
            self.status = ModelStatus.READY
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s (accelerator: {accelerator}, "
                  f"gpu_layers: {n_gpu_layers})")
            return True
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            print(f"❌ Error loading model: {e}")
            return False
    
    def unload_model(self) -> bool:
        """Unload model from memory"""
        if self.status == ModelStatus.UNLOADED:
            return True
        
        if self.model is not None:
            try:
                # Check if it's a backend instance and use its unload method
                if hasattr(self.model, 'unload'):
                    self.model.unload()
                else:
                    # Fallback to direct deletion for legacy Llama instances
                    del self.model
                self.model = None
            except:
                pass
        
        self.status = ModelStatus.UNLOADED
        self.config = None
        return True
    
    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        if self.status != ModelStatus.READY or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Use the backend's generate method
        if self.backend_type == "llama.cpp" and isinstance(self.model, LlamaCppBackend):
            # Use the backend's generate method
                        # Pass parameters directly to backend's generate method
            params = GenerateParams(max_tokens=max_tokens, temperature=temperature)
            start_time = time.time()
            output = self.model.generate(prompt, params)
            generation_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Note: This measures total generation time, not true TTFT (which requires streaming)
            print(f"Generation time: {generation_time_ms:.1f}ms")
            return output
        else:
            # Fallback to original method for other backends or if model is not a backend
            # This should not happen with our current implementation
            raise RuntimeError("Unsupported backend type for generation")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current manager status"""
        return {
            "status": self.status.value,
            "hardware": self.hardware_info,
            "model_loaded": self.model is not None,
            "can_accelerate": self.can_use_gpu(),
            "environment": self.get_environment_type()
        }
    
    def can_use_gpu(self) -> bool:
        """
        Check if GPU acceleration is available.
        Returns False for Forge container (CPU-only emulation).
        """
        accelerator = self.hardware_info.get("accelerator", "cpu")
        return accelerator in ("metal", "cuda")
    
    def get_environment_type(self) -> str:
        """
        Get the current execution environment type.
        
        Returns:
        - "production": Jetson with CUDA (real hardware)
        - "development": macOS with Metal (dev machine)
        - "forge": Hillium-Forge container (compilation/testing only)
        - "unknown": Unrecognized environment
        """
        hw = self.hardware_info
        
        if hw.get("is_jetson", False):
            return "production"
        elif hw.get("is_forge_container", False):
            return "forge"
        elif hw.get("system") == "Darwin" and hw.get("metal_available", False):
            return "development"
        elif hw.get("cuda_available", False):
            return "production"  # Non-Jetson Linux with CUDA
        else:
            return "unknown"
    
    def is_inference_recommended(self) -> Tuple[bool, str]:
        """
        Check if running inference is recommended in current environment.
        
        Returns:
            (is_recommended, reason)
        """
        env = self.get_environment_type()
        
        if env == "production":
            return True, "Running on production hardware (Jetson/CUDA)"
        elif env == "development":
            return True, "Running on development machine (Metal acceleration)"
        elif env == "forge":
            return False, "Running in Forge container - CPU only, inference will be slow"
        else:
            return False, "Unknown environment - GPU acceleration not detected"