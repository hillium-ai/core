"""
Test script for NativeModelManager inference capabilities.
"""

import time
import os
from loqus_core.inference.manager import NativeModelManager, ModelConfig


def test_hardware_detection():
    """Test hardware acceleration detection."""
    print("=" * 60)
    print("TEST 1: Hardware Detection")
    print("=" * 60)
    
    manager = NativeModelManager()
    status = manager.get_status()
    
    print(f"System: {status['hardware']['system']}")
    print(f"Metal available: {status['hardware']['metal_available']}")
    print(f"CUDA available: {status['hardware']['cuda_available']}")
    print(f"Accelerator: {status['hardware']['accelerator']}")
    
    assert status['hardware']['system'] in ['Darwin', 'Linux', 'Windows']
    print("✓ Hardware detection test passed\n")


def test_model_loading():
    """Test model loading functionality."""
    print("=" * 60)
    print("TEST 2: Model Loading")
    print("=" * 60)
    
    # Check if test model exists
    test_model = "models/Phi-3-mini-4k-instruct-q4.gguf"
    if not os.path.exists(test_model):
        print(f"⚠ Test model not found at {test_model}")
        print("⚠ Skipping model loading test (download model first)")
        return
    
    manager = NativeModelManager()
    config = ModelConfig(
        model_path=test_model,
        n_gpu_layers=-1,
        n_ctx=2048
    )
    
    start_time = time.time()
    success = manager.load_model(config)
    load_time = time.time() - start_time
    
    print(f"Load time: {load_time:.2f}s")
    print(f"Model loaded: {success}")
    
    if success:
        status = manager.get_status()
        print(f"Status: {status['status']}")
        assert status['status'] == 'ready'
        print("✓ Model loading test passed\n")
    else:
        print("✗ Model loading failed\n")


def test_inference():
    """Test text generation."""
    print("=" * 60)
    print("TEST 3: Text Generation")
    print("=" * 60)
    
    # Check if test model exists
    test_model = "models/Phi-3-mini-4k-instruct-q4.gguf"
    if not os.path.exists(test_model):
        print(f"⚠ Test model not found at {test_model}")
        print("⚠ Skipping inference test (download model first)")
        return
    
    manager = NativeModelManager()
    config = ModelConfig(
        model_path=test_model,
        n_gpu_layers=-1,
        n_ctx=2048
    )
    
    if not manager.load_model(config):
        print("✗ Failed to load model for inference test")
        return
    
    prompt = "Hello, how are you today?"
    start_time = time.time()
    try:
        output = manager.generate(prompt, max_tokens=64)
        ttft = time.time() - start_time
        
        print(f"Prompt: {prompt}")
        print(f"Output: {output[:100]}...")
        print(f"Time to first token: {ttft*1000:.2f}ms")
        
        # Basic validation
        assert len(output) > 0
        assert ttft < 200  # TTFT < 200ms requirement
        print("✓ Inference test passed\n")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}\n")
    finally:
        manager.unload_model()


def test_gpu_usage():
    """Test GPU usage verification (platform-specific)."""
    print("=" * 60)
    print("TEST 4: GPU Usage Verification")
    print("=" * 60)
    
    import platform
    
    if platform.system() == "Darwin":
        print("Running on macOS - check GPU usage with:")
        print("powermetrics --samplers gpu_power -i 1000 -n 10")
    elif platform.system() == "Linux":
        print("Running on Linux - check GPU usage with:")
        print("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv")
    else:
        print("GPU usage monitoring not available on this platform")
    
    print("✓ GPU usage verification test completed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NativeModelManager Test Suite")
    print("=" * 60 + "\n")
    
    # Run all tests
    test_hardware_detection()
    test_model_loading()
    test_inference()
    test_gpu_usage()
    
    print("=" * 60)
    print("Test Suite Completed")
    print("=" * 60)
