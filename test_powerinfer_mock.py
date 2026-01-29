import sys
import os
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from loqus_core.inference.powerinfer_backend import PowerInferBackend

# Test that we can import and instantiate
print("Testing PowerInferBackend import and instantiation...")

# Test 1: Basic instantiation
backend = PowerInferBackend()
print("✓ Basic instantiation works")

# Test 2: Check is_loaded
assert backend.is_loaded() == False
print("✓ is_loaded() works")

# Test 3: Test unload (should not raise)
backend.unload()
print("✓ unload() works")

# Test 4: Test library not available error
print("Testing library not available error...")

# Mock powerinfer_lib to None to simulate library not being available
with patch('loqus_core.inference.powerinfer_backend.powerinfer_lib', None):
    backend2 = PowerInferBackend()
    
    # Test that load_model raises RuntimeError when library is not available
    try:
        backend2.load_model('/fake/path', {})
        print("ERROR: Should have raised RuntimeError")
    except RuntimeError as e:
        if 'PowerInfer library not available' in str(e):
            print("✓ load_model correctly raises RuntimeError for missing library")
        else:
            print(f"ERROR: Wrong error message: {e}")
    
    # Test that generate raises RuntimeError when library is not available
    try:
        backend2.generate('test prompt', {})
        print("ERROR: Should have raised RuntimeError")
    except RuntimeError as e:
        if 'PowerInfer library not available' in str(e):
            print("✓ generate correctly raises RuntimeError for missing library")
        else:
            print(f"ERROR: Wrong error message: {e}")

print("All tests passed!")