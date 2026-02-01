"""
PowerInfer FFI module for Python integration.

This module provides Python bindings to the Rust PowerInfer backend.
"""

import os
import sys
from typing import Optional, Dict, Any

# This will be populated with the actual FFI functions
# when the Rust extension is built and imported

try:
    # Try to import the compiled Rust extension
    import pyo3_powerinfer
    _HAS_POWERINFER = True
    
    # Bind the FFI functions
    powerinfer_load_model = pyo3_powerinfer.powerinfer_load_model
    powerinfer_generate = pyo3_powerinfer.powerinfer_generate
    powerinfer_destroy_model = pyo3_powerinfer.powerinfer_destroy_model
    powerinfer_is_loaded = pyo3_powerinfer.powerinfer_is_loaded
    is_powerinfer_available = pyo3_powerinfer.is_powerinfer_available
    
except ImportError:
    # Fallback to mock implementation if not available
    _HAS_POWERINFER = False
    
    def powerinfer_load_model(path: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print(f"[MOCK] Loading model from {path}")
        return {'model_id': path, 'mock': True}
    
    def powerinfer_generate(handle: Dict[str, Any], prompt: str, params: Dict[str, Any]) -> Optional[str]:
        print(f"[MOCK] Generating for prompt: {prompt[:50]}...")
        return '{"text": "[MOCK] Generated text", "tokens_generated": 10, "latency_ms": 10.0, "finish_reason": "stop"}'
    
    def powerinfer_destroy_model(handle: Dict[str, Any]) -> None:
        print("[MOCK] Destroying model")
    
    def powerinfer_is_loaded(handle: Dict[str, Any]) -> bool:
        return True
    
    def is_powerinfer_available() -> bool:
        return False

__all__ = [
    'powerinfer_load_model',
    'powerinfer_generate',
    'powerinfer_destroy_model',
    'powerinfer_is_loaded',
    'is_powerinfer_available',
    '_HAS_POWERINFER'
]
