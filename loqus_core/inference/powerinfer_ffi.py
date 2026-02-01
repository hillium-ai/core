#!/usr/bin/env python3

"""
PowerInfer FFI Helper Module

This module provides ctypes-based interface to the Rust PowerInfer library.
"""

import ctypes
import os
from typing import Optional

# Global FFI handle
_powerinfer_lib = None


def load_powerinfer_library() -> Optional[ctypes.CDLL]:
    """
    Load the PowerInfer Rust library if available.
    
    Returns:
        ctypes.CDLL object if library loaded successfully, None otherwise
    """
    global _powerinfer_lib
    
    if _powerinfer_lib is not None:
        return _powerinfer_lib
    
    # Look for the library in standard locations
    lib_paths = [
        "target/release/libpowerinfer_rs.so",
        "target/debug/libpowerinfer_rs.so",
        "libpowerinfer_rs.so"
    ]
    
    for path in lib_paths:
        if os.path.exists(path):
            try:
                _powerinfer_lib = ctypes.CDLL(path)
                print(f"Loaded PowerInfer library from {path}")
                return _powerinfer_lib
            except Exception as e:
                print(f"Failed to load PowerInfer library from {path}: {e}")
                continue
    
    print("PowerInfer library not found, falling back to mock mode")
    return None


def get_powerinfer_library() -> Optional[ctypes.CDLL]:
    """
    Get the PowerInfer library handle, loading it if necessary.
    
    Returns:
        ctypes.CDLL object or None if not available
    """
    global _powerinfer_lib
    
    if _powerinfer_lib is None:
        return load_powerinfer_library()
    
    return _powerinfer_lib


def is_powerinfer_available() -> bool:
    """
    Check if PowerInfer library is available.
    
    Returns:
        True if library is available, False otherwise
    """
    return get_powerinfer_library() is not None


# Export functions
__all__ = ["load_powerinfer_library", "get_powerinfer_library", "is_powerinfer_available"]
