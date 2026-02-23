# Python bindings for fibonacci_math crate

import sys
from ctypes import cdll, c_double, c_int, c_size_t

# Load the shared library
try:
    # This will be updated to load the actual compiled library
    lib = cdll.LoadLibrary("libfibonacci_math.so")
except OSError:
    # Fallback for development
    try:
        lib = cdll.LoadLibrary("target/debug/libfibonacci_math.so")
    except OSError:
        print("Could not load fibonacci_math library")
        raise
