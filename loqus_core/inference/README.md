# PowerInfer Backend for HilliumOS

The PowerInfer backend is a hybrid CPU/GPU sparse inference implementation designed for HilliumOS. It provides efficient model inference using Rust FFI (Foreign Function Interface) for optimal performance on Jetson devices.

## Features

- Hybrid CPU/GPU inference
- Sparse matrix operations
- GGUF model format support
- Memory-efficient model loading
- FFI integration with Rust

## Architecture

The PowerInfer backend consists of:

1. **Python Interface** (`powerinfer_backend.py`): Provides the Python API that integrates with the core inference system
2. **Rust FFI Module** (`powerinfer_ffi.py`): Provides Python bindings to the Rust implementation
3. **Rust Implementation** (external): Core inference logic implemented in Rust for performance

## Usage

