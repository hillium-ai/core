# Loqus Core Configuration

## Backend Configuration

The system supports switching between different inference backends:

- `llama.cpp` - Uses llama-cpp-python for model inference
- `powerinfer` - Placeholder for PowerInfer backend (not yet implemented)

### Usage

To configure the backend, set the `backend_type` parameter when initializing `NativeModelManager`:

```python
from loqus_core.inference.manager import NativeModelManager

# Use llama.cpp backend (default)
manager = NativeModelManager(backend_type="llama.cpp")

# Use powerinfer backend (placeholder)
manager = NativeModelManager(backend_type="powerinfer")
```
