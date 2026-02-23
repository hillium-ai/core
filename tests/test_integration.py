from loqus_core.inference.backend import LlamaCppBackend, GenerateParams
from loqus_core.inference.manager import NativeModelManager

print("Testing backend and manager integration...")

# Test backend creation
backend = LlamaCppBackend()
print("✅ LlamaCppBackend created successfully")

# Test manager creation
manager = NativeModelManager()
print("✅ NativeModelManager created successfully")

print("All integration tests passed!")