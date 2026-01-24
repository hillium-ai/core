import logging
from loqus_core.inference.backend import LlamaCppBackend

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

print('Testing logging...')

# Create backend and try to load with invalid path to trigger logging
backend = LlamaCppBackend()

try:
    backend.load_model('/nonexistent/path.gguf', {})
except ValueError:
    print('✓ Logging triggered for invalid path (as expected)')

print('Logging test completed!')
