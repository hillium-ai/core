from loqus_core.inference.backend import LlamaCppBackend

# Test error handling
print('Testing error handling...')

backend = LlamaCppBackend()

# Test invalid path
try:
    backend.load_model('', {})
    print('ERROR: Should have raised ValueError for empty path')
except ValueError as e:
    print(f'✓ Correctly caught ValueError for empty path: {e}')

try:
    backend.load_model(None, {})
    print('ERROR: Should have raised ValueError for None path')
except ValueError as e:
    print(f'✓ Correctly caught ValueError for None path: {e}')

# Test invalid prompt
try:
    backend.generate('', None)
    print('ERROR: Should have raised ValueError for empty prompt')
except ValueError as e:
    print(f'✓ Correctly caught ValueError for empty prompt: {e}')

print('Error handling tests completed!')
