import sys
sys.path.insert(0, '/workspace')

try:
    from loqus_core.inference.backend import LlamaCppBackend, PowerInferBackend, get_backend
    print('All imports successful')
    
    # Test the factory
    backend = get_backend('powerinfer')
    print('PowerInfer backend created successfully')
    
    # Test that it's the right type
    print(f'Backend type: {type(backend)}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()