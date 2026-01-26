from loqus_core.inference.backend import LlamaCppBackend, PowerInferBackend, GenerateParams

# Test basic imports and instantiation
print('Testing basic imports...')

# Test GenerateParams
params = GenerateParams(max_tokens=100, temperature=0.7)
print(f'GenerateParams created: max_tokens={params.max_tokens}, temperature={params.temperature}')

# Test LlamaCppBackend instantiation
llama_backend = LlamaCppBackend()
print('LlamaCppBackend instantiated successfully')

# Test PowerInferBackend instantiation
power_backend = PowerInferBackend()
print('PowerInferBackend instantiated successfully')

print('All basic tests passed!')
