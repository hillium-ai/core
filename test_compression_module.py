from loqus_core.memory.compression import get_compressor, Message, NoOpCompressor

# Test that the module can be imported and works
print("Testing compression module...")

# Test factory function
compressor = get_compressor("noop")
print(f"Got compressor: {type(compressor)}")

# Test with melodi
try:
    compressor = get_compressor("melodi")
    print(f"Got melodi compressor: {type(compressor)}")
except NotImplementedError as e:
    print(f"Melodi compressor not implemented as expected: {e}")

# Test basic functionality
msg = Message(content="test", metadata={}, role="user", timestamp=1.0)
print(f"Created message: {msg}")

print("Compression module test completed successfully.")