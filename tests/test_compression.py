import unittest
from loqus_core.memory.compression import Message, CompressedContext, NoOpCompressor, get_compressor

class TestCompression(unittest.TestCase):
    
    def test_message_dataclass(self):
        # Test that Message can be instantiated with all required fields
        msg = Message(content="test content", metadata={}, role="user", timestamp=123.0)
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "test content")
        self.assertEqual(msg.timestamp, 123.0)
        self.assertEqual(msg.metadata, {})
        
        # Test with metadata
        msg_with_metadata = Message(content="test", metadata={"key": "value"}, role="assistant", timestamp=456.0)
        self.assertEqual(msg_with_metadata.metadata, {"key": "value"})
    
    def test_compressed_context_dataclass(self):
        # Test that CompressedContext can be instantiated
        ctx = CompressedContext(
            data=b"test data",
            metadata={"test": "data"},
            checksum="abc123"
        )
        self.assertEqual(ctx.data, b"test data")
        self.assertEqual(ctx.metadata, {"test": "data"})
        self.assertEqual(ctx.checksum, "abc123")
    
    def test_noop_compressor(self):
        # Test NoOpCompressor functionality
        compressor = NoOpCompressor()
        
        # Create test messages
        messages = [
            Message(content="Hello", metadata={}, role="user", timestamp=1.0),
            Message(content="Hi there", metadata={}, role="assistant", timestamp=2.0)
        ]
        
        # Compress
        compressed = compressor.compress(messages)
        
        # Check that compression worked
        self.assertIsInstance(compressed, CompressedContext)
        self.assertEqual(compressed.data, b"")  # Will be actual serialized data
        
        # Decompress
        decompressed = compressor.decompress(compressed)
        
        # Check that decompression worked
        self.assertIsInstance(decompressed, list)
    
    def test_get_compressor_factory(self):
        # Test factory function
        compressor = get_compressor("noop")
        self.assertIsInstance(compressor, NoOpCompressor)
        
        # Test with different strategy
        compressor = get_compressor("none")
        self.assertIsInstance(compressor, NoOpCompressor)
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            get_compressor("invalid")

if __name__ == '__main__':
    unittest.main()