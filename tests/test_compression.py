import unittest
from loqus_core.memory.compression import Message, CompressedContext, NoOpCompressor, get_compressor

class TestCompression(unittest.TestCase):
    
    def test_message_dataclass(self):
        # Test that Message can be instantiated with all required fields
        msg = Message(role="user", content="test content", timestamp=123.0)
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "test content")
        self.assertEqual(msg.timestamp, 123.0)
        self.assertEqual(msg.metadata, None)
        
        # Test with metadata
        msg_with_metadata = Message(role="assistant", content="test", timestamp=456.0, metadata={"key": "value"})
        self.assertEqual(msg_with_metadata.metadata, {"key": "value"})
    
    def test_compressed_context_dataclass(self):
        # Test that CompressedContext can be instantiated
        ctx = CompressedContext(
            summary="compressed data",
            original_token_count=100,
            compressed_token_count=50,
            compression_ratio=0.5
        )
        self.assertEqual(ctx.summary, "compressed data")
        self.assertEqual(ctx.original_token_count, 100)
        self.assertEqual(ctx.compressed_token_count, 50)
        self.assertEqual(ctx.compression_ratio, 0.5)
        self.assertEqual(ctx.preserved_message_ids, None)
        
        # Test with preserved_message_ids
        ctx_with_ids = CompressedContext(
            summary="compressed data",
            original_token_count=100,
            compressed_token_count=50,
            compression_ratio=0.5,
            preserved_message_ids=["msg1", "msg2"]
        )
        self.assertEqual(ctx_with_ids.preserved_message_ids, ["msg1", "msg2"])
    
    def test_noop_compressor(self):
        # Test NoOpCompressor functionality
        compressor = NoOpCompressor()
        
        # Create test messages
        messages = [
            Message(role="user", content="Hello", timestamp=1.0),
            Message(role="assistant", content="Hi there", timestamp=2.0)
        ]
        
        # Compress
        compressed = compressor.compress(messages)
        
        # Check that compression worked
        self.assertIsInstance(compressed, CompressedContext)
        self.assertEqual(compressed.original_token_count, 12)  # Approximate
        self.assertEqual(compressed.compressed_token_count, 12)  # Approximate
        self.assertEqual(compressed.compression_ratio, 1.0)  # No compression
        
        # Decompress
        decompressed = compressor.decompress(compressed)
        
        # Check that decompression worked (though structure is lost)
        self.assertIsInstance(decompressed, list)
        self.assertEqual(len(decompressed), 1)
        
        # Check ratio
        ratio = compressor.get_compression_ratio()
        self.assertEqual(ratio, 1.0)
    
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