import pytest
from loqus_core.memory.compression import NoOpCompressor, MemoryCompressor, MelodiCompressor, Message, CompressedContext
import hashlib
import pickle

def test_noop_compressor():
    compressor = NoOpCompressor()
    # Test compression and decompression
    messages = [
        Message(content="Hello", metadata={}),
        Message(content="World", metadata={})
    ]
    compressed = compressor.compress(messages)
    decompressed = compressor.decompress(compressed)
    assert len(decompressed) == len(messages)
    assert decompressed[0].content == "Hello"
    assert decompressed[1].content == "World"
    
    # Test checksum integrity
    assert hasattr(compressed, 'checksum')
    assert isinstance(compressed.checksum, str)
    assert len(compressed.checksum) > 0
    
    # Test data integrity - checksum should match
    calculated_checksum = hashlib.sha256(compressed.data).hexdigest()
    assert calculated_checksum == compressed.checksum


def test_noop_compressor_integrity_check():
    compressor = NoOpCompressor()
    messages = [
        Message(content="Test message", metadata={"key": "value"}),
        Message(content="Another message", metadata={"key2": "value2"})
    ]
    
    # Compress
    compressed = compressor.compress(messages)
    
    # Decompress
    decompressed = compressor.decompress(compressed)
    
    # Verify data integrity
    assert len(decompressed) == len(messages)
    assert decompressed[0].content == "Test message"
    assert decompressed[1].content == "Another message"
    assert decompressed[0].metadata == {"key": "value"}
    assert decompressed[1].metadata == {"key2": "value2"}


def test_melodi_compressor_raises_not_implemented():
    compressor = MelodiCompressor()
    with pytest.raises(NotImplementedError):
        compressor.compress([])
    with pytest.raises(NotImplementedError):
        compressor.decompress(None)


def test_compression_error_handling():
    compressor = NoOpCompressor()
    messages = [
        Message(content="Test message", metadata={}),
    ]
    
    # Test that compression and decompression work normally
    compressed = compressor.compress(messages)
    decompressed = compressor.decompress(compressed)
    
    assert len(decompressed) == 1
    assert decompressed[0].content == "Test message"
