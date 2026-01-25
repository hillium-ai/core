# tests/test_compression.py
import pytest
from loqus_core.memory.compression import (
    NoOpCompressor, MelodiCompressor, Message, get_compressor
)

def test_noop_passthrough():
    """NoOp should not change token count."""
    compressor = NoOpCompressor()
    messages = [
        Message(role="user", content="Hello", timestamp=1.0),
        Message(role="assistant", content="Hi there", timestamp=2.0),
    ]
    
    result = compressor.compress(messages)
    assert result.compression_ratio == 1.0


def test_melodi_not_implemented():
    """MELODI should raise NotImplementedError."""
    compressor = MelodiCompressor()
    
    with pytest.raises(NotImplementedError):
        compressor.compress([])


def test_factory():
    """Factory should return correct compressor."""
    assert isinstance(get_compressor("noop",), NoOpCompressor)
    assert isinstance(get_compressor("none"), NoOpCompressor)
    assert isinstance(get_compressor("melodi"), MelodiCompressor)
