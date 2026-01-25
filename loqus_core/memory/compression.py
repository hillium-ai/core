import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict

# Define types for clarity
Message = Dict[str, Any]
CompressedContext = Any

class MemoryCompressor(ABC):
    """Abstract base class for memory compression interfaces."""
    
    @abstractmethod
    def compress(self, context: List[Message]) -> CompressedContext:
        """Compress a list of messages."""
        pass
    
    @abstractmethod
    def decompress(self, compressed: CompressedContext) -> List[Message]:
        """Decompress a compressed context back to list of messages."""
        pass

class NoOpCompressor(MemoryCompressor):
    """Default compressor that passes data through without modification."""
    
    def compress(self, context: List[Message]) -> CompressedContext:
        return context
    
    def decompress(self, compressed: CompressedContext) -> List[Message]:
        return compressed

class MelodiCompressor(MemoryCompressor):
    """Stub implementation for future MELODI integration."""
    
    def compress(self, context: List[Message]) -> CompressedContext:
        raise NotImplementedError("MelodiCompressor not yet implemented")
    
    def decompress(self, compressed: CompressedContext) -> List[Message]:
        raise NotImplementedError("MelodiCompressor not yet implemented")