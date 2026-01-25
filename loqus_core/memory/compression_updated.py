"""
Memory Compression Interface

Prepares architecture for MELODI integration (v8.7+).
For MVP, only NoOpCompressor is implemented.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""
    content: str
    metadata: Dict[str, Any]
    role: str = "user"  # Add role field for better context
    timestamp: float = 0.0  # Add timestamp for ordering
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CompressedContext:
    """Result of context compression."""
    summary: str                    # Compressed text representation
    original_token_count: int       # Tokens before compression
    compressed_token_count: int     # Tokens after compression
    compression_ratio: float        # original / compressed
    preserved_message_ids: List[str] = None  # IDs of messages kept verbatim
    
    def __post_init__(self):
        if self.preserved_message_ids is None:
            self.preserved_message_ids = []


class MemoryCompressor(ABC):
    """
    Abstract base class for memory compression strategies.
    
    Compressors reduce context window usage while preserving critical information.
    This enables longer conversations without hitting token limits.
    
    Example:
        compressor = NoOpCompressor()
        compressed = compressor.compress(messages)
        ratio = compressor.get_compression_ratio()
    """
    
    @abstractmethod
    def compress(self, context: List[Message]) -> CompressedContext:
        """
        Compress a list of messages.
        
        Args:
            context: List of Message objects to compress
            
        Returns:
            CompressedContext with summary and metadata
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed: CompressedContext) -> List[Message]:
        """
        Decompress back to messages.
        
        Note: May be lossy - not all compressors can fully decompress.
        
        Args:
            compressed: CompressedContext to decompress
            
        Returns:
            List of Message objects (may be reconstructed)
        """
        pass
    
    @abstractmethod
    def get_compression_ratio(self) -> float:
        """
        Get the compression ratio from last operation.
        
        Returns:
            Ratio (compressed_tokens / original_tokens). 
            1.0 = no compression, >1.0 = compression achieved.
        """
        pass


class NoOpCompressor(MemoryCompressor):
    """
    No-operation compressor that passes through unchanged.
    
    This is the default for MVP. It ensures compression hooks
    exist without affecting behavior.
    """
    
    def __init__(self):
        self._last_ratio = 1.0
    
    def compress(self, context: List[Message]) -> CompressedContext:
        """Return context unchanged."""
        # Estimate token count (rough: 4 chars per token)
        total_chars = sum(len(m.content) for m in context)
        token_estimate = max(1, total_chars // 4)
        
        # Concatenate all content as "summary"
        summary = "\n".join(f"[{m.role}]: {m.content}" for m in context)
        
        self._last_ratio = 1.0  # No compression
        
        logger.debug(f"NoOpCompressor: {len(context)} messages, {token_estimate} tokens")
        
        return CompressedContext(
            summary=summary,
            original_token_count=token_estimate,
            compressed_token_count=token_estimate,
            compression_ratio=1.0,
        )
    
    def decompress(self, compressed: CompressedContext) -> List[Message]:
        """
        Attempt to decompress.
        
        Note: NoOp loses structure - returns single system message.
        """
        return [Message(
            content=f"[Reconstructed context]\n{compressed.summary}",
            metadata={},
            role="system",
            timestamp=0.0,
        )]
    
    def get_compression_ratio(self) -> float:
        return self._last_ratio

class MelodiCompressor(MemoryCompressor):
    """
    MELODI-based hierarchical compression.
    
    NOT IMPLEMENTED for MVP.
    See ADR-016 for architecture details.
    """
    
    def __init__(self):
        logger.warning("MelodiCompressor is not implemented for MVP")
    
    def compress(self, context: List[Message]) -> CompressedContext:
        raise NotImplementedError(
            "MELODI compression is planned for v8.7+. "
            "Use NoOpCompressor for MVP."
        )
    
    def decompress(self, compressed: CompressedContext) -> List[Message]:
        raise NotImplementedError("MelodiCompressor not implemented")
    
    def get_compression_ratio(self) -> float:
        return 1.0

def get_compressor(strategy: str = "noop") -> MemoryCompressor:
    """
    Factory function for memory compressors.
    
    Args:
        strategy: "noop" or "melodi"
        
    Returns:
        MemoryCompressor instance
    """
    compressors = {
        "noop": NoOpCompressor,
        "none": NoOpCompressor,
        "melodi": MelodiCompressor,
    }
    
    if strategy.lower() not in compressors:
        logger.error(f"Unknown compressor: {strategy}")
        raise ValueError(f"Unknown compressor: {strategy}")
    
    return compressors[strategy.lower()]()
