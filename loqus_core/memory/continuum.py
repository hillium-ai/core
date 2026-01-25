"""
Continuum Memory System (CMS) - v8.0.0
Orchestrates 4-level memory hierarchy with different update frequencies.
"""

from typing import List, Dict, Any, Optional
import logging
import hashlib
import pickle

from loqus_core.hippo import HippoLink, AssociativeCoreHandle
from loqus_core.memory.compression import get_compressor, Message, CompressedContext

logger = logging.getLogger(__name__)


class ContinuumMemorySystem:
    """
    Orchestrates access to 4 memory levels.

    Based on ADR-005 (Nested Learning) - different frequencies:
    - Level 1 (Sensory): 10kHz+ updates
    - Level 2 (Working): 10Hz updates
    - Level 2.5 (Associative): 1-10Hz updates
    - Level 3 (Episodic): 0.01Hz updates (consolidation)
    """

    def __init__(self, hippo: HippoLink):
        self.hippo = hippo

        # Level 2.5: Associative Core
        self.associative_core = None # Will be initialized via HippoLink

        # Level 3: Episodic (deferred in MVP)
        self.episodic_enabled = False

        # Compression support
        self.compressor = get_compressor()

        logger.info("ContinuumMemorySystem initialized (4 levels)")

    # === LEVEL 1: SENSORY BUFFER ===

    def read_sensory(self, buffer_type: str = "conversation") -> str:
        """
        Read from Level 1 (Sensory Buffer).

        Args:
            buffer_type: "conversation", "audio", "telemetry"

        Returns:
            Buffer contents
        """
        if buffer_type == "conversation":
            return self.hippo.read_conversation()
        elif buffer_type == "telemetry":
            return self.hippo.get_telemetry()
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")

    def write_sensory(self, text: str):
        """Write to Level 1 conversation buffer."""
        self.hippo.write_conversation(text)

    # === LEVEL 2: WORKING MEMORY ===

    def store_note(self, note_id: str, content: str, domain: str):
        """
        Store in Level 2 (Working Memory - Sled DB).
        Persists across restarts, GC after hours/days.
        """
        self.hippo.store_note(note_id, content, domain)

    def retrieve_note(self, note_id: str) -> str:
        """Retrieve from Level 2."""
        return self.hippo.get_note(note_id)

    def query_working_memory(self, domain: str, limit: int = 10) -> List[Dict]:
        """
        Query Level 2 by domain.

        Returns recent notes for context building.
        """
        # For MVP: return empty if not implemented in HippoServer
        return []

    # === LEVEL 2.5: ASSOCIATIVE CORE ===

    def learn_from_feedback(
        self,
        context: List[float],
        target: List[float]
    ):
        """
        Update Level 2.5 (Associative Core) with feedback.
        Fast weights updated via delta-rule (<10µs).
        """
        if hasattr(self.hippo, 'learn_associative'):
             self.hippo.learn_associative(context, target)

    def predict_from_core(self, context: List[float]) -> List[float]:
        """
        Predict using Level 2.5 fast weights (<1ms).
        100x faster than Level 3 vector search.
        """
        if hasattr(self.hippo, 'predict_associative'):
            return self.hippo.predict_associative(context)
        return []

    # === LEVEL 3: EPISODIC STORE ===

    def consolidate_to_episodic(self):
        """
        Consolidate Level 2.5 → Level 3 (async).
        Deferred in MVP.
        """
        if not self.episodic_enabled:
            logger.debug("Episodic memory disabled in MVP")
            return

        # TODO: Implement consolidation pipeline
        logger.info("Consolidation triggered (stub implementation)")

    # === MULTI-LEVEL QUERIES ===

    def query_multi_level(
        self,
        query: str,
        levels: List[int] = [1, 2, 2.5]
    ) -> Dict[str, Any]:
        """
        Query across multiple memory levels.

        Args:
            query: Search query
            levels: List of levels to query (1, 2, 2.5, 3)

        Returns:
            Dictionary with results from each level
        """
        results = {}

        if 1 in levels:
            # Level 1: Sensory buffer
            results["level1"] = self.read_sensory()

        if 2 in levels:
            # Level 2: Working memory
            results["level2"] = []
        return results

    # === COMPRESSION HOOKS ===

    def compress_context(self, messages: List[Message]) -> CompressedContext:
        """
        Compress a list of messages using the configured compressor.
        
        Args:
            messages: List of Message objects to compress
            
        Returns:
            CompressedContext object
        """
        try:
            # Validate input
            if not isinstance(messages, list):
                raise TypeError("Messages must be a list")
            
            # Quick check to avoid unnecessary compression
            if len(messages) == 0:
                # Return empty context
                return CompressedContext(
                    data=b'',
                    metadata={"compression_type": "none", "message_count": 0},
                    checksum=""
                )
            
            compressed = self.compressor.compress(messages)
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return original content if compression fails
            # Create a minimal compressed context with original data
            try:
                serialized = pickle.dumps(messages)
                checksum = hashlib.sha256(serialized).hexdigest()
                return CompressedContext(
                    data=serialized,
                    metadata={"compression_type": "none", "error": str(e), "error_type": type(e).__name__, "fallback": True},
                    checksum=checksum
                )
            except Exception as e2:
                logger.error(f"Failed to create fallback compressed context: {e2}")
                # Return minimal fallback
                return CompressedContext(
                    data=b'',
                    metadata={"compression_type": "none", "error": str(e2), "fallback": True},
                    checksum=""
                )

    def decompress_context(self, compressed_context: CompressedContext) -> List[Message]:
        """
        Decompress a compressed context back to messages.
        
        Args:
            compressed_context: CompressedContext object
            
        Returns:
            List of Message objects
        """
        try:
            # Validate input
            if not isinstance(compressed_context, CompressedContext):
                raise TypeError("Invalid compressed context type")
            
            # Quick check for empty data
            if not compressed_context.data:
                logger.debug("Empty compressed data, returning empty list")
                return []
            
            result = self.compressor.decompress(compressed_context)
            return result
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            # Return empty list if decompression fails
            # Try to provide more context about the error
            try:
                # Attempt to deserialize the data directly for debugging
                if hasattr(compressed_context, 'data') and compressed_context.data:
                    logger.debug(f"Attempting to deserialize raw data: {len(compressed_context.data)} bytes")
            except Exception as debug_e:
                logger.debug(f"Debug deserialization failed: {debug_e}")
            
            return []

