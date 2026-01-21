# 🧠 DataAggregator for Multi-Source Collection

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import sleep
from random import uniform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    instruction: str
    response: str
    source: str
    metadata: Dict[str, Any] = None


class VectorDBInterface(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    def query(self, filter_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass


class QDrantClient(VectorDBInterface):
    """Concrete implementation for QDrant."""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        # Simulate connection
        logger.info(f"Connected to QDrant at {host}:{port}")

    def query(self, filter_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Simulate QDrant query with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # In a real implementation, this would call QDrant API
                logger.info(f"Querying QDrant with filter: {filter_conditions}")
                # Simulated result
                return [
                    {
                        "instruction": "Simulated instruction from QDrant",
                        "response": "Simulated response from QDrant",
                        "source": filter_conditions.get("source", "unknown"),
                        "metadata": {"timestamp": "2023-01-01T00:00:00Z", "source_type": filter_conditions.get("source", "unknown")}
                    }
                ]
            except Exception as e:
                logger.warning(f"QDrant query failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt + uniform(0, 1))  # Exponential backoff
                else:
                    raise


class DataAggregator:
    """Aggregates memories from multiple sources and exports them in a balanced format."""

    def __init__(self, db_client: VectorDBInterface = None):
        self.db_client = db_client or QDrantClient()
        self.ratio_real = float(os.getenv("DATASET_RATIO_REAL", "0.7"))
        self.ratio_synthetic = float(os.getenv("DATASET_RATIO_SYNTHETIC", "0.3"))
        self.sources = [
            "real_world",
            "synthetic",
            "human_correction",
            "failure_logs"
        ]
        # Validate ratios
        if abs((self.ratio_real + self.ratio_synthetic) - 1.0) > 1e-6:
            logger.warning("Dataset ratios do not sum to 1.0. Using default 0.7/0.3 split.")
            self.ratio_real = 0.7
            self.ratio_synthetic = 0.3

    def _validate_memory(self, memory: Dict[str, Any]) -> bool:
        """Validate that a memory entry has required fields."""
        required_fields = ["instruction", "response", "source"]
        for field in required_fields:
            if not memory.get(field):
                logger.warning(f"Memory missing required field: {field}")
                return False
        # Validate that instruction and response are not empty strings
        if not str(memory["instruction"]).strip() or not str(memory["response"]).strip():
            logger.warning("Memory has empty instruction or response")
            return False
        return True

    def _sanitize_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or sanitize sensitive information."""
        # Example: Remove or redact sensitive data
        sanitized = memory.copy()
        if "metadata" in sanitized:
            # Remove or redact sensitive metadata fields
            sensitive_fields = ["user_id", "email", "phone", "password"]
            for field in sensitive_fields:
                if isinstance(sanitized["metadata"], dict):
                    sanitized["metadata"].pop(field, None)
        # Sanitize text fields
        if "instruction" in sanitized:
            sanitized["instruction"] = str(sanitized["instruction"]).strip()
        if "response" in sanitized:
            sanitized["response"] = str(sanitized["response"]).strip()
        return sanitized

    def _query_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query a specific source for memories with error handling."""
        filter_conditions = {"source": source}
        try:
            results = self.db_client.query(filter_conditions)
            logger.info(f"Retrieved {len(results)} entries from {source}")
            return results
        except Exception as e:
            logger.error(f"Failed to query {source}: {e}")
            return []

    def export_lora_dataset(self, output_path: str = "data/exports/lora_training.jsonl") -> bool:
        """Export a balanced dataset to JSONL format with error handling and validation."""
        logger.info("Starting export of LORA dataset")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get all memories from sources
        all_memories = []
        for source in self.sources:
            memories = self._query_source(source)
            for mem in memories:
                if self._validate_memory(mem):
                    sanitized_mem = self._sanitize_memory(mem)
                    all_memories.append(sanitized_mem)

        # Balance dataset
        balanced_memories = self._balance_dataset(all_memories)

        # Export to JSONL
        try:
            with open(output_path, "w") as f:
                for mem in balanced_memories:
                    f.write(json.dumps(mem) + "\n")
            logger.info(f"Exported {len(balanced_memories)} entries to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write to {output_path}: {e}")
            return False

    def _balance_dataset(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance dataset according to configured ratios."""
        logger.info("Balancing dataset")
        
        # Separate by source
        real_memories = [m for m in memories if m.get("source") == "real_world"]
        synthetic_memories = [m for m in memories if m.get("source") == "synthetic"]
        
        # Calculate target numbers
        total_memories = len(memories)
        target_real = int(total_memories * self.ratio_real)
        target_synthetic = int(total_memories * self.ratio_synthetic)
        
        # Sample from each source
        import random
        balanced = []
        
        # Add real memories
        real_sample = random.sample(real_memories, min(target_real, len(real_memories))) if real_memories else []
        balanced.extend(real_sample)
        
        # Add synthetic memories
        synthetic_sample = random.sample(synthetic_memories, min(target_synthetic, len(synthetic_memories))) if synthetic_memories else []
        balanced.extend(synthetic_sample)
        
        logger.info(f"Dataset balanced: {len(real_sample)} real, {len(synthetic_sample)} synthetic")
        return balanced


# Example usage
if __name__ == "__main__":
    aggregator = DataAggregator()
    success = aggregator.export_lora_dataset()
    if success:
        logger.info("Dataset export completed successfully")
    else:
        logger.error("Dataset export failed")
