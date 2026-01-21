import logging
import math
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class CognitiveDepthCalculator:
    """
    Implements Neuro-Fibonacci logic for determining reasoning depth.
    Maps uncertainty (σ) and complexity (C) to Fibonacci levels [1, 2, 3, 5].
    """
    
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        # Fibonacci steps: 1 (Atomic), 2 (Reflective), 3 (Recursive), 5 (Deep)
        self.levels = [1, 2, 3, 5]

    def calculate(self, query: str, uncertainty_score: float = 0.0, retry_count: int = 0) -> int:
        """
        Computes required depth (τ).
        Logic: τ = ceil(fib_index * (1 + uncertainty_score + (retry_count * 0.5)))
        """
        # Base complexity factor based on query length and keywords
        complexity = min(1.0, (len(query.split()) / 50.0))
        if any(word in query.lower() for word in ["analyze", "compare", "why", "how"]):
            complexity = min(1.0, complexity + 0.3)
            
        # Combine factors
        raw_score = complexity + uncertainty_score + (retry_count * 0.3)
        
        # Map to Fibonacci index (0-3)
        index = 0
        if raw_score > 0.8:
            index = 3 # Depth 5
        elif raw_score > 0.5:
            index = 2 # Depth 3
        elif raw_score > 0.2:
            index = 1 # Depth 2
            
        final_depth = self.levels[index]
        logger.info(f"Depth calculation: score={raw_score:.2f} (complexity={complexity:.2f}, uncertainty={uncertainty_score:.2f}) -> Depth {final_depth}")
        
        return final_depth

def determine_cognitive_depth(query: str, state: Optional[Dict[str, Any]] = None) -> int:
    """Wrapper for global usage."""
    calc = CognitiveDepthCalculator()
    uncertainty = state.get("uncertainty", 0.0) if state else 0.0
    retries = state.get("retries", 0) if state else 0
    return calc.calculate(query, uncertainty, retries)

if __name__ == "__main__":
    # Self-test
    print(f"Simple query: {determine_cognitive_depth('Hello')}")
    print(f"Complex analysis: {determine_cognitive_depth('Analyze the socio-economic impacts of GGUF quantization on Jetson devices.')}")
