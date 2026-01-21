import json
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelProfile:
    name: str
    size: str  # 'small', 'medium', 'large'
    speed: float  # 1-10 (higher is faster)
    accuracy: float  # 0-1
    cost: float  # arbitrary units per token
    available: bool = True

class CascadeRouter:
    """
    Implements the Hillium Cascade Routing logic:
    τ = q̂ - λc
    
    Where:
    - τ: Selection threshold
    - q̂: Query complexity estimate (0-1)
    - λ: System load/battery penalty
    - c: System state (resource availability)
    """
    
    def __init__(self, manifest_path: str = "models_manifest.json"):
        self.manifest_path = manifest_path
        self.models: Dict[str, ModelProfile] = {}
        self.load_manifest()
        
    def load_manifest(self):
        """Loads model profiles from JSON manifest."""
        try:
            # Fallback for relative vs absolute path
            path = self.manifest_path
            if not os.path.exists(path):
                # Try relative to the script location
                path = os.path.join(os.path.dirname(__file__), self.manifest_path)
                
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    for m in data.get("models", []):
                        profile = ModelProfile(**m)
                        self.models[profile.name] = profile
                logger.info(f"Loaded {len(self.models)} model profiles from {path}")
            else:
                logger.warning(f"Manifest not found at {path}. Using default profiles.")
                self._load_defaults()
        except Exception as e:
            logger.error(f"Error loading models manifest: {e}")
            self._load_defaults()

    def _load_defaults(self):
        """Standard default models if manifest missing."""
        self.models = {
            "tiny": ModelProfile("tiny", "small", 9.0, 0.6, 0.01),
            "base": ModelProfile("base", "medium", 5.0, 0.8, 0.05),
            "large": ModelProfile("large", "large", 2.0, 0.95, 0.20)
        }

    def route(self, query_complexity: float, battery_level: float = 1.0, cpu_load: float = 0.0) -> str:
        """
        Determines the target model based on complexity and system state.
        λ is calculated based on (1 - battery_level) + cpu_load.
        """
        # Calculate penalty λ
        load_penalty = (1.0 - battery_level) + (cpu_load * 0.5)
        
        # Selection Score τ
        # Higher score -> prefers larger models
        # Increased penalty weight to ensure battery state impacts routing
        selection_score = query_complexity - (0.5 * load_penalty)
        
        logger.debug(f"Routing logic: complexity={query_complexity:.2f}, penalty={load_penalty:.2f} -> τ={selection_score:.2f}")
        
        if selection_score > 0.7:
            return self._select_best_of_size("large")
        elif selection_score > 0.3:
            return self._select_best_of_size("medium")
        else:
            return self._select_best_of_size("small")

    def _select_best_of_size(self, size: str) -> str:
        """Finds first available model of requested size."""
        for name, profile in self.models.items():
            if profile.size == size and profile.available:
                return name
        
        # Fallback logic: if requested size unavailable, try smaller
        fallbacks = {"large": "medium", "medium": "small", "small": "tiny"}
        if size in fallbacks:
            return self._select_best_of_size(fallbacks[size])
        
        # Ultimate fallback
        return next(iter(self.models.keys()))

if __name__ == "__main__":
    # Self-test
    router = CascadeRouter()
    print(f"Low complexity (0.2): {router.route(0.2)}")
    print(f"High complexity (0.9): {router.route(0.9)}")
    print(f"High complexity (0.9) @ 10% battery: {router.route(0.9, battery_level=0.1)}")
