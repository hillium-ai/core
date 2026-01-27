import math
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ModelArm:
    """
    Represents a model/role as a "bandit arm" with tracking statistics.
    """
    name: str
    size: str  # 'small', 'medium', 'large'
    speed: float  # 1-10 (higher is faster)
    accuracy: float  # 0-1
    cost: float  # arbitrary units per token
    available: bool = True
    
    # Bandit statistics
    count: int = 0
    total_reward: float = 0.0
    total_latency: float = 0.0
    latency_samples: int = 0
    
    def mean_reward(self) -> float:
        return self.total_reward / self.count if self.count > 0 else 0.0
    
    def average_latency(self) -> float:
        return self.total_latency / self.latency_samples if self.latency_samples > 0 else 0.0
    
    def ucb1_score(self, total_pulls: int) -> float:
        """
        Calculate UCB1 score for this arm.
        
        UCB1 formula: score = mean_reward + sqrt(2 * ln(total_pulls) / count)
        """
        if self.count == 0:
            return float('inf')  # Explore untried arms
        
        try:
            exploitation = self.mean_reward()
            exploration = math.sqrt(2 * math.log(total_pulls) / self.count)
            return exploitation + exploration
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Error calculating UCB1 score for {self.name}: {e}")
            # Fallback to mean reward if calculation fails
            return self.mean_reward()


class BanditRouter:
    """
    Implements a Multi-Armed Bandit (UCB1) router that dynamically selects
    the optimal model/role based on historical performance.
    """
    
    def __init__(self, manifest_path: str = "models_manifest.json", config: Optional[Dict] = None):
        self.manifest_path = manifest_path
        self.models: Dict[str, ModelArm] = {}
        self.total_pulls = 0
        self.config = config or {}
        self.load_manifest()
        
        # Manual overrides
        self.manual_overrides = self.config.get("manual_overrides", {})
        
        logger.info(f"Initialized BanditRouter with {len(self.models)} models")
    
    def load_manifest(self):
        """
        Loads model profiles from JSON manifest and converts them to ModelArm objects.
        """
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
                        arm = ModelArm(**m)
                        self.models[arm.name] = arm
                logger.info(f"Loaded {len(self.models)} model arms from {path}")
            else:
                logger.warning(f"Manifest not found at {path}. Using default profiles.")
                self._load_defaults()
        except Exception as e:
            logger.error(f"Error loading models manifest: {e}")
            self._load_defaults()

    def _load_defaults(self):
        """
        Standard default models if manifest missing.
        """
        default_models = [
            {
                "name": "tiny-llama",
                "size": "small",
                "speed": 9.5,
                "accuracy": 0.55,
                "cost": 0.005,
                "available": True
            },
            {
                "name": "qwen2-7b",
                "size": "medium",
                "speed": 6.0,
                "accuracy": 0.82,
                "cost": 0.05,
                "available": True
            },
            {
                "name": "mistral-large",
                "size": "large",
                "speed": 2.5,
                "accuracy": 0.94,
                "cost": 0.15,
                "available": True
            }
        ]
        
        for m in default_models:
            arm = ModelArm(**m)
            self.models[arm.name] = arm
    
    def route(self, query_complexity: float, query: str = "", **kwargs) -> str:
        """
        Determines the target model based on UCB1 algorithm.
        
        Args:
            query_complexity: Estimated complexity of the query (0-1)
            query: The actual query text for potential analysis
            **kwargs: Additional parameters (e.g., latency, success metrics)
            
        Returns:
            Name of the selected model
        """
        # Check for manual override
        if self.manual_overrides:
            # Check if there's a specific override for this query complexity
            for complexity_range, model_name in self.manual_overrides.items():
                if complexity_range == "default" or self._is_in_range(query_complexity, complexity_range):
                    logger.debug(f"Using manual override: {model_name} for complexity {query_complexity}")
                    return model_name
        
        # If no valid models available
        available_models = [arm for arm in self.models.values() if arm.available]
        if not available_models:
            logger.warning("No available models found, using fallback")
            return self._select_fallback_model()
        
        # Calculate UCB1 scores for all available models
        scores = {}
        for arm in available_models:
            scores[arm.name] = arm.ucb1_score(self.total_pulls)
            
        # Select model with highest UCB1 score
        selected_model = max(scores, key=scores.get)
        
        # Increment total pulls
        self.total_pulls += 1
        
        logger.debug(f"Selected model {selected_model} with UCB1 score {scores[selected_model]:.4f}")
        return selected_model
    
    def _is_in_range(self, value: float, range_str: str) -> bool:
        """
        Check if a value is within a range string like "0.0-0.3" or "0.7-1.0"
        """
        try:
            if "-" in range_str:
                start, end = map(float, range_str.split("-"))
                return start <= value <= end
            else:
                return value == float(range_str)
        except ValueError:
            return False
    
    def _select_fallback_model(self) -> str:
        """
        Select a fallback model when no models are available or when UCB1 fails.
        """
        # Try to find any available model
        for name, arm in self.models.items():
            if arm.available:
                return name
        
        # If no available models, return first one
        return next(iter(self.models.keys()))
    
    def update_performance(self, model_name: str, reward: float, latency: float = 0.0, success: bool = True):
        """
        Update performance statistics for a model after a selection.
        
        Args:
            model_name: Name of the model
            reward: Reward received (e.g., from soft scoring framework)
            latency: Latency of the model response
            success: Whether the model execution was successful
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found in router")
            return
        
        arm = self.models[model_name]
        
        # Update bandit statistics
        arm.count += 1
        arm.total_reward += reward
        
        # Update latency statistics
        if latency > 0:
            arm.total_latency += latency
            arm.latency_samples += 1
        
        logger.debug(f"Updated performance for {model_name}: reward={reward:.4f}, latency={latency:.4f}")
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific model.
        """
        if model_name not in self.models:
            return {}
        
        arm = self.models[model_name]
        return {
            "name": arm.name,
            "count": arm.count,
            "total_reward": arm.total_reward,
            "mean_reward": arm.mean_reward(),
            "average_latency": arm.average_latency(),
            "ucb1_score": arm.ucb1_score(self.total_pulls) if arm.count > 0 else 0.0
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all models.
        """
        return {name: self.get_model_stats(name) for name in self.models.keys()}

# For testing purposes
if __name__ == "__main__":
    # Self-test
    router = BanditRouter()
    print(f"Initial routing: {router.route(0.2)}")
    print(f"Initial routing: {router.route(0.9)}")
    
    # Test performance updates
    router.update_performance("tiny-llama", reward=0.8, latency=0.1)
    router.update_performance("qwen2-7b", reward=0.9, latency=0.2)
    
    print(f"After updates: {router.route(0.5)}")