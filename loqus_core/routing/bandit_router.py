import math
from dataclasses import dataclass
from typing import List, Optional


class ModelArm:
    def __init__(self, name: str, size: str, speed: float, accuracy: float, cost: float, count: int = 0, total_reward: float = 0.0):
        self.name = name
        self.size = size
        self.speed = speed
        self.accuracy = accuracy
        self.cost = cost
        self.count = count
        self.total_reward = total_reward

    def ucb1_score(self, total_pulls: int) -> float:
        """
        Calculate UCB1 score for this arm.
        
        Args:
            total_pulls: Total number of pulls so far
        
        Returns:
            UCB1 score for this arm
        """
        if self.count == 0:
            return float('inf')  # Explore untried arms
        
        # Handle potential edge cases
        if total_pulls <= 0:
            return float('inf')  # If no pulls yet, explore all
        
        exploitation = self.total_reward / self.count
        exploration = math.sqrt(2 * math.log(total_pulls) / self.count)
        
        return exploitation + exploration


class BanditRouter:
    def __init__(self):
        self.arms = []
        self.total_pulls = 0

    def _ucb1_score(self, arm: ModelArm, total_pulls: int) -> float:
        """
        Calculate UCB1 score for a given arm.
        
        Args:
            arm: The ModelArm to score
            total_pulls: Total number of pulls so far
        
        Returns:
            UCB1 score for the arm
        """
        if arm.count == 0:
            return float('inf')  # Explore untried arms
        
        # Handle potential edge cases
        if total_pulls <= 0:
            return float('inf')  # If no pulls yet, explore all
        
        exploitation = arm.total_reward / arm.count
        exploration = math.sqrt(2 * math.log(total_pulls) / arm.count)
        
        return exploitation + exploration

    def select_arm(self, arms: List[ModelArm]) -> ModelArm:
        """
        Select the best arm using UCB1 algorithm.
        
        Args:
            arms: List of available ModelArms
        
        Returns:
            The selected ModelArm
        """
        if not arms:
            raise ValueError("No arms available")
        
        best_arm = None
        best_score = float('-inf')
        
        for arm in arms:
            score = self._ucb1_score(arm, self.total_pulls)
            if score > best_score:
                best_score = score
                best_arm = arm
        
        return best_arm

    def route(self, confidence_threshold: float = 0.5) -> str:
        """
        Route to the best model based on UCB1 scores.
        
        Args:
            confidence_threshold: Threshold for model selection confidence
        
        Returns:
            Name of the selected model
        """
        if not self.arms:
            return None
        
        best_arm = self.select_arm(self.arms)
        return best_arm.name

    def update_performance(self, model_name: str, reward: float, latency: float = 0.0):
        """
        Update the performance metrics for a specific model.
        
        Args:
            model_name: Name of the model
            reward: Reward received (e.g., accuracy)
            latency: Latency of the model response
        """
        # Find the arm with the given name
        for arm in self.arms:
            if arm.name == model_name:
                arm.count += 1
                arm.total_reward += reward
                self.total_pulls += 1
                break
        else:
            # If the model doesn't exist, create a new arm
            new_arm = ModelArm(
                name=model_name,
                size="unknown",
                speed=latency,
                accuracy=reward,
                cost=0.0
            )
            new_arm.count = 1
            new_arm.total_reward = reward
            self.arms.append(new_arm)
            self.total_pulls += 1

    def get_all_stats(self) -> dict:
        """
        Get statistics for all models.
        
        Returns:
            Dictionary containing statistics for all models
        """
        stats = {}
        for arm in self.arms:
            stats[arm.name] = {
                "count": arm.count,
                "total_reward": arm.total_reward,
                "average_reward": arm.total_reward / arm.count if arm.count > 0 else 0.0,
                "ucb1_score": arm.ucb1_score(self.total_pulls)
            }
        return stats
