import math
from dataclasses import dataclass


@dataclass
class ModelArm:
    name: str
    count: int = 0
    total_reward: float = 0.0
    total_latency: float = 0.0

    def ucb1_score(self, total_pulls: int) -> float:
        if self.count == 0:
            return float('inf')
        
        if total_pulls <= 0:
            return float('inf')
        
        exploitation = self.total_reward / self.count
        exploration = math.sqrt(2 * math.log(total_pulls) / self.count)
        
        return exploitation + exploration

class BanditRouter:
    def __init__(self):
        self.arms = {}
        self.total_pulls = 0
    
    def route(self, request: dict) -> str:
        if not self.arms:
            return None
        
        best_arm = None
        best_score = float('-inf')
        
        for arm in self.arms.values():
            score = arm.ucb1_score(self.total_pulls)
            if score > best_score:
                best_score = score
                best_arm = arm
        
        return best_arm.name if best_arm else None
    
    def update_performance(self, arm_name: str, reward: float, latency: float):
        if reward < 0:
            raise ValueError("Reward cannot be negative")
        if latency < 0:
            raise ValueError("Latency cannot be negative")
        
        if arm_name not in self.arms:
            self.arms[arm_name] = ModelArm(name=arm_name)
        
        arm = self.arms[arm_name]
        arm.count += 1
        arm.total_reward += reward
        arm.total_latency += latency
        self.total_pulls += 1
    
    def get_all_stats(self) -> dict:
        stats = {}
        for arm_name, arm in self.arms.items():
            stats[arm_name] = {
                "count": arm.count,
                "total_reward": arm.total_reward,
                "average_reward": arm.total_reward / arm.count if arm.count > 0 else 0.0,
                "ucb1_score": arm.ucb1_score(self.total_pulls)
            }
        return stats