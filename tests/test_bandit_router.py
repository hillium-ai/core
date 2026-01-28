import unittest
from unittest.mock import Mock
import math

# Import the actual BanditRouter
from loqus_core.routing.bandit_router import BanditRouter, ModelArm

class TestBanditRouter(unittest.TestCase):
    
    def test_model_arm_dataclass(self):
        """Test that ModelArm dataclass works correctly with all required fields."""
        arm = ModelArm(name="test_model", size="small", speed=5.0, accuracy=0.8, cost=0.5)
        self.assertEqual(arm.name, "test_model")
        self.assertEqual(arm.size, "small")
        self.assertEqual(arm.speed, 5.0)
        self.assertEqual(arm.accuracy, 0.8)
        self.assertEqual(arm.cost, 0.5)
        
    def test_model_arm_ucb1_score_first_pull(self):
        """Test that untried arms get infinite score for exploration."""
        # Create a ModelArm with zero count
        arm = ModelArm(name="test_model", size="small", speed=5.0, accuracy=0.8, cost=0.5)
        arm.count = 0
        arm.total_reward = 0.0
        
        # Test the ucb1_score method directly on the arm
        score = arm.ucb1_score(10)
        self.assertEqual(score, float('inf'))
        
    def test_model_arm_ucb1_score_with_rewards(self):
        """Test UCB1 calculation with actual rewards."""
        # Create a ModelArm with some rewards
        arm = ModelArm(name="test_model", size="small", speed=5.0, accuracy=0.8, cost=0.5)
        arm.count = 5
        arm.total_reward = 3.0
        
        # Test the ucb1_score method directly on the arm
        score = arm.ucb1_score(10)
        
        # Expected: 0.6 + sqrt(2 * ln(10) / 5)
        exploitation = 3.0 / 5  # 0.6
        exploration = math.sqrt(2 * math.log(10) / 5)
        expected = exploitation + exploration
        self.assertAlmostEqual(score, expected, places=5)
        
    def test_router_instantiation(self):
        """Test that BanditRouter can be instantiated."""
        router = BanditRouter()
        self.assertIsNotNone(router)
        
    def test_ucb1_score_first_pull(self):
        """Test that router's _ucb1_score method works for untried arms."""
        router = BanditRouter()
        arm = ModelArm(name="test_model", size="small", speed=5.0, accuracy=0.8, cost=0.5)
        arm.count = 0
        arm.total_reward = 0.0
        
        score = router._ucb1_score(arm, 10)
        self.assertEqual(score, float('inf'))
        
    def test_ucb1_score_with_rewards(self):
        """Test router's _ucb1_score method with actual rewards."""
        router = BanditRouter()
        arm = ModelArm(name="test_model", size="small", speed=5.0, accuracy=0.8, cost=0.5)
        arm.count = 5
        arm.total_reward = 3.0
        
        score = router._ucb1_score(arm, 10)
        
        # Expected: 0.6 + sqrt(2 * ln(10) / 5)
        exploitation = 3.0 / 5  # 0.6
        exploration = math.sqrt(2 * math.log(10) / 5)
        expected = exploitation + exploration
        self.assertAlmostEqual(score, expected, places=5)
