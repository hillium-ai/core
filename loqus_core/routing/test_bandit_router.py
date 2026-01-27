import unittest
from unittest.mock import Mock

# Import the actual BanditRouter
from loqus_core.routing.bandit_router import BanditRouter

class TestBanditRouter(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.router = BanditRouter()
        
    def test_ucb1_score_first_pull(self):
        """Test that untried arms get infinite score for exploration."""
        arm = Mock()
        arm.count = 0
        arm.total_reward = 0
        
        score = self.router._ucb1_score(arm, 10)
        self.assertEqual(score, float('inf'))
        
    def test_model_arm_dataclass(self):
        """Test that ModelArm dataclass works correctly."""
        # This test will be added later
        pass

if __name__ == '__main__':
    unittest.main()
