import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'loqus_core'))

# Import and test the bandit router implementation
from loqus_core.routing.bandit_router import BanditRouter, ModelArm
import math

print('Testing BanditRouter implementation...')

# Test 1: ModelArm dataclass
arm = ModelArm(name='test_model', size='small', speed=5.0, accuracy=0.8, cost=0.5)
assert arm.name == 'test_model'
assert arm.size == 'small'
assert arm.speed == 5.0
assert arm.accuracy == 0.8
assert arm.cost == 0.5
print('✅ ModelArm dataclass works correctly')

# Test 2: UCB1 score for untried arm
arm.count = 0
arm.total_reward = 0.0
score = arm.ucb1_score(10)
assert score == float('inf')
print('✅ UCB1 score for untried arm is infinity')

# Test 3: UCB1 score with rewards
arm.count = 5
arm.total_reward = 3.0
score = arm.ucb1_score(10)
exploitation = 3.0 / 5
exploration = math.sqrt(2 * math.log(10) / 5)
expected = exploitation + exploration
assert abs(score - expected) < 1e-5
print('✅ UCB1 score with rewards calculated correctly')

# Test 4: BanditRouter instantiation
router = BanditRouter()
assert router is not None
print('✅ BanditRouter can be instantiated')

# Test 5: Router UCB1 score calculation
arm = ModelArm(name='test_model', size='small', speed=5.0, accuracy=0.8, cost=0.5)
arm.count = 0
arm.total_reward = 0.0
score = router._ucb1_score(arm, 10)
assert score == float('inf')
print('✅ Router UCB1 score for untried arm is infinity')

print('All tests passed! BanditRouter implementation is correct.')
