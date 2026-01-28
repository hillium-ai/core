import sys
sys.path.insert(0, '.')

from loqus_core.cognitive_council.multi_inspector import (
    multi_inspector,
    ActionPlan,
    ConsensusResult
)

import asyncio

def test_implementation():
    print("Testing multi_inspector implementation...")
    
    # Test case 1: All inspectors approve
    action_plan = ActionPlan("This is a safe action")
    result = asyncio.run(multi_inspector(action_plan))
    print(f"Test 1 - Safe action result: {result}")
    
    # Test case 2: Any inspector rejects
    action_plan = ActionPlan("This is an unsafe action")
    result = asyncio.run(multi_inspector(action_plan))
    print(f"Test 2 - Unsafe action result: {result}")
    
    # Test case 3: Mixed results
    action_plan = ActionPlan("This is a neutral action")
    result = asyncio.run(multi_inspector(action_plan))
    print(f"Test 3 - Neutral action result: {result}")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_implementation()