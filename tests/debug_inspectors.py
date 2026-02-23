import sys
sys.path.insert(0, '.')

from loqus_core.cognitive_council.multi_inspector import (
    mock_inspector_1,
    mock_inspector_2,
    mock_inspector_3,
    ActionPlan
)

# Test what happens with the exact string from the test
action_plan = ActionPlan("This is a permitted action")

print("Testing with 'This is a permitted action':")
print(f"Inspector 1: {mock_inspector_1(action_plan)}")
print(f"Inspector 2: {mock_inspector_2(action_plan)}")
print(f"Inspector 3: {mock_inspector_3(action_plan)}")

# Test with a string that should make all approve
action_plan2 = ActionPlan("This is a safe and permitted action")

print("\nTesting with 'This is a safe and permitted action':")
print(f"Inspector 1: {mock_inspector_1(action_plan2)}")
print(f"Inspector 2: {mock_inspector_2(action_plan2)}")
print(f"Inspector 3: {mock_inspector_3(action_plan2)}")