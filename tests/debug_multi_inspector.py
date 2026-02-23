import asyncio

# Import the module we just created
import sys
sys.path.insert(0, '.')

from loqus_core.cognitive_council.multi_inspector import (
    mock_inspector_1,
    mock_inspector_2,
    mock_inspector_3,
    ActionPlan
)

# Test what each inspector returns for a safe action
action_plan = ActionPlan("This is a safe action")

print("Testing individual inspectors:")
result1 = mock_inspector_1(action_plan)
print(f"Inspector 1 result: {result1}")

result2 = mock_inspector_2(action_plan)
print(f"Inspector 2 result: {result2}")

result3 = mock_inspector_3(action_plan)
print(f"Inspector 3 result: {result3}")

print("\nTesting with unsafe action:")
unsafe_plan = ActionPlan("This is an unsafe action")
result1_unsafe = mock_inspector_1(unsafe_plan)
print(f"Inspector 1 result (unsafe): {result1_unsafe}")