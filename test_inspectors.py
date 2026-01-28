from loqus_core.cognitive_council.multi_inspector import (
    mock_inspector_1,
    mock_inspector_2,
    mock_inspector_3,
    ActionPlan
)

# Test individual inspectors
action_plan = ActionPlan("This is a safe action")

print("Testing individual inspectors:")
print(f"Inspector 1: {mock_inspector_1(action_plan)}")
print(f"Inspector 2: {mock_inspector_2(action_plan)}")
print(f"Inspector 3: {mock_inspector_3(action_plan)}")

# Test with a plan that should make all inspectors approve
action_plan2 = ActionPlan("This is a permitted action")

print("\nTesting with permitted action:")
print(f"Inspector 1: {mock_inspector_1(action_plan2)}")
print(f"Inspector 2: {mock_inspector_2(action_plan2)}")
print(f"Inspector 3: {mock_inspector_3(action_plan2)}")