# Test script for TheSolver implementation

from loqus_core.cognitive_council.the_solver import TheSolver

# Create an instance
solver = TheSolver()

print("Testing TheSolver implementation...")

# Test 1: Valid mathematical expression
print("\nTest 1: Valid expression")
result = solver.solve("What is 2 + 2?")
print(f"Result: {result}")

# Test 2: Invalid syntax (should trigger retry)
print("\nTest 2: Invalid syntax")
result = solver.solve("What is 2 +")
print(f"Result: {result}")

# Test 3: Complex expression
print("\nTest 3: Complex expression")
result = solver.solve("Calculate 10 * (5 + 3) / 2")
print(f"Result: {result}")

# Test 4: Division by zero
print("\nTest 4: Division by zero")
result = solver.solve("What is 5 / 0?")
print(f"Result: {result}")

# Test 5: Capabilities
print("\nTest 5: Capabilities")
capabilities = solver.get_capabilities()
print(f"Capabilities: {capabilities}")

print("\nTest completed.")
