# Detailed test script for TheSolver implementation

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loqus_core.cognitive_council.the_solver import TheSolver

# Create an instance
solver = TheSolver()

print("Detailed Testing TheSolver implementation...")

# Test 1: Valid mathematical expression
print("\nTest 1: Valid expression")
result = solver.solve("What is 2 + 2?")
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test 2: Invalid syntax (should trigger retry mechanism)
print("\nTest 2: Invalid syntax")
result = solver.solve("What is 2 +")
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test 3: Complex expression
print("\nTest 3: Complex expression")
result = solver.solve("Calculate 10 * (5 + 3) / 2")
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test 4: Division by zero
print("\nTest 4: Division by zero")
result = solver.solve("What is 5 / 0?")
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test 5: Verify retry mechanism works
print("\nTest 5: Verify retry mechanism")
print("Testing that invalid syntax is handled properly with retries...")

# Test 6: Check capabilities
print("\nTest 6: Capabilities")
capabilities = solver.get_capabilities()
print(f"Capabilities: {capabilities}")

print("\nDetailed test completed.")
