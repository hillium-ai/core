# Verification script for TheSolver implementation

print("=== Verifying TheSolver Implementation ===")

# Check 1: File exists
import os
file_exists = os.path.exists('loqus_core/cognitive_council/the_solver.py')
print(f"1. File exists: {file_exists}")

# Check 2: RestrictedPython import
try:
    from RestrictedPython import compile_restricted
    print("2. RestrictedPython import: SUCCESS")
except ImportError as e:
    print(f"2. RestrictedPython import: FAILED - {e}")

# Check 3: Basic class structure
try:
    import sys
    sys.path.insert(0, '.')
    from loqus_core.cognitive_council.the_solver import TheSolver
    ts = TheSolver()
    print("3. Class instantiation: SUCCESS")
    
    # Test basic functionality
    result = ts.solve('solve 2 + 2')
    print(f"4. Basic math execution: {result}")
    
    print("=== Implementation Verification Complete ===")
    
except Exception as e:
    print(f"3. Class instantiation: FAILED - {e}")
    import traceback
    traceback.print_exc()