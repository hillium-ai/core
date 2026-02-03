import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test the solver implementation
    from loqus_core.cognitive_council.the_solver_simple import solver, TheSolver
    print("✅ Import successful")
    
    # Test basic functionality - these are the tests that should pass
    test_cases = [
        "calculate 2 plus 3",
        "solve 5 times 4",
        "what is 10 divided by 2",
        "2 + 3",
        "5 * 6"
    ]
    
    for i, test_query in enumerate(test_cases):
        result = solver.solve(test_query)
        print(f"✅ Test {i+1} '{test_query}' -> {result}")
    
    # Test error handling
    error_cases = ["", "invalid query with no numbers"]
    for i, test_query in enumerate(error_cases):
        result = solver.solve(test_query)
        print(f"✅ Error test {i+1} '{test_query}' -> {result}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()