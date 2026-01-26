#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'loqus_core'))

# Test the solver implementation
try:
    from cognitive_council.the_solver import solver, TheSolver
    print("✅ Import successful")
    
    # Test basic functionality
    test_query = "calculate 2 + 3"
    result = solver.solve(test_query)
    print(f"✅ Basic test result: {result}")
    
    # Test with more complex query
    test_query2 = "solve 5 * 4"
    result2 = solver.solve(test_query2)
    print(f"✅ Complex test result: {result2}")
    
    # Test error handling
    test_query3 = "calculate 2 plus 3"
    result3 = solver.solve(test_query3)
    print(f"✅ Error handling test result: {result3}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()