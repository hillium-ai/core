import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test the solver implementation
    from loqus_core.cognitive_council.the_solver_simple import solver, TheSolver
    print("✅ Import successful")
    
    # Test the exact cases from the test files
    test_cases = [
        ("calculate 2 plus 3", 5.0),
        ("solve 5 times 4", 20.0),
        ("what is 10 divided by 2", 5.0),
        ("2 + 3", 5.0),
        ("5 * 6", 30.0),
        ("10 / 2", 5.0)
    ]
    
    print("\n=== Testing Core Functionality ===")
    all_passed = True
    for i, (query, expected) in enumerate(test_cases):
        result = solver.solve(query)
        if result['success'] and result['result'] == expected:
            print(f"✅ Test {i+1} '{query}' -> {result['result']} (PASS)")
        else:
            print(f"❌ Test {i+1} '{query}' -> {result} (FAIL - expected {expected})")
            all_passed = False
    
    print("\n=== Testing Error Handling ===")
    error_cases = ["", "invalid query with no numbers"]
    for i, query in enumerate(error_cases):
        result = solver.solve(query)
        if not result['success']:
            print(f"✅ Error test {i+1} '{query}' -> {result} (PASS)")
        else:
            print(f"❌ Error test {i+1} '{query}' -> {result} (FAIL - should have failed)")
            all_passed = False
    
    print("\n=== Testing Security ===")
    security_cases = [
        "import os",
        "exec('print(1)')",
        "eval('1+1')"
    ]
    for i, query in enumerate(security_cases):
        result = solver.solve(query)
        if not result['success']:
            print(f"✅ Security test {i+1} '{query}' -> {result} (PASS - blocked)")
        else:
            print(f"❌ Security test {i+1} '{query}' -> {result} (FAIL - should have been blocked)")
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Implementation is working correctly.")
    else:
        print("\n❌ Some tests failed.")
        
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()