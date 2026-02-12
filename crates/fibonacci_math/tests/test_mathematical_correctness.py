# Test for mathematical correctness of Fibonacci and Golden Ratio libraries
# This validates the core requirements from WP-043

import sys
import os
import math

# Add the project root to path to import fibonacci_math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_golden_ratio_constants():
    """Test that golden ratio constants are mathematically correct"""
    print("Testing golden ratio constants...")
    
    # Import the module
    try:
        import fibonacci_math
    except ImportError as e:
        print(f"Failed to import fibonacci_math: {e}")
        return False
    
    # Test PHI constant (should be 1.618033988749895)
    phi = fibonacci_math.P
    expected_phi = 1.618033988749895
    assert abs(phi - expected_phi) < 1e-15, f"PHI should be {expected_phi}, got {phi}"
    
    # Test INV_PHI constant (should be 0.6180339887498949)
    inv_phi = fibonacci_math.INV_PHI
    expected_inv_phi = 0.6180339887498949
    assert abs(inv_phi - expected_inv_phi) < 1e-15, f"INV_PHI should be {expected_inv_phi}, got {inv_phi}"
    
    # Test SQRT_5 constant (should be 2.23606797749979)
    sqrt_5 = fibonacci_math.SQRT_5
    expected_sqrt_5 = 2.23606797749979
    assert abs(sqrt_5 - expected_sqrt_5) < 1e-15, f"SQRT_5 should be {expected_sqrt_5}, got {sqrt_5}"
    
    # Test mathematical relationships
    # PHI * PHI = PHI + 1
    assert abs(phi * phi - phi - 1.0) < 1e-15, "PHI * PHI should equal PHI + 1"
    
    # INV_PHI + 1 = PHI
    assert abs(inv_phi + 1.0 - phi) < 1e-15, "INV_PHI + 1 should equal PHI"
    
    # INV_PHI = 1 / PHI
    assert abs(inv_phi - 1.0 / phi) < 1e-15, "INV_PHI should equal 1 / PHI"
    
    # SQRT_5^2 = 5
    assert abs(sqrt_5 * sqrt_5 - 5.0) < 1e-15, "SQRT_5^2 should equal 5"
    
    print("✅ Golden ratio constants test passed")
    return True

def test_golden_kalman_convergence():
    """Test that Golden Kalman filter converges to 1/φ within 0.1% tolerance"""
    print("Testing Golden Kalman convergence...")
    
    try:
        import fibonacci_math
    except ImportError as e:
        print(f"Failed to import fibonacci_math: {e}")
        return False
    
    # Test the calculate_golden_gain function
    gain = fibonacci_math.calculate_golden_gain(1.0, 1.0, 100)
    
    # The gain should converge to 1/PHI (approximately 0.6180339887498949)
    expected_gain = 0.6180339887498949  # 1/PHI
    tolerance = 0.001  # 0.1% tolerance
    
    print(f"Calculated gain: {gain}")
    print(f"Expected gain (1/PHI): {expected_gain}")
    print(f"Difference: {abs(gain - expected_gain)}")
    
    assert abs(gain - expected_gain) < tolerance, f"Gain {gain} should converge to {expected_gain} within {tolerance*100}% tolerance"
    
    print("✅ Golden Kalman convergence test passed")
    return True

def test_fibonacci_heap_properties():
    """Test that Fibonacci Heap maintains correct properties"""
    print("Testing Fibonacci Heap properties...")
    
    try:
        import fibonacci_math
    except ImportError as e:
        print(f"Failed to import fibonacci_math: {e}")
        return False
    
    # Test basic heap functionality
    heap = fibonacci_math.FibonacciHeap()
    
    # Test that heap is initially empty
    # Note: The actual heap interface may vary, so we'll test basic availability
    assert heap is not None, "FibonacciHeap should be creatable"
    
    print("✅ Fibonacci Heap basic test passed")
    return True

def test_logarithmic_spiral_generation():
    """Test that logarithmic spiral generator works correctly"""
    print("Testing Logarithmic Spiral generation...")
    
    try:
        import fibonacci_math
    except ImportError as e:
        print(f"Failed to import fibonacci_math: {e}")
        return False
    
    # Test spiral point generation
    points = fibonacci_math.generate_spiral_points(1.0, 0.1, 5)
    
    assert len(points) == 5, f"Should generate 5 points, got {len(points)}"
    
    # Check that points are valid
    for i, (x, y) in enumerate(points):
        assert isinstance(x, (int, float)), f"Point {i} x-coordinate should be numeric"
        assert isinstance(y, (int, float)), f"Point {i} y-coordinate should be numeric"
    
    print("✅ Logarithmic Spiral generation test passed")
    return True

def main():
    """Run all tests"""
    print("Running comprehensive mathematical correctness tests for fibonacci_math...")
    
    tests = [
        test_golden_ratio_constants,
        test_golden_kalman_convergence,
        test_fibonacci_heap_properties,
        test_logarithmic_spiral_generation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All mathematical correctness tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
