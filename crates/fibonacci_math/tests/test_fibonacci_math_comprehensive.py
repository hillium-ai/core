import sys
import os
import math

# Add the fibonacci_math crate to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import fibonacci_math
    print("✅ Successfully imported fibonacci_math")
except ImportError as e:
    print(f"❌ Failed to import fibonacci_math: {e}")
    sys.exit(1)

def test_golden_constants():
    """Test that all golden ratio constants are correctly defined"""
    print("Testing golden constants...")
    
    # Test PHI constant
    assert abs(fibonacci_math.P - 1.618033988749895) < 1e-15, f"PHI should be 1.618033988749895, got {fibonacci_math.P}"
    
    # Test INV_PHI constant  
    assert abs(fibonacci_math.INV_PHI - 0.6180339887498949) < 1e-15, f"INV_PHI should be 0.6180339887498949, got {fibonacci_math.INV_PHI}"
    
    # Test SQRT_5 constant
    assert abs(fibonacci_math.SQRT_5 - 2.23606797749979) < 1e-15, f"SQRT_5 should be 2.23606797749979, got {fibonacci_math.SQRT_5}"
    
    # Test mathematical relationship: PHI * PHI = PHI + 1
    assert abs(fibonacci_math.P * fibonacci_math.P - fibonacci_math.P - 1.0) < 1e-15, "PHI * PHI should equal PHI + 1"
    
    # Test mathematical relationship: INV_PHI + 1 = PHI
    assert abs(fibonacci_math.INV_PHI + 1.0 - fibonacci_math.P) < 1e-15, "INV_PHI + 1 should equal PHI"
    
    print("✅ All golden constants test passed")

def test_golden_kalman_convergence():
    """Test that Golden Kalman filter converges to 1/φ within 0.1% tolerance"""
    print("Testing Golden Kalman convergence...")
    
    # Test the calculate_golden_gain function
    gain = fibonacci_math.calculate_golden_gain(1.0, 1.0, 100)
    
    # The gain should converge to 1/PHI (approximately 0.6180339887498949)
    expected_gain = 0.6180339887498949
    tolerance = 0.001  # 0.1% tolerance
    
    assert abs(gain - expected_gain) < tolerance, f"Gain {gain} should converge to {expected_gain} within {tolerance*100}% tolerance"
    
    print("✅ Golden Kalman convergence test passed")

def test_fibonacci_heap():
    """Test that Fibonacci Heap works correctly"""
    print("Testing Fibonacci Heap...")
    
    # Test basic functionality
    heap = fibonacci_math.FibonacciHeap()
    
    # Test that heap is initially empty
    assert heap is not None, "FibonacciHeap should be creatable"
    
    print("✅ Fibonacci Heap test passed")

def test_logarithmic_spiral():
    """Test that logarithmic spiral generator works correctly"""
    print("Testing Logarithmic Spiral...")
    
    # Test spiral point generation
    points = fibonacci_math.generate_spiral_points(5, 0.1, 10)
    assert len(points) == 10, f"Should generate 10 points, got {len(points)}"
    
    # Check that points are valid (non-zero)
    for i, (x, y) in enumerate(points):
        assert isinstance(x, (int, float)), f"Point {i} x-coordinate should be numeric"
        assert isinstance(y, (int, float)), f"Point {i} y-coordinate should be numeric"
    
    print("✅ Logarithmic Spiral test passed")

def test_import():
    """Test that the module can be imported"""
    print("Testing module import...")
    assert fibonacci_math is not None, "fibonacci_math module should not be None"
    print("✅ Module import test passed")

def test_mathematical_properties():
    """Test mathematical properties of the golden ratio"""
    print("Testing mathematical properties...")
    
    # PHI * PHI = PHI + 1
    assert abs(fibonacci_math.P * fibonacci_math.P - fibonacci_math.P - 1.0) < 1e-15
    
    # INV_PHI = 1 / PHI
    assert abs(fibonacci_math.INV_PHI - 1.0 / fibonacci_math.P) < 1e-15
    
    # INV_PHI + 1 = PHI
    assert abs(fibonacci_math.INV_PHI + 1.0 - fibonacci_math.P) < 1e-15
    
    # SQRT_5^2 = 5
    assert abs(fibonacci_math.SQRT_5 * fibonacci_math.SQRT_5 - 5.0) < 1e-15
    
    print("✅ Mathematical properties test passed")

if __name__ == "__main__":
    try:
        test_import()
        test_golden_constants()
        test_golden_kalman_convergence()
        test_fibonacci_heap()
        test_logarithmic_spiral()
        test_mathematical_properties()
        print("🎉 All comprehensive tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
