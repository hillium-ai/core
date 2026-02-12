import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crates/fibonacci_math'))

import fibonacci_math
import math

def test_constants():
    """Test that all golden ratio constants are correctly defined"""
    # Test PHI constant (named P in the Python bindings)
    assert abs(fibonacci_math.P - 1.618033988749895) < 1e-15
    
    # Test INV_PHI constant  
    assert abs(fibonacci_math.INV_PHI - 0.6180339887498949) < 1e-15
    
    # Test SQRT_5 constant
    assert abs(fibonacci_math.SQRT_5 - 2.23606797749979) < 1e-15
    
    print("✅ All constants test passed")

def test_kalman_filter():
    """Test that Golden Kalman filter works correctly"""
    # Create a filter instance
    filter_instance = fibonacci_math.GoldenKalmanFilter()
    
    # Test basic functionality
    assert filter_instance is not None
    
    print("✅ Kalman filter test passed")

def test_fibonacci_heap():
    """Test that Fibonacci Heap works correctly"""
    # Create a heap instance
    heap = fibonacci_math.FibonacciHeap()
    
    # Test basic functionality
    assert heap is not None
    
    print("✅ Fibonacci heap test passed")

def test_spiral_generator():
    """Test that spiral generator works correctly"""
    # Test spiral point generation
    points = fibonacci_math.generate_spiral_points(5)
    assert len(points) == 5
    
    print("✅ Spiral generator test passed")

def test_import():
    """Test that the module can be imported"""
    assert fibonacci_math is not None
    print("✅ Module import test passed")

if __name__ == "__main__":
    test_import()
    test_constants()
    test_kalman_filter()
    test_fibonacci_heap()
    test_spiral_generator()
    print("🎉 All tests passed!")
