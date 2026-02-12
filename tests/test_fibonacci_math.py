import pytest
import fibonacci_math

def test_golden_constants():
    """Test that golden constants are correctly defined"""
    assert abs(fibonacci_math.P - 1.618033988749895) < 1e-15
    assert abs(fibonacci_math.INV_PHI - 0.6180339887498949) < 1e-15
    assert abs(fibonacci_math.SQRT_5 - 2.23606797749979) < 1e-12

def test_calculate_golden_gain():
    """Test the calculate_golden_gain function"""
    # Test with simple values
    gain = fibonacci_math.calculate_golden_gain(1.0, 1.0, 100)
    expected = 0.6180339887498949  # 1/φ
    assert abs(gain - expected) < 0.001  # 0.1% tolerance

def test_verify_gain_convergence():
    """Test that gain converges to 1/φ within 0.1%"""
    # Test convergence
    q, r = 1.0, 1.0
    gain = fibonacci_math.calculate_golden_gain(q, r, 1000)
    expected = 0.6180339887498949  # 1/φ
    tolerance = 0.001  # 0.1%
    assert abs(gain - expected) / expected < tolerance

def test_golden_kalman_filter():
    """Test Golden Kalman Filter functionality"""
    filter = fibonacci_math.GoldenKalmanFilter(1.0, 1.0)
    filter.predict()
    filter.update(5.0)
    state = filter.get_state()
    assert isinstance(state, float)

def test_fibonacci_heap():
    """Test Fibonacci Heap functionality"""
    heap = fibonacci_math.FibonacciHeap()
    heap.push(1, 10.0)
    heap.push(2, 5.0)
    assert heap.size() == 2

def test_logarithmic_spiral():
    """Test Logarithmic Spiral functionality"""
    points = fibonacci_math.generate_spiral_points(1.0, 0.1, 10)
    assert len(points) == 10
    assert isinstance(points[0], tuple)
    assert len(points[0]) == 2
