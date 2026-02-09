import pytest
from fibonacci_math import (
    PHI,
    INV_PHI,
    SQRT_5,
    GoldenKalmanFilter,
    FibonacciHeap,
    generate_spiral_points
)

def test_golden_constants():
    assert abs(PHI * INV_PHI - 1.0) < 1e-10
    assert abs(PHI - 1.0 - INV_PHI) < 1e-10
    assert abs(SQRT_5 * SQRT_5 - 5.0) < 1e-10

def test_golden_kalman_filter():
    filter = GoldenKalmanFilter(1.0, 1.0)
    filter.predict()
    filter.update()
    # Should converge to golden ratio
    assert abs(filter.get_gain() - INV_PHI) < 0.001

def test_fibonacci_heap():
    heap = FibonacciHeap()
    assert heap.is_empty()
    
    heap.insert(5.0, "five")
    heap.insert(2.0, "two")
    heap.insert(8.0, "eight")
    
    assert heap.len() == 3
    assert not heap.is_empty()
    
    min_val = heap.extract_min()
    assert min_val == "two"
    
    min_val = heap.extract_min()
    assert min_val == "five"
    
    min_val = heap.extract_min()
    assert min_val == "eight"
    
    assert heap.is_empty()

def test_spiral_points():
    points = generate_spiral_points(1.0, 0.1, 10)
    assert len(points) == 10
    assert points[0] == (1.0, 0.0)
