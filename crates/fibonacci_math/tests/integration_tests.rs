//! Integration tests for Fibonacci Math Library

use fibonacci_math::*;

#[test]
fn test_golden_constants() {
    assert_eq!(golden_constants::PHI, 1.618033988749895);
    assert_eq!(golden_constants::INV_PHI, 0.6180339887498949);
    assert_eq!(golden_constants::SQRT_5, 2.23606797749979);
}

#[test]
fn test_golden_kalman_filter() {
    let mut filter = GoldenKalmanFilter::new(1.0, 1.0);
    filter.predict();
    filter.update(5.0);
    
    // Basic sanity check - should not panic
    assert!(filter.x.is_finite());
    assert!(filter.p.is_finite());
}

#[test]
fn test_fibonacci_heap() {
    let mut heap = FibonacciHeap::new();
    heap.insert(1.0);
    heap.insert(2.0);
    
    // Basic sanity check - should not panic
    assert_eq!(heap.size, 2);
}

#[test]
fn test_logarithmic_spiral() {
    let spiral = LogarithmicSpiral::new(1.0, 0.1);
    let points = spiral.trajectory(10);
    
    // Should generate 10 points
    assert_eq!(points.len(), 10);
    
    // All points should be finite
    for &(x, y) in &points {
        assert!(x.is_finite());
        assert!(y.is_finite());
    }
}
