// Integration tests for Fibonacci Math Library

use fibonacci_math::*;

#[test]
fn test_golden_constants() {
    assert_eq!(PHI * INV_PHI, 1.0);
    assert_eq!(PHI - 1.0, INV_PHI);
    assert_eq!(SQRT_5 * SQRT_5, 5.0);
}

#[test]
fn test_golden_kalman_convergence() {
    let mut filter = GoldenKalmanFilter::new();
    
    // Test convergence to 1/PHI
    let q = 0.1;
    let r = 0.1;
    
    // Run several iterations to see convergence
    for _ in 0..100 {
        filter.predict(q, r);
    }
    
    let gain = filter.gain();
    let expected_gain = 0.6180339887498949; // 1/PHI
    
    // Should converge within 0.1% tolerance
    assert!((gain - expected_gain).abs() < 0.001);
}

#[test]
fn test_fibonacci_heap_operations() {
    let mut heap = FibonacciHeap::new();
    
    // Test insert
    heap.insert(5.0, "five");
    heap.insert(3.0, "three");
    heap.insert(7.0, "seven");
    
    assert_eq!(heap.size(), 3);
    
    // Test find_min
    assert_eq!(heap.find_min(), Some(&"three"));
    
    // Test extract_min
    assert_eq!(heap.extract_min(), Some("three"));
    assert_eq!(heap.extract_min(), Some("five"));
    assert_eq!(heap.extract_min(), Some("seven"));
    assert_eq!(heap.extract_min(), None);
}

#[test]
fn test_logarithmic_spiral_generation() {
    let spiral = LogarithmicSpiral::new(1.0, 0.1);
    
    // Test point generation
    let (x, y) = spiral.point_at_angle(0.0);
    assert_eq!(x, 1.0);
    assert_eq!(y, 0.0);
    
    // Test multiple points
    let points = spiral.generate_points(0.0, 1.0, 5);
    assert_eq!(points.len(), 5);
}

#[test]
fn test_fibonacci_heap_decrease_key() {
    let mut heap = FibonacciHeap::new();
    let node = heap.insert(5.0, "five");
    
    // Test decrease key operation
    heap.decrease_key(&node, 2.0);
    assert_eq!(heap.find_min(), Some(&"five"));
}