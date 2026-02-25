use fibonacci_math::*;

#[test]
fn test_golden_constants() {
    assert!((PHI - 1.0 - INV_PHI).abs() < 1e-10);
    assert!((PHI * PHI - PHI - 1.0).abs() < 1e-10);
    assert!((SQRT_5 * SQRT_5 - 5.0).abs() < 1e-10);
}

#[test]
fn test_golden_kalman_gain_convergence() {
    let gain = golden_kalman_gain(1.0, 1.0, 100);
    assert!((gain - INV_PHI).abs() < 0.001, "Gain should converge to 1/φ");
}

#[test]
fn test_fibonacci_heap() {
    let mut heap = FibonacciHeap::new();
    assert!(heap.is_empty());
    
    let index = heap.insert(1.0);
    assert!(!heap.is_empty());
    assert_eq!(heap.len(), 1);
    
    heap.decrease_key(index, 0.5);
    assert_eq!(heap.len(), 1);
}

#[test]
fn test_logarithmic_spiral() {
    let spiral = LogarithmicSpiral::golden_spiral();
    let points = spiral.generate_points(10, None, None);
    assert_eq!(points.len(), 10);
    
    // Check that points are generated
    assert!(points[0].0 != 0.0 || points[0].1 != 0.0);
}
