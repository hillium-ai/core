//! Integration tests for fibonacci_math crate

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_golden_constants() {
        // Test that golden constants are correctly defined
        assert_eq!(fibonacci_math::golden_constants::PHI, 1.618033988749895);
        assert_eq!(fibonacci_math::golden_constants::INV_PHI, 0.6180339887498949);
        assert_eq!(fibonacci_math::golden_constants::SQRT_5, 2.23606797749979);
    }
    
    #[test]
    fn test_golden_kalman_convergence() {
        // Test that Kalman gain converges to 1/PHI
        let gain = fibonacci_math::golden_kalman::golden_kalman_gain(1.0, 1.0, 100);
        let expected = 0.6180339887498949; // 1/PHI
        assert!((gain - expected).abs() < 0.001, "Gain should converge to 1/PHI");
    }
    
    #[test]
    fn test_fibonacci_heap() {
        // Test basic Fibonacci heap functionality
        let mut heap = fibonacci_math::fibonacci_heap::FibonacciHeap::new();
        heap.push(5.0, "item1");
        heap.push(10.0, "item2");
        assert_eq!(heap.pop_min(), Some((5.0, "item1")));
    }
    
    #[test]
    fn test_logarithmic_spiral() {
        // Test logarithmic spiral generation
        let spiral = fibonacci_math::logarithmic_spiral::LogarithmicSpiral::new(1.0, 0.5);
        let point = spiral.point_at_angle(0.0);
        assert_eq!(point.0, 1.0);
        assert_eq!(point.1, 0.0);
    }
}