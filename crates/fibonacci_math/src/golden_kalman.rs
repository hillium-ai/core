// Golden Kalman Filter implementation

pub fn golden_kalman_gain(q: f64, r: f64, iterations: usize) -> f64 {
    // Validate inputs
    if q <= 0.0 || r <= 0.0 {
        panic!("q and r must be positive");
    }
    
    let mut p = 1.0;
    for _ in 0..iterations {
        p = q + p - (p * p) / (p + r);
    }
    p / (p + r)  // Converges to INV_PHI
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_golden_kalman_gain_convergence() {
        let gain = golden_kalman_gain(1.0, 1.0, 100);
        let expected = 0.6180339887498949; // INV_PHI
        
        // Should converge within 0.1%
        assert!((gain - expected).abs() < 0.001);
    }
}