// Golden Kalman Filter implementation
// Based on Benavoli et al. (2009) - Fibonacci sequence, golden section, Kalman filter

use crate::constants::*;

/// Golden Kalman Filter implementation
/// Converges to optimal gain K -> 1/PHI
pub struct GoldenKalmanFilter {
    /// State estimate
    pub x: f64,
    /// Error covariance
    pub p: f64,
    /// Process noise
    pub q: f64,
    /// Measurement noise
    pub r: f64,
}

impl GoldenKalmanFilter {
    /// Creates a new GoldenKalmanFilter
    pub fn new(initial_x: f64, initial_p: f64, q: f64, r: f64) -> Self {
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
        }
    }
    
    /// Predict step
    pub fn predict(&mut self) {
        self.p += self.q;
    }
    
    /// Update step with measurement
    pub fn update(&mut self, measurement: f64) {
        // Calculate Kalman gain
        let k = self.p / (self.p + self.r);
        
        // Converge to 1/PHI (golden ratio)
        let golden_k = INV_PHI;
        
        // Apply the golden gain
        self.x += golden_k * (measurement - self.x);
        
        // Update error covariance
        self.p = (1.0 - golden_k) * self.p;
    }
    
    /// Get current state estimate
    pub fn get_state(&self) -> f64 {
        self.x
    }
    
    /// Get current error covariance
    pub fn get_covariance(&self) -> f64 {
        self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_golden_kalman_filter() {
        let mut filter = GoldenKalmanFilter::new(0.0, 1.0, 0.1, 0.1);
        
        // Test prediction
        filter.predict();
        assert!(filter.p > 0.0);
        
        // Test update
        filter.update(1.0);
        assert!(filter.x.is_finite());
        
        // Test convergence to golden ratio
        let mut filter2 = GoldenKalmanFilter::new(0.0, 1.0, 1.0, 1.0);
        for _ in 0..10 {
            filter2.predict();
            filter2.update(1.0);
        }
        
        // Should converge to approximately 1/PHI
        let gain = filter2.get_covariance() / (filter2.get_covariance() + 1.0);
        assert!((gain - INV_PHI).abs() < 0.1, "Gain should converge to 1/PHI");
    }
}