// Golden Kalman Filter implementation

use pyo3::prelude::*;

/// Golden Kalman Filter with convergence to 1/PHI
#[pyclass]
#[derive(Clone)]
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

#[pymethods]
impl GoldenKalmanFilter {
    /// Create a new Golden Kalman Filter
    #[new]
    pub fn new(q: f64, r: f64) -> Self {
        GoldenKalmanFilter {
            x: 0.0,
            p: 1.0,
            q,
            r,
        }
    }
    
    /// Predict step
    pub fn predict(&mut self) {
        self.p += self.q;
    }
    
    /// Update step with convergence to 1/PHI
    pub fn update(&mut self, measurement: f64) {
        // Convergence to 1/PHI (0.618...) for optimal gain
        let k = self.p / (self.p + self.r);
        
        // Golden ratio convergence: gain should approach 1/PHI
        let golden_k = 0.6180339887498949; // INV_PHI
        
        // Apply the golden ratio convergence
        let adjusted_k = k * (1.0 - golden_k) + golden_k;
        
        self.x += adjusted_k * (measurement - self.x);
        self.p = (1.0 - adjusted_k) * self.p;
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
    fn test_golden_kalman_creation() {
        let filter = GoldenKalmanFilter::new(0.1, 0.1);
        assert_eq!(filter.x, 0.0);
        assert_eq!(filter.p, 1.0);
        assert_eq!(filter.q, 0.1);
        assert_eq!(filter.r, 0.1);
    }
    
    #[test]
    fn test_golden_kalman_predict_update() {
        let mut filter = GoldenKalmanFilter::new(0.1, 0.1);
        filter.predict();
        filter.update(1.0);
        assert!(filter.x.is_finite());
        assert!(filter.p.is_finite());
    }
}