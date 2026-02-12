//! Golden Kalman Filter implementation

use pyo3::prelude::*;

/// Golden Kalman Filter implementation
#[pyclass]
#[derive(Debug, Clone)]
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
    
    /// Update step with measurement
    pub fn update(&mut self, measurement: f64) {
        // Golden Kalman update equation
        let k = self.p / (self.p + self.r); // Kalman gain
        self.x += k * (measurement - self.x);
        self.p = (1.0 - k) * self.p;
    }
    
    /// Get the current state estimate
    pub fn get_state(&self) -> f64 {
        self.x
    }
    
    /// Get the current error covariance
    pub fn get_covariance(&self) -> f64 {
        self.p
    }
    
    /// Calculate the Kalman gain (should converge to 1/φ)
    pub fn kalman_gain(&self) -> f64 {
        self.p / (self.p + self.r)
    }
}

/// Calculate the Golden Kalman gain that converges to 1/φ
/// 
/// This function demonstrates the mathematical convergence to the golden ratio
/// as described in Benavoli et al. (2009)
pub fn golden_kalman_gain(q: f64, r: f64, iterations: usize) -> f64 {
    // The Riccati equation converges to (sqrt(5) - 1) / 2 = 0.6180339887498949
    let mut p = 1.0;
    for _ in 0..iterations {
        p = q + p - (p * p) / (p + r);
    }
    p / (p + r)  // Converges to INV_PHI (1/φ)
}
