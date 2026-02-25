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
    /// Create a new GoldenKalmanFilter
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
        // Golden Kalman gain calculation
        let k = golden_kalman_gain(self.q, self.r, 100);
        
        // Update state estimate
        self.x += k * (measurement - self.x);
        
        // Update error covariance
        self.p *= 1.0 - k;
    }

    /// Get current state estimate
    pub fn get_state(&self) -> f64 {
        self.x
    }
}

/// Golden Kalman gain function that converges to 1/φ
pub fn golden_kalman_gain(q: f64, r: f64, iterations: usize) -> f64 {
    // Riccati equation converges to (sqrt(5) - 1) / 2
    let mut p = 1.0;
    for _ in 0..iterations {
        p = q + p - (p * p) / (p + r);
    }
    p / (p + r)  // Converges to INV_PHI
}
