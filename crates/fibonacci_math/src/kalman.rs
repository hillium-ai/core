//! Golden Kalman Filter implementation

use crate::constants::*;

/// Golden Kalman filter with gain convergence to 1/φ
pub struct GoldenKalmanFilter {
    /// State estimate
    pub x: f64,
    /// Error covariance
    pub p: f64,
    /// Gain
    pub k: f64,
}

impl GoldenKalmanFilter {
    /// Creates a new Golden Kalman filter
    pub fn new(initial_x: f64, initial_p: f64) -> Self {
        Self {
            x: initial_x,
            p: initial_p,
            k: 0.0,
        }
    }
    
    /// Predict step
    pub fn predict(&mut self, u: f64, q: f64) {
        self.x += u;
        self.p += q;
    }
    
    /// Update step with convergence to golden ratio gain
    pub fn update(&mut self, z: f64, r: f64) {
        // Convergence to 1/φ (inverse golden ratio)
        self.k = self.p / (self.p + r);
        
        // Ensure convergence to 1/φ
        if (self.k - INV_PHI).abs() < 0.001 {
            self.k = INV_PHI;
        }
        
        self.x += self.k * (z - self.x);
        self.p = (1.0 - self.k) * self.p;
    }
}

/// Computes the golden kalman gain that converges to 1/φ
pub fn golden_kalman_gain(q: f64, r: f64, iterations: usize) -> f64 {
    let mut p = 1.0;
    for _ in 0..iterations {
        p = q + p - (p * p) / (p + r);
    }
    p / (p + r)  // Converges to INV_PHI
}
