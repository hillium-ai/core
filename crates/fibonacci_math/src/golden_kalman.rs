//! Golden Kalman Filter implementation with convergence to 1/phi

use pyo3::prelude::*;

/// Golden Kalman Filter implementation
#[pyclass]
#[derive(Debug, Clone)]
pub struct GoldenKalmanFilter {
    /// State estimate
    pub state: f64,
    /// Error covariance
    pub covariance: f64,
    /// Gain
    pub gain: f64,
}

#[pymethods]
impl GoldenKalmanFilter {
    /// Create a new Golden Kalman Filter
    #[new]
    pub fn new(initial_state: f64, initial_covariance: f64) -> Self {
        GoldenKalmanFilter {
            state: initial_state,
            covariance: initial_covariance,
            gain: 0.0, // Will be set during update
        }
    }

    /// Predict step
    pub fn predict(&mut self, process_noise: f64) {
        self.covariance += process_noise;
    }

    /// Update step with measurement
    pub fn update(&mut self, measurement: f64, measurement_noise: f64) {
        // Golden Kalman update with convergence to 1/phi
        let k = golden_kalman_gain(self.covariance, measurement_noise, 100);
        self.gain = k;
        self.state += k * (measurement - self.state);
        self.covariance = (1.0 - k) * self.covariance;
    }

    /// Get the current state estimate
    pub fn get_state(&self) -> f64 {
        self.state
    }

    /// Get the current gain
    pub fn get_gain(&self) -> f64 {
        self.gain
    }
}

/// Calculate Golden Kalman gain that converges to 1/phi
pub fn golden_kalman_gain(q: f64, r: f64, iterations: usize) -> f64 {
    // This implements the convergence to 1/PHI as described in Benavoli 2009
    let mut p = q;
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
        let tolerance = 0.001; // 0.1% tolerance
        assert!((gain - 0.6180339887498949).abs() < tolerance);
    }
    
    #[test]
    fn test_golden_kalman_filter() {
        let mut filter = GoldenKalmanFilter::new(0.0, 1.0);
        filter.update(1.0, 1.0);
        assert!(filter.get_gain() > 0.0);
    }
}