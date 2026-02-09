// Golden Kalman Filter implementation for Fibonacci Math Library

/// Golden Kalman Filter
pub struct GoldenKalmanFilter {
    /// State estimate
    pub state: f64,
    /// Error covariance
    pub p: f64,
    /// Gain
    pub gain: f64,
}

impl GoldenKalmanFilter {
    /// Creates a new Golden Kalman Filter
    pub fn new() -> Self {
        GoldenKalmanFilter {
            state: 0.0,
            p: 1.0,
            gain: 0.0,
        }
    }

    /// Predict step
    pub fn predict(&mut self, q: f64, r: f64) {
        // Update error covariance
        self.p = self.p + q;
        
        // Calculate gain (converges to 1/PHI)
        self.gain = self.p / (self.p + r);
        
        // Update error covariance
        self.p = self.p - self.gain * self.p;
    }

    /// Update step
    pub fn update(&mut self, measurement: f64, q: f64, r: f64) -> f64 {
        // Predict step
        self.predict(q, r);
        
        // Update state estimate
        let innovation = measurement - self.state;
        self.state = self.state + self.gain * innovation;
        
        self.state
    }

    /// Gets the current gain
    pub fn gain(&self) -> f64 {
        self.gain
    }
}

/// Calculates the golden kalman gain (converges to 1/PHI)
pub fn golden_kalman_gain(q: f64, r: f64, iterations: usize) -> f64 {
    let mut p = 1.0;
    for _ in 0..iterations {
        p = q + p - (p * p) / (p + r);
    }
    p / (p + r)  // Converges to 1/PHI
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_golden_kalman_gain_function() {
        let gain = golden_kalman_gain(1.0, 1.0, 100);
        let expected_gain = 0.6180339887498949; // 1/PHI
        
        assert!((gain - expected_gain).abs() < 0.001);
    }
}
