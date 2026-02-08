// Logarithmic Spiral implementation for Rust

use std::f64;

/// Logarithmic Spiral trajectory generator
#[derive(Debug, Clone)]
pub struct LogarithmicSpiral {
    /// Spiral scaling factor
    a: f64,
    /// Spiral tightness (related to golden ratio)
    b: f64,
    /// Spiral center point
    center: (f64, f64),
}

impl LogarithmicSpiral {
    /// Creates a new logarithmic spiral with given parameters
    pub fn new(a: f64, b: f64, center: (f64, f64)) -> Self {
        LogarithmicSpiral { a, b, center }
    }

    /// Generates a point along the spiral at angle theta
    pub fn point_at(&self, theta: f64) -> (f64, f64) {
        let r = self.a * f64::exp(self.b * theta);
        let x = r * f64::cos(theta) + self.center.0;
        let y = r * f64::sin(theta) + self.center.1;
        (x, y)
    }

    /// Calculates the arc length from theta_start to theta_end
    pub fn arc_length(&self, theta_start: f64, theta_end: f64) -> f64 {
        let numerator = f64::sqrt(self.b * self.b + 1.0) * self.a * (f64::exp(self.b * theta_end) - f64::exp(self.b * theta_start));
        numerator / self.b
    }

    /// Generates a trajectory segment with specified resolution
    pub fn trajectory(&self, theta_start: f64, theta_end: f64, resolution: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(resolution);
        let step = (theta_end - theta_start) / (resolution as f64 - 1.0);
        
        for i in 0..resolution {
            let theta = theta_start + (i as f64) * step;
            points.push(self.point_at(theta));
        }
        
        points
    }

    /// Gets the spiral scaling factor
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Gets the spiral tightness
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Gets the spiral center
    pub fn center(&self) -> (f64, f64) {
        self.center
    }
}

/// Golden ratio related constants
pub const PHI: f64 = 1.618033988749895;
pub const INV_PHI: f64 = 0.6180339887498949; // 1/PHI = PHI - 1
pub const SQRT_5: f64 = 2.23606797749979;

/// Golden spiral with parameters based on the golden ratio
pub fn golden_spiral(center: (f64, f64)) -> LogarithmicSpiral {
    // For golden spiral, b = 1/phi
    LogarithmicSpiral::new(1.0, INV_PHI, center)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiral_creation() {
        let spiral = LogarithmicSpiral::new(1.0, 0.1, (0.0, 0.0));
        assert_eq!(spiral.a(), 1.0);
        assert_eq!(spiral.b(), 0.1);
        assert_eq!(spiral.center(), (0.0, 0.0));
    }

    #[test]
    fn test_point_at() {
        let spiral = LogarithmicSpiral::new(1.0, 0.0, (0.0, 0.0)); // Circle
        let point = spiral.point_at(0.0);
        assert_eq!(point.0, 1.0);
        assert_eq!(point.1, 0.0);
    }

    #[test]
    fn test_golden_spiral() {
        let spiral = golden_spiral((0.0, 0.0));
        assert_eq!(spiral.a(), 1.0);
        assert!((spiral.b() - INV_PHI).abs() < 1e-10);
    }
}