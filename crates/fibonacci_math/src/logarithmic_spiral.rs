// Logarithmic spiral trajectory generator using golden ratio principles

use pyo3::prelude::*;
use crate::golden_constants::INV_PHI;

/// Logarithmic spiral trajectory generator
#[pyclass]
#[derive(Debug, Clone)]
pub struct LogarithmicSpiral {
    /// The constant 'a' in the equation r = a * e^(b * theta)
    a: f64,
    /// The constant 'b' in the equation r = a * e^(b * theta)
    b: f64,
}

#[pymethods]
impl LogarithmicSpiral {
    /// Create a new logarithmic spiral
    ///
    /// For a golden spiral, b = 1/phi
    #[new]
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Create a golden spiral (using phi-based b parameter)
    #[staticmethod]
    pub fn golden_spiral() -> Self {
        // For a golden spiral, we typically use b = 1/phi
        Self {
            a: 1.0,
            b: INV_PHI,
        }
    }

    /// Generate points along the spiral
    ///
    /// # Arguments
    /// * `n` - Number of points to generate
    /// * `start_angle` - Starting angle in radians (default: 0.0)
    /// * `angle_step` - Angle increment in radians (default: 0.1)
    ///
    /// # Returns
    /// Vector of (x, y) coordinates
    pub fn generate_points(&self, n: usize, start_angle: Option<f64>, angle_step: Option<f64>) -> Vec<(f64, f64)> {
        let start = start_angle.unwrap_or(0.0);
        let step = angle_step.unwrap_or(0.1);
        
        let mut points = Vec::with_capacity(n);
        let mut theta = start;
        
        for _ in 0..n {
            let r = self.a * (self.b * theta).exp();
            let x = r * theta.cos();
            let y = r * theta.sin();
            points.push((x, y));
            theta += step;
        }
        
        points
    }

    /// Get the radius at a given angle
    pub fn radius_at(&self, theta: f64) -> f64 {
        self.a * (self.b * theta).exp()
    }

    /// Get the angle at a given radius
    pub fn angle_at_radius(&self, r: f64) -> f64 {
        if r <= 0.0 {
            0.0
        } else {
            (r / self.a).ln() / self.b
        }
    }

    /// Get the curvature at a given angle
    pub fn curvature_at(&self, theta: f64) -> f64 {
        let r = self.radius_at(theta);
        if r == 0.0 {
            0.0
        } else {
            let r_prime = self.b * r;
            let r_double_prime = self.b * self.b * r;
            (r * r_double_prime - 2.0 * r_prime * r_prime) / (r * r + r_prime * r_prime).powf(1.5)
        }
    }

    /// Get the arc length from angle 0 to theta
    pub fn arc_length(&self, theta: f64) -> f64 {
        if theta == 0.0 {
            0.0
        } else {
            let r = self.radius_at(theta);
            let r_prime = self.b * r;
            let integral = (r * r + r_prime * r_prime).sqrt();
            integral / self.b
        }
    }

    /// Get the (x, y) point at a given angle
    pub fn point_at_angle(&self, theta: f64) -> (f64, f64) {
        let r = self.radius_at(theta);
        let x = r * theta.cos();
        let y = r * theta.sin();
        (x, y)
    }
}
