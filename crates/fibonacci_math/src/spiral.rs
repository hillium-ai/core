//! Logarithmic spiral trajectory generator

use crate::constants::*;

/// Spiral trajectory point
#[derive(Debug, Clone)]
pub struct SpiralPoint {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
}

/// Logarithmic spiral generator
pub struct LogarithmicSpiral {
    /// The constant a in the spiral equation r = a * e^(b*theta)
    pub a: f64,
    /// The constant b in the spiral equation r = a * e^(b*theta)
    pub b: f64,
    /// Starting angle
    pub start_angle: f64,
    /// Number of points to generate
    pub points: usize,
}

impl LogarithmicSpiral {
    /// Creates a new logarithmic spiral
    pub fn new(a: f64, b: f64, start_angle: f64, points: usize) -> Self {
        Self {
            a,
            b,
            start_angle,
            points,
        }
    }
    
    /// Generates points along the spiral
    pub fn generate_points(&self) -> Vec<SpiralPoint> {
        let mut points = Vec::with_capacity(self.points);
        
        for i in 0..self.points {
            let angle = self.start_angle + (i as f64) * 0.1; // Step size
            let radius = self.a * (self.b * angle).exp();
            
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            
            points.push(SpiralPoint { x, y, angle });
        }
        
        points
    }
    
    /// Generates a golden spiral (specific case)
    pub fn golden_spiral(points: usize) -> Self {
        // For golden spiral, b = 1/phi
        Self::new(1.0, INV_PHI, 0.0, points)
    }
}
