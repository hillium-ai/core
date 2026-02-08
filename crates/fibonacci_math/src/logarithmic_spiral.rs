// Logarithmic Spiral trajectory generator for HilliumOS

//! A logarithmic spiral is a curve that often appears in nature.
//! It is defined by the equation: r = a * e^(b * θ)
//! where r is the distance from the origin, θ is the angle, and a, b are constants.

/// Logarithmic Spiral generator
pub struct LogarithmicSpiral {
    /// The growth rate parameter
    pub a: f64,
    /// The spiral tightness parameter
    pub b: f64,
}

impl LogarithmicSpiral {
    /// Create a new logarithmic spiral with given parameters
    pub fn new(a: f64, b: f64) -> Self {
        LogarithmicSpiral { a, b }
    }

    /// Generate a point on the spiral at angle θ
    pub fn point_at_angle(&self, theta: f64) -> (f64, f64) {
        let r = self.a * (self.b * theta).exp();
        (r * theta.cos(), r * theta.sin())
    }

    /// Generate multiple points along the spiral
    pub fn generate_points(&self, start_theta: f64, end_theta: f64, num_points: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(num_points);
        let step = (end_theta - start_theta) / (num_points - 1) as f64;
        
        for i in 0..num_points {
            let theta = start_theta + i as f64 * step;
            points.push(self.point_at_angle(theta));
        }
        
        points
    }

    /// Generate 3D points along the spiral
    pub fn generate_3d_points(&self, start_theta: f64, end_theta: f64, num_points: usize, z_offset: f64) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::with_capacity(num_points);
        let step = (end_theta - start_theta) / (num_points - 1) as f64;
        
        for i in 0..num_points {
            let theta = start_theta + i as f64 * step;
            let (x, y) = self.point_at_angle(theta);
            points.push((x, y, z_offset + theta * 0.1)); // Simple z progression
        }
        
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logarithmic_spiral_creation() {
        let spiral = LogarithmicSpiral::new(1.0, 0.1);
        assert_eq!(spiral.a, 1.0);
        assert_eq!(spiral.b, 0.1);
    }
    
    #[test]
    fn test_spiral_point_generation() {
        let spiral = LogarithmicSpiral::new(1.0, 0.1);
        let (x, y) = spiral.point_at_angle(0.0);
        assert_eq!(x, 1.0);
        assert_eq!(y, 0.0);
    }
    
    #[test]
    fn test_spiral_multiple_points() {
        let spiral = LogarithmicSpiral::new(1.0, 0.1);
        let points = spiral.generate_points(0.0, 1.0, 5);
        assert_eq!(points.len(), 5);
    }
    
    #[test]
    fn test_3d_spiral_generation() {
        let spiral = LogarithmicSpiral::new(1.0, 0.1);
        let points = spiral.generate_3d_points(0.0, 1.0, 3, 0.5);
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].2, 0.5); // Check z offset
    }
}