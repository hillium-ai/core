//! Logarithmic Spiral trajectory generator

/// Logarithmic Spiral trajectory generator
pub struct LogarithmicSpiral {
    /// Initial radius
    a: f64,
    /// Growth rate (related to golden ratio)
    b: f64,
    /// Current angle
    theta: f64,
}

impl LogarithmicSpiral {
    /// Create a new logarithmic spiral
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b, theta: 0.0 }
    }
    
    /// Generate next point on the spiral
    pub fn next_point(&mut self) -> (f64, f64) {
        let x = self.a * (self.b * self.theta).cos();
        let y = self.a * (self.b * self.theta).sin();
        self.theta += 0.1; // Increment angle
        (x, y)
    }
    
    /// Generate a sequence of points
    pub fn points(&mut self, count: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(count);
        for _ in 0..count {
            points.push(self.next_point());
        }
        points
    }
    
    /// Reset the spiral
    pub fn reset(&mut self) {
        self.theta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spiral_creation() {
        let mut spiral = LogarithmicSpiral::new(1.0, 0.5);
        let point = spiral.next_point();
        assert_eq!(point.0, 1.0); // First point should be at (1, 0)
        assert_eq!(point.1, 0.0);
    }
    
    #[test]
    fn test_spiral_points() {
        let mut spiral = LogarithmicSpiral::new(1.0, 0.5);
        let points = spiral.points(3);
        assert_eq!(points.len(), 3);
        
        // Check that we get valid coordinates
        for (x, y) in &points {
            assert!(*x >= 0.0 || *x <= 0.0); // Should be valid f64
            assert!(*y >= 0.0 || *y <= 0.0); // Should be valid f64
        }
    }
}