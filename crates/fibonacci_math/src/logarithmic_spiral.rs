// Logarithmic Spiral trajectory generator

#[derive(Debug, Clone)]
pub struct LogarithmicSpiral {
    pub a: f64,  // Initial radius
    pub b: f64,  // Growth factor (related to golden ratio)
}

impl LogarithmicSpiral {
    pub fn new(a: f64, b: f64) -> Self {
        LogarithmicSpiral { a, b }
    }
    
    /// Calculate position at angle theta
    pub fn position(&self, theta: f64) -> (f64, f64) {
        let r = self.a * (self.b * theta).exp();
        (r * theta.cos(), r * theta.sin())
    }
    
    /// Generate a sequence of points along the spiral
    pub fn points(&self, start_theta: f64, end_theta: f64, steps: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::with_capacity(steps);
        let delta = (end_theta - start_theta) / (steps - 1) as f64;
        
        for i in 0..steps {
            let theta = start_theta + i as f64 * delta;
            points.push(self.position(theta));
        }
        
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logarithmic_spiral() {
        let spiral = LogarithmicSpiral::new(1.0, 0.1);
        let pos = spiral.position(0.0);
        assert_eq!(pos, (1.0, 0.0));
        
        let points = spiral.points(0.0, 1.0, 5);
        assert_eq!(points.len(), 5);
    }
}