// Logarithmic spiral trajectory generator

use pyo3::prelude::*;

/// Generate points along a logarithmic spiral
#[pyfunction]
pub fn generate_spiral_points(
    a: f64,
    b: f64,
    start_angle: f64,
    end_angle: f64,
    num_points: usize,
) -> Vec<(f64, f64)> {
    let mut points = Vec::with_capacity(num_points);
    
    // Generate logarithmic spiral points: r = a * e^(b * theta)
    let angle_step = (end_angle - start_angle) / (num_points - 1) as f64;
    
    for i in 0..num_points {
        let angle = start_angle + i as f64 * angle_step;
        let radius = a * (b * angle).exp();
        
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        
        points.push((x, y));
    }
    
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_spiral_points() {
        let points = generate_spiral_points(1.0, 0.1, 0.0, 2.0 * std::f64::consts::PI, 10);
        assert_eq!(points.len(), 10);
        
        // Check that we get valid coordinates
        for (x, y) in &points {
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }
}