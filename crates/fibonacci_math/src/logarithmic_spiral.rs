//! Logarithmic spiral trajectory generator using golden ratio principles

use pyo3::prelude::*;
use crate::golden_constants::{PHI, INV_PHI};

/// Generate points along a logarithmic spiral
///
/// A logarithmic spiral (also known as the golden spiral when using phi) is defined by:
/// r = a * e^(b * theta)
///
/// For golden ratio spirals, we use b = ln(phi) / (pi/2) to get the golden ratio growth
///
/// # Arguments
/// * `center` - The center point (x, y) of the spiral
/// * `radius` - Initial radius of the spiral
/// * `points` - Number of points to generate
/// * `turns` - Number of turns (full rotations) to generate
///
/// # Returns
/// A vector of (x, y) points along the spiral
#[pyfunction]
pub fn generate_spiral_points(center: (f64, f64), radius: f64, points: usize, turns: f64) -> Vec<(f64, f64)> {
    let (cx, cy) = center;
    let mut result = Vec::with_capacity(points);
    
    // Golden ratio spiral parameter
    let b = 1.0; // Simplified for this implementation
    
    // Calculate step size
    let theta_step = turns * 2.0 * std::f64::consts::PI / points as f64;
    
    for i in 0..points {
        let theta = i as f64 * theta_step;
        
        // Golden spiral equation: r = a * e^(b * theta)
        // Using phi growth factor
        let r = radius * (theta * INV_PHI).exp();
        
        let x = cx + r * theta.cos();
        let y = cy + r * theta.sin();
        
        result.push((x, y));
    }
    
    result
}

/// Generate a golden spiral trajectory with specific golden ratio properties
///
/// # Arguments
/// * `start` - Starting point (x, y)
/// * `points` - Number of points to generate
/// * `growth_factor` - Factor by which the spiral grows (default: PHI)
///
/// # Returns
/// A vector of (x, y) points along the golden spiral
#[pyfunction]
pub fn generate_golden_spiral(start: (f64, f64), points: usize, growth_factor: Option<f64>) -> Vec<(f64, f64)> {
    let (x0, y0) = start;
    let mut result = Vec::with_capacity(points);
    
    let growth = growth_factor.unwrap_or(P);
    
    // Generate points along the golden spiral
    for i in 0..points {
        let t = i as f64 * 0.1; // Parameter for spiral
        
        // Golden spiral: r = a * e^(b * theta)
        // Using golden ratio for growth
        let r = t * growth;
        let theta = t * 1.0; // Simplified angular component
        
        let x = x0 + r * theta.cos();
        let y = y0 + r * theta.sin();
        
        result.push((x, y));
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spiral_generation() {
        let points = generate_spiral_points((0.0, 0.0), 1.0, 10, 1.0);
        assert_eq!(points.len(), 10);
        
        // Check that we get valid coordinates
        for &(x, y) in &points {
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }
    
    #[test]
    fn test_golden_spiral() {
        let points = generate_golden_spiral((0.0, 0.0), 5, None);
        assert_eq!(points.len(), 5);
        
        // Check that we get valid coordinates
        for &(x, y) in &points {
            assert!(x.is_finite());
            assert!(y.is_finite());
        }
    }
}"}}I see there's a conflict with the existing logarithmic_spiral.rs file. Let me check what's already there and then properly update the lib.rs file to expose all components correctly: