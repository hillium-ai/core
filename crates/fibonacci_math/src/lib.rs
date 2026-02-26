use pyo3::prelude::*;

pub mod golden_constants;
pub mod golden_kalman;
pub mod fibonacci_heap;
pub mod logarithmic_spiral;

pub use golden_constants::{PHI, INV_PHI, SQRT_5};
pub use golden_kalman::{golden_kalman_gain, GoldenKalmanFilter};
pub use fibonacci_heap::FibonacciHeap;
pub use logarithmic_spiral::LogarithmicSpiral;

#[pyfunction]
fn calculate_golden_gain(q: f64, r: f64, iterations: usize) -> f64 {
    golden_kalman::golden_kalman_gain(q, r, iterations)
}

#[pyfunction]
fn generate_spiral_points(a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    let spiral = LogarithmicSpiral::new(a, b);
    spiral.generate_points(n, None, None)
}

#[pymodule]
fn fibonacci_math(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Export constants
    m.add("P", PHI)?;
    m.add("INV_PHI", INV_PHI)?;
    m.add("SQRT_5", SQRT_5)?;
    
    // Export structs
    m.add_class::<GoldenKalmanFilter>()?;
    m.add_class::<FibonacciHeap>()?;
    m.add_class::<LogarithmicSpiral>()?;
    
    // Export functions
    m.add_function(wrap_pyfunction!(calculate_golden_gain, m)?)?;
    m.add_function(wrap_pyfunction!(generate_spiral_points, m)?)?;
    
    Ok(())
}
