use pyo3::prelude::*;

mod golden_constants;
mod golden_kalman;
mod fibonacci_heap;
mod logarithmic_spiral;

use golden_constants::*;
use golden_kalman::*;
use fibonacci_heap::*;
use logarithmic_spiral::*;

#[pyfunction]
fn calculate_golden_gain(q: f64, r: f64, iterations: usize) -> f64 {
    golden_kalman::golden_kalman_gain(q, r, iterations)
}

#[pyfunction]
fn generate_spiral_points(a: f64, b: f64, n: usize) -> Vec<(f64, f64)> {
    let spiral = LogarithmicSpiral::new(a, b);
    spiral.generate_points(n)
}

#[pymodule]
fn _fibonacci_math(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export constants
    m.add(\
    m.add("INV_PHI", INV_PHI)?;
    m.add("SQRT_5", SQRT_5)?;
    
    // Export structs
    m.add_class::<GoldenKalmanFilter>()?;
    m.add_class::<FibonacciHeap>()?;
    m.add_class::<LogarithmicSpiral>()?;
    
    // Export functions
    m.add_function(wrap_pyfunction!(generate_spiral_points, m)?)?;
    
   (())
}
