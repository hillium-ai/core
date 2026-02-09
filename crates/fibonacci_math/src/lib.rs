//! Fibonacci Math Library with Golden Ratio optimizations

pub mod golden_constants;
pub mod golden_kalman;
pub mod fibonacci_heap;
pub mod logarithmic_spiral;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn _fibonacci_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<golden_kalman::GoldenKalmanFilter>()?;
    m.add_class::<fibonacci_heap::FibonacciHeap>()?;
    m.add_function(wrap_pyfunction!(logarithmic_spiral::generate_spiral_points, m)?)?;
    m.add("PHI", golden_constants::PHI)?;
    m.add("INV_PHI", golden_constants::INV_PHI)?;
    m.add("SQRT_5", golden_constants::SQRT_5)?;
    Ok(())
}