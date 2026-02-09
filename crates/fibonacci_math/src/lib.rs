//! Fibonacci Math Library for HilliumOS
//! Provides neuro-fibonacci optimized mathematical primitives for robotics

use pyo3::prelude::*;

pub mod golden_constants;
pub mod golden_kalman;
pub mod fibonacci_heap;
pub mod logarithmic_spiral;

/// Python module definition for fibonacci_math
#[pymodule]
fn _fibonacci_math(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export constants
    m.add("PHI", golden_constants::PHI)?;
    m.add("INV_PHI", golden_constants::INV_PHI)?;
    m.add("SQRT_5", golden_constants::SQRT_5)?;
    
    // Export structs
    m.add_class::<golden_kalman::GoldenKalmanFilter>()?;
    m.add_class::<fibonacci_heap::FibonacciHeap>()?;
    m.add_class::<logarithmic_spiral::LogarithmicSpiral>()?;
    
    // Export functions
    m.add_function(wrap_pyfunction!(logarithmic_spiral::generate_spiral_points, m)?)?;
    m.add_function(wrap_pyfunction!(logarithmic_spiral::generate_golden_spiral, m)?)?;
    
    Ok(())
}