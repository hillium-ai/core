use pyo3::prelude::*;

#[pymodule]
fn hillium_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}

#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}