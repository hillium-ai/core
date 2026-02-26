//! Cognitive Safety Validator for Project Mirror

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(not(feature = "pyo3"))]
use serde::{Deserialize, Serialize};

/// Cognitive safety validation result
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(not(feature = "pyo3"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSafetyResult {
    #[cfg(feature = "pyo3")]
    #[pyo3(get, set)]
    pub is_safe: bool,
    #[cfg(not(feature = "pyo3"))]
    pub is_safe: bool,
    #[cfg(feature = "pyo3")]
    #[pyo3(get, set)]
    pub confidence: f32,
    #[cfg(not(feature = "pyo3"))]
    pub confidence: f32,
    #[cfg(feature = "pyo3")]
    #[pyo3(get, set)]
    pub reason: String,
    #[cfg(not(feature = "pyo3"))]
    pub reason: String,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl CognitiveSafetyResult {
    #[new]
    pub fn new(is_safe: bool, confidence: f32, reason: String) -> Self {
        Self {
            is_safe,
            confidence,
            reason,
        }
    }
}

/// Cognitive Safety Validator
#[cfg(feature = "pyo3")]
#[pyclass]
#[derive(Clone)]
#[cfg(not(feature = "pyo3"))]
#[derive(Clone)]
pub struct CognitiveSafetyValidator {
    threshold: f32,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl CognitiveSafetyValidator {
    #[new]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Validate cognitive safety for a given input
    pub fn validate(&self, input: &str) -> PyResult<CognitiveSafetyResult> {
        let is_safe = input.len() > 0 && input.len() < 10000;
        let confidence = if is_safe { 0.95 } else { 0.0 };
        let reason = if is_safe {
            "Input length within acceptable range".to_string()
        } else {
            "Input length out of range".to_string()
        };

        Ok(CognitiveSafetyResult::new(is_safe, confidence, reason))
    }
}

#[cfg(not(feature = "pyo3"))]
impl CognitiveSafetyValidator {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn validate(&self, input: &str) -> CognitiveSafetyResult {
        let is_safe = input.len() > 0 && input.len() < 10000;
        let confidence = if is_safe { 0.95 } else { 0.0 };
        let reason = if is_safe {
            "Input length within acceptable range".to_string()
        } else {
            "Input length out of range".to_string()
        };

        CognitiveSafetyResult::new(is_safe, confidence, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_safe_input() {
        let validator = CognitiveSafetyValidator::new(0.8);
        let result = validator.validate("Hello, world!");
        assert!(result.is_safe);
    }

    #[test]
    fn test_validator_unsafe_input() {
        let validator = CognitiveSafetyValidator::new(0.8);
        let long_input = "x".repeat(10001);
        let result = validator.validate(&long_input);
        assert!(!result.is_safe);
    }
}
