//! Cognitive Safety Validator for Project Mirror

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Cognitive safety validation result
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveSafetyResult {
    #[pyo3(get, set)]
    pub is_safe: bool,
    #[pyo3(get, set)]
    pub confidence: f32,
    #[pyo3(get, set)]
    pub reason: String,
}

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
#[pyclass]
#[derive(Clone)]
pub struct CognitiveSafetyValidator {
    threshold: f32,
}

#[pymethods]
impl CognitiveSafetyValidator {
    #[new]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Validate cognitive safety for a given input
    pub fn validate(&self, input: &str) -> PyResult<CognitiveSafetyResult> {
        // Simple validation logic - in production this would use ML models
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_safe_input() {
        let validator = CognitiveSafetyValidator::new(0.8);
        let result = validator.validate("Hello, world!");
        assert!(result.is_ok());
        assert!(result.unwrap().is_safe);
    }

    #[test]
    fn test_validator_unsafe_input() {
        let validator = CognitiveSafetyValidator::new(0.8);
        let long_input = "x".repeat(10001);
        let result = validator.validate(&long_input);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_safe);
    }
}
