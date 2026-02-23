//! ReStraV Visual Validator integration for Aegis Layer 7

use crate::validation::ValidationResult;
use crate::values::Image;
use restrav_validator::{VisualValidator, ReStraVDetector};

/// Cognitive Safety Validator that integrates ReStraV
pub struct CognitiveSafetyValidator {
    visual_validator: Option<ReStraVDetector>,
}

impl CognitiveSafetyValidator {
    /// Creates a new CognitiveSafetyValidator
    pub fn new() -> Self {
        Self {
            visual_validator: Some(ReStraVDetector::new()),
        }
    }
    
    /// Validates visual input using ReStraV detector
    pub fn validate_visual_input(&mut self, frames: &[Image]) -> ValidationResult {
        #[cfg(feature = "visual-validation")]
        if let Some(detector) = &mut self.visual_validator {
            let result = detector.analyze(frames);
            if result.is_synthetic && result.confidence > 0.8 {
                return ValidationResult::Rejected { 
                    reason: "Synthetic input detected",
                    evidence: "ReStraV validation failed",
                };
            }
        }
        ValidationResult::Approved
    }
}