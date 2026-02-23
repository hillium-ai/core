// Cognitive Safety Validator for Aegis Layer 7

use crate::layer7::validation::ValidationResult;
use crate::layer7::validation::SyntheticInputDetected;

// Import the ReStraV detector
use crate::visual::validator::VisualValidator;
use crate::visual::validator::Image;

/// Cognitive Safety Validator implementation
pub struct CognitiveSafetyValidator {
    #[cfg(feature = "visual-validation")]
    visual_validator: Option<ReStraVDetector>,
}

impl CognitiveSafetyValidator {
    /// Creates a new CognitiveSafetyValidator
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "visual-validation")]
            visual_validator: Some(ReStraVDetector::new()),
            #[cfg(not(feature = "visual-validation"))]
            visual_validator: None,
        }
    }

    /// Validates visual input using ReStraV detector when feature is enabled
    pub fn validate_visual_input(&mut self, frames: &[Image]) -> ValidationResult {
        #[cfg(feature = "visual-validation")]
        if let Some(detector) = &mut self.visual_validator {
            let result = detector.analyze(frames);
            if result.is_synthetic && result.confidence > 0.8 {
                return ValidationResult::Rejected { 
                    reason: SyntheticInputDetected,
                    evidence: "Synthetic input detected by ReStraV detector".to_string()
                };
            }
        }
        
        ValidationResult::Approved
    }
}