// Cognitive Safety Validator for Aegis Layer 7

use crate::layer7::validation::ValidationResult;
use crate::layer7::validation::SyntheticInputDetected;

// Import the ReStraV detector
#[cfg(feature = \
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
            match result {
(detection_result) => {
                    if detection_result.is_synthetic && detection_result.confidence > 0.8 {
                        return ValidationResult::Rejected { 
                            reason: SyntheticInputDetected,
                            evidence: "Synthetic input detected by ReStraV detector".to_string()
                        };
                    }
                }
                Err(_) => {
                    // Log error but continue processing
                    // In a real implementation, this would be logged properly
                }
            }
        }
        
        ValidationResult::Approved
    }
}
