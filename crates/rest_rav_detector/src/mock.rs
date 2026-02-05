//! Mock implementation for non-GPU environments

use crate::detector::{Image, VisualValidator, SyntheticDetectionResult, DetectionThresholds, ValidatorStats};

/// Mock ReStraV Detector that returns deterministic results
pub struct MockReStraVDetector;

impl MockReStraVDetector {
    pub fn new() -> Self {
        Self
    }
}

impl VisualValidator for MockReStraVDetector {
    fn analyze(&mut self, _frames: &[Image]) -> SyntheticDetectionResult {
        // Return deterministic results for testing
        SyntheticDetectionResult {
            approved: true,
            confidence: 0.95,
            content_type: "real".to_string(),
        }
    }
    
    fn set_thresholds(&mut self, _thresholds: DetectionThresholds) {
        // Mock implementation - no-op
    }
    
    fn is_enabled(&self) -> bool {
        // Mock implementation - always enabled
        true
    }
}
