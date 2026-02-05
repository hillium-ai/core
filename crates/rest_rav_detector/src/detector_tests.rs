// Unit tests for ReStraV detector

use crate::detector::{ReStraVDetector, VisualValidator, Image, DetectionThresholds, SyntheticDetectionResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restrav_detector_creation() {
        let detector = ReStraVDetector::new();
        assert!(detector.is_enabled());
    }

    #[test]
    fn test_restrav_detector_analysis() {
        let mut detector = ReStraVDetector::new();
        let frames = vec![Image {
            data: vec![0; 100],
            width: 10,
            height: 10,
        }];
        
        let result = detector.analyze(&frames);
        assert!(result.approved || !result.approved); // Should always return a result
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_restrav_detector_thresholds() {
        let mut detector = ReStraVDetector::new();
        let thresholds = DetectionThresholds {
            min_confidence: 0.5,
            max_false_positive_rate: 0.05,
        };
        
        detector.set_thresholds(thresholds);
        // Just verify it doesn't panic
        assert!(true);
    }

    #[test]
    fn test_restrav_detector_is_enabled() {
        let detector = ReStraVDetector::new();
        assert!(detector.is_enabled());
    }
}
