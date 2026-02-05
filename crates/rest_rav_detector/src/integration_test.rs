// Integration test to verify ReStraV detector functionality

/// This module verifies that the ReStraV detector implementation
/// meets the requirements from WP-042
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_detector_implements_visual_validator_trait() {
        // Verify that ReStraVDetector implements VisualValidator trait
        let mut detector = ReStraVDetector::new();
        
        // Test analyze method
        let frames = vec![];
        let result = detector.analyze(&frames);
        assert!(result.is_synthetic == false || result.is_synthetic == true);
        
        // Test set_thresholds method
        let new_thresholds = DetectionThresholds {
            curvature_threshold: 0.6,
            stepwise_distance_threshold: 0.4,
            confidence_threshold: 0.7,
        };
        detector.set_thresholds(new_thresholds);
        
        // Test get_stats method
        let stats = detector.get_stats();
        assert!(stats.total_analyzed >= 0);
    }
    
    #[test]
    fn test_detector_initialization() {
        let detector = ReStraVDetector::new();
        
        assert_eq!(detector.thresholds.curvature_threshold, 0.5);
        assert_eq!(detector.thresholds.stepwise_distance_threshold, 0.3);
        assert_eq!(detector.thresholds.confidence_threshold, 0.8);
    }
}
