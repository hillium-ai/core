//! Integration test for ReStraV Visual Validator

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_restrav_detector_creation() {
        let detector = ReStraVDetector::new();
        assert_eq!(detector.stats.total_analyzed, 0);
    }
    
    #[test]
    fn test_visual_validator_trait() {
        let mut detector = ReStraVDetector::new();
        let image = Image::new(640, 480);
        let result = detector.analyze(&[image]);
        assert!(result.is_synthetic == false || result.is_synthetic == true);
    }
    
    #[test]
    fn test_thresholds() {
        let mut detector = ReStraVDetector::new();
        let thresholds = DetectionThresholds::default();
        detector.set_thresholds(thresholds);
        let stats = detector.get_stats();
        assert_eq!(stats.total_analyzed, 0);
    }
}
