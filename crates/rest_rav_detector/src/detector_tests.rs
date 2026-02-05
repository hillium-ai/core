// Unit tests for ReStraV detector implementation

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_detector() {
        let detector = ReStraVDetector::new();
        assert_eq!(detector.thresholds.curvature_threshold, 0.5);
        assert_eq!(detector.thresholds.stepwise_distance_threshold, 0.3);
        assert_eq!(detector.thresholds.confidence_threshold, 0.8);
    }
    
    #[test]
    fn test_set_thresholds() {
        let mut detector = ReStraVDetector::new();
        let new_thresholds = DetectionThresholds {
            curvature_threshold: 0.7,
            stepwise_distance_threshold: 0.4,
            confidence_threshold: 0.9,
        };
        detector.set_thresholds(new_thresholds.clone());
        assert_eq!(detector.thresholds, new_thresholds);
    }
    
    #[test]
    fn test_analyze_empty_frames() {
        let mut detector = ReStraVDetector::new();
        let result = detector.analyze(&[]);
        assert_eq!(result.is_synthetic, false);
        assert_eq!(result.curvature_score, 0.0);
        assert_eq!(result.stepwise_distance, 0.0);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.frame_anomalies.len(), 0);
    }
    
    #[test]
    fn test_get_stats() {
        let detector = ReStraVDetector::new();
        let stats = detector.get_stats();
        assert_eq!(stats.total_analyzed, 0);
        assert_eq!(stats.synthetic_detected, 0);
        assert_eq!(stats.avg_curvature_score, 0.0);
        assert_eq!(stats.avg_stepwise_distance, 0.0);
    }
    
    #[test]
    fn test_analyze_with_mock_data() {
        let mut detector = ReStraVDetector::new();
        let frames = vec![
            Image {
                data: vec![0; 100],
                width: 640,
                height: 480,
            },
            Image {
                data: vec![0; 100],
                width: 640,
                height: 480,
            }
        ];
        let result = detector.analyze(&frames);
        // Mock implementation should return some values
        assert_eq!(result.is_synthetic, false); // Based on mock values
        assert!(result.curvature_score >= 0.0);
        assert!(result.stepwise_distance >= 0.0);
        assert!(result.confidence >= 0.0);
        assert!(result.frame_anomalies.len() >= 0);
    }
}
