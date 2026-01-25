//! Tests for visual validator interface

#[cfg(test)]
mod tests {
    use super::validator::*;
    use crate::visual::validator::{VisualValidator, NoOpValidator, Image, SyntheticDetectionResult, DetectionThresholds};

    #[test]
    fn test_noop_validator_returns_approved() {
        let mut validator = NoOpValidator::new();
        let frames = vec![Image {
            data: vec![],
            width: 100,
            height: 100,
        }];
        
        let result = validator.analyze(&frames);
        assert!(result.approved);
        assert_eq!(result.content_type, "no-op");
    }

    #[test]
    fn test_noop_validator_is_not_enabled() {
        let validator = NoOpValidator::new();
        assert!(!validator.is_enabled());
    }

    #[test]
    fn test_trait_bounds() {
        // Verify that VisualValidator implements Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoOpValidator>();
    }

    #[test]
    fn test_feature_flag_compilation() {
        // This test will only compile if the feature flag is properly set up
        // The test itself doesn't do anything, but it verifies compilation
        let _ = NoOpValidator::new();
    }
}