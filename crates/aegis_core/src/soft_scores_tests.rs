//! Comprehensive tests for the soft scoring framework

#[cfg(test)]
mod comprehensive_tests {
    use crate::soft_scores::*;

    #[test]
    fn test_comprehensive_score_validation() {
        // Test valid scores
        let valid_scores = SoftScores::new(0.5, 0.7, 0.8, 0.6);
        assert!(valid_scores.is_valid());
        
        // Test invalid scores (should be clamped)
        let invalid_scores = SoftScores::new(-0.1, 1.2, 0.5, 0.8);
        let normalized = invalid_scores.to_normalized();
        assert_eq!(normalized.safety, 0.0);  // Clamped to 0.0
        assert_eq!(normalized.logic, 1.0);   // Clamped to 1.0
        assert_eq!(normalized.efficiency, 0.5);
        assert_eq!(normalized.ethics, 0.8);
    }

    #[test]
    fn test_weight_validation() {
        // Test valid weights
        let valid_weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        assert_eq!(valid_weights.safety, 0.25);
        assert_eq!(valid_weights.logic, 0.25);
        assert_eq!(valid_weights.efficiency, 0.25);
        assert_eq!(valid_weights.ethics, 0.25);
        
        // Test invalid weights (should fail)
        let invalid_weights = ScoreWeights::new(0.3, 0.3, 0.3, 0.3);
        assert!(invalid_weights.is_err());
    }

    #[test]
    fn test_aggregation_with_different_weights() {
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        
        // Test with equal weights
        let weights_equal = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        let aggregated_equal = scores.aggregate(&weights_equal);
        assert_eq!(aggregated_equal, 0.8); // (0.8 + 0.9 + 0.7 + 0.8) / 4 = 0.8
        
        // Test with different weights
        let weights_different = ScoreWeights::new(0.3, 0.3, 0.2, 0.2).unwrap();
        let aggregated_different = scores.aggregate(&weights_different);
        assert_eq!(aggregated_different, 0.8); // (0.8 * 0.3 + 0.9 * 0.3 + 0.7 * 0.2 + 0.8 * 0.2) = 0.8
    }

    #[test]
    fn test_threshold_policy_comprehensive() {
        // Test policy with default thresholds
        let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
        
        // Test scores that meet policy
        let scores_meet = SoftScores::new(0.8, 0.9, 0.8, 0.8);
        let verdict_meet = policy.determine_verdict(&scores_meet);
        assert_eq!(verdict_meet, Verdict::Approved);
        
        // Test scores that fail individual threshold
        let scores_fail_individual = SoftScores::new(0.8, 0.9, 0.6, 0.8);
        let verdict_fail_individual = policy.determine_verdict(&scores_fail_individual);
        assert_eq!(verdict_fail_individual, Verdict::Rejected);
        
        // Test scores that meet individual but fail overall
        let scores_fail_overall = SoftScores::new(0.8, 0.9, 0.8, 0.8);
        let policy_low_threshold = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.9);
        let verdict_fail_overall = policy_low_threshold.determine_verdict(&scores_fail_overall);
        assert_eq!(verdict_fail_overall, Verdict::RequiresHuman);
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that existing code still works
        let default_result = ValidationResult::default();
        // Depending on implementations, check logical defaults
        // Assuming default() creates something usable
        assert!(default_result.scores.is_valid());
    }

    #[test]
    fn test_edge_cases() {
        // Test with exactly 0.0 and 1.0 values
        let edge_scores = SoftScores::new(0.0, 1.0, 0.0, 1.0);
        assert!(edge_scores.is_valid());
        
        // Test aggregation with edge values
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        let aggregated = edge_scores.aggregate(&weights);
        assert_eq!(aggregated, 0.5); // (0.0 + 1.0 + 0.0 + 1.0) / 4 = 0.5
    }
}
