//! Verification tests for WP-035 implementation

use crate::soft_scores::{SoftScores, ScoreWeights, ThresholdPolicy, Verdict, determine_verdict_from_scores};
use crate::validation::ValidationResult;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soft_scores_integration() {
        // Test that SoftScores can be created and used
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        assert_eq!(scores.safety, 0.8);
        assert_eq!(scores.logic, 0.9);
        assert_eq!(scores.efficiency, 0.7);
        assert_eq!(scores.ethics, 0.8);
        
        // Test that scores are valid
        assert!(scores.is_valid());
        
        // Test normalization
        let normalized = scores.to_normalized();
        assert!(normalized.safety >= 0.0 && normalized.safety <= 1.0);
    }
    
    #[test]
    fn test_threshold_policy_integration() {
        // Test that ThresholdPolicy works correctly
        let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
        
        // Test scores that should pass
        let passing_scores = SoftScores::new(0.8, 0.9, 0.8, 0.9);
        let verdict = policy.determine_verdict(&passing_scores);
        assert_eq!(verdict, Verdict::Approved);
        
        // Test scores that should fail
        let failing_scores = SoftScores::new(0.6, 0.9, 0.8, 0.9);
        let verdict = policy.determine_verdict(&failing_scores);
        assert_eq!(verdict, Verdict::Rejected);
        
        // Test scores that should require human review
        let human_review_scores = SoftScores::new(0.8, 0.9, 0.8, 0.8);
        let verdict = policy.determine_verdict(&human_review_scores);
        assert_eq!(verdict, Verdict::RequiresHuman);
    }
    
    #[test]
    fn test_validation_result_structure() {
        // Test that ValidationResult can be created with SoftScores
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        let result = ValidationResult {
            verdict: Verdict::Approved,
            scores: scores.clone(),
            reason: "Test validation".to_string(),
        };
        
        assert_eq!(result.scores, scores);
        assert_eq!(result.verdict, Verdict::Approved);
    }
    
    #[test]
    fn test_aggregation_function() {
        // Test the aggregation function
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        
        let aggregated = scores.aggregate(&weights);
        assert!(aggregated >= 0.0 && aggregated <= 1.0);
    }
    
    #[test]
    fn test_determine_verdict_from_scores() {
        // Test the new function that determines verdict from scores
        let scores = SoftScores::new(0.8, 0.9, 0.8, 0.9);
        let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
        
        let verdict = determine_verdict_from_scores(&scores, &policy);
        assert_eq!(verdict, Verdict::Approved);
    }
}