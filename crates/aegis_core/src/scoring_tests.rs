//!
//! Tests for the scoring module

#[cfg(test)]
mod tests {
    use super::scoring::*;
    
    #[test]
    fn test_scoring_module_integration() {
        // Test creating SoftScores
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        assert_eq!(scores.safety, 0.8);
        assert_eq!(scores.logic, 0.9);
        assert_eq!(scores.efficiency, 0.7);
        assert_eq!(scores.ethics, 0.8);
        
        // Test validation
        assert!(scores.is_valid());
        
        // Test normalization
        let normalized = scores.to_normalized();
        assert_eq!(normalized.safety, 0.8);
        assert_eq!(normalized.logic, 0.9);
        assert_eq!(normalized.efficiency, 0.7);
        assert_eq!(normalized.ethics, 0.8);
        
        // Test weights
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        assert_eq!(weights.safety, 0.25);
        assert_eq!(weights.logic, 0.25);
        assert_eq!(weights.efficiency, 0.25);
        assert_eq!(weights.ethics, 0.25);
        
        // Test weighted score computation
        let weighted_score = compute_weighted_score(&scores, &weights);
        assert_eq!(weighted_score, 0.8);
        
        // Test threshold policy
        let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
        let verdict = policy.determine_verdict(&scores);
        assert_eq!(verdict, Verdict::Approved);
    }
    
    #[test]
    fn test_invalid_scores() {
        let invalid_scores = SoftScores::new(1.2, 0.9, 0.7, 0.8);
        assert!(!invalid_scores.is_valid());
        
        let normalized = invalid_scores.to_normalized();
        assert_eq!(normalized.safety, 1.0); // Should be clamped to 1.0
    }
    
    #[test]
    fn test_weight_validation() {
        // This should succeed
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25);
        assert!(weights.is_ok());
        
        // This should fail
        let weights = ScoreWeights::new(0.3, 0.3, 0.3, 0.3);
        assert!(weights.is_err());
    }
}