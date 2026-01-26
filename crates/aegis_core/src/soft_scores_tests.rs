//! Tests for the Soft Scores Framework

#[cfg(test)]
mod tests {
    use crate::soft_scores::*;
    
    #[test]
    fn test_soft_scores_creation() {
        let scores = SoftScores::new(500, 750, 250, 1000);
        assert_eq!(scores.safety, 500);
        assert_eq!(scores.logic, 750);
        assert_eq!(scores.efficiency, 250);
        assert_eq!(scores.ethics, 1000);
    }

    #[test]
    fn test_soft_scores_clamping() {
        let scores = SoftScores::new(1500, 500, 250, 1000); // 1500 > 1000
        assert_eq!(scores.safety, 1000); // Should be clamped to 1000
        assert_eq!(scores.logic, 500);
        assert_eq!(scores.efficiency, 250);
        assert_eq!(scores.ethics, 1000);
    }

    #[test]
    fn test_soft_scores_validity() {
        let scores = SoftScores::new(500, 750, 250, 1000);
        assert!(scores.is_valid());
        
        let invalid_scores = SoftScores::new(500, 750, 250, 1500); // 1500 > 1000
        assert!(!invalid_scores.is_valid());
    }

    #[test]
    fn test_normalized_scores() {
        let scores = SoftScores::new(500, 750, 250, 1000);
        let normalized = scores.to_normalized();
        assert_eq!(normalized.safety, 0.5);
        assert_eq!(normalized.logic, 0.75);
        assert_eq!(normalized.efficiency, 0.25);
        assert_eq!(normalized.ethics, 1.0);
    }

    #[test]
    fn test_score_weights() {
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25);
        assert_eq!(weights.safety, 0.25);
        assert_eq!(weights.logic, 0.25);
        assert_eq!(weights.efficiency, 0.25);
        assert_eq!(weights.ethics, 0.25);
    }

    #[test]
    fn test_safety_policy() {
        let policy = SafetyPolicy::new(0.5, 0.5, 0.5, 0.5);
        assert_eq!(policy.safety, 0.5);
        assert_eq!(policy.logic, 0.5);
        assert_eq!(policy.efficiency, 0.5);
        assert_eq!(policy.ethics, 0.5);
        
        let scores = SoftScores::new(750, 750, 750, 750);
        assert!(policy.meets_policy(&scores));
        
        let scores = SoftScores::new(250, 750, 750, 750); // Safety below threshold
        assert!(!policy.meets_policy(&scores));
    }

    #[test]
    fn test_aggregation() {
        let scores = SoftScores::new(500, 750, 250, 1000);
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25);
        let normalized = scores.to_normalized();
        let aggregated = (normalized.safety * weights.safety) + 
                        (normalized.logic * weights.logic) + 
                        (normalized.efficiency * weights.efficiency) + 
                        (normalized.ethics * weights.ethics);
        assert_eq!(aggregated, 0.5); // (0.5 * 0.25) + (0.75 * 0.25) + (0.25 * 0.25) + (1.0 * 0.25) = 0.5
    }
}
