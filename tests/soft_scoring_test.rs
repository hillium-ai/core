use aegis_core::soft_scores::{SoftScores, ScoreWeights, ThresholdPolicy, Verdict};
use aegis_core::validation::ValidationResult;

#[test]
fn test_soft_scores_structure() {
    let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
    assert_eq!(scores.safety, 0.8);
    assert_eq!(scores.logic, 0.9);
    assert_eq!(scores.efficiency, 0.7);
    assert_eq!(scores.ethics, 0.8);
    
    assert!(scores.is_valid());
    
    let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
    let aggregated = scores.aggregate(&weights);
    assert_eq!(aggregated, 0.8);
}

#[test]
fn test_threshold_policy() {
    let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
    let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
    
    // Test that it meets policy
    assert!(policy.meets_policy(&scores));
    
    // Test verdict determination
    let verdict = policy.determine_verdict(&scores);
    assert_eq!(verdict, Verdict::Approved);
}

#[test]
fn test_validation_result_structure() {
    let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
    let result = ValidationResult {
        verdict: Verdict::Approved,
        scores: scores.clone(),
        reason: "Test".to_string(),
    };
    
    assert_eq!(result.scores.safety, 0.8);
    assert_eq!(result.verdict, Verdict::Approved);
}