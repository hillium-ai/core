//! Aegis Core - Safety validation and soft scores

pub mod validation;
pub mod soft_scores;
pub mod values;

#[cfg(feature = "pyo3")]
pub mod layer7;

#[cfg(not(feature = "pyo3"))]
pub mod layer7;

pub use validation::{ValidationResult};
pub use soft_scores::{SoftScores, ScoreWeights, ThresholdPolicy, Verdict};

/// A safety validation result wrapper
#[derive(Debug, Clone)]
pub struct SafetyValidation {
    /// The validation result
    pub result: validation::ValidationResult,
    /// The soft scores
    pub soft_scores: Option<soft_scores::SoftScores>,
}

impl SafetyValidation {
    /// Create a new safety validation
    pub fn new(result: validation::ValidationResult, soft_scores: Option<soft_scores::SoftScores>) -> Self {
        Self {
            result,
            soft_scores,
        }
    }

    /// Check if the validation is valid
    pub fn is_valid(&self) -> bool {
        match &self.soft_scores {
            None => true, // No scores, so we consider it valid
            Some(scores) => scores.is_valid(),
        }
    }

    /// Get the normalized scores
    pub fn normalized_scores(&self) -> Option<soft_scores::SoftScores> {
        self.soft_scores.as_ref().map(|scores| scores.to_normalized())
    }

    /// Check if the validation meets a policy
    pub fn meets_policy(&self, policy: &impl Policy) -> bool {
        match &self.soft_scores {
            None => true, // No scores, so we consider it valid
            Some(scores) => policy.meets_policy(scores),
        }
    }
}

/// A policy for validation
pub trait Policy {
    /// Check if the scores meet the policy
    fn meets_policy(&self, scores: &soft_scores::SoftScores) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_validation() {
        let scores = soft_scores::SoftScores {
            safety: 0.8,
            logic: 0.9,
            efficiency: 0.7,
            ethics: 0.8,
        };
        
        let result = validation::ValidationResult {
            verdict: Verdict::Approved,
            scores: scores.clone(),
            reason: "Test validation".to_string(),
        };

        let validation = SafetyValidation::new(result, Some(scores));

        assert!(validation.is_valid());
        assert!(validation.normalized_scores().is_some());
    }

    #[test]
    fn test_scoring_integration() {
        let scores = soft_scores::SoftScores::new(0.8, 0.9, 0.7, 0.8);
        let weights = soft_scores::ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        let weighted_score = scores.aggregate(&weights);
        assert_eq!(weighted_score, 0.8);
        
        let policy = soft_scores::ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
        let verdict = policy.determine_verdict(&scores);
        assert_eq!(verdict, Verdict::Approved);
    }
}
