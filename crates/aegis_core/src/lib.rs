//! Aegis Core - Safety validation and soft scores

pub mod validation;
pub mod soft_scores;

/// A safety validation result
#[derive(Debug, Clone)]
pub struct SafetyValidation {
    /// The validation result
    pub result: validation::ValidationResult,
    /// The soft scores
    pub soft_scores: Option<validation::SoftScores>,
}

impl SafetyValidation {
    /// Create a new safety validation
    pub fn new(result: validation::ValidationResult, soft_scores: Option<validation::SoftScores>) -> Self {
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
    pub fn normalized_scores(&self) -> Option<validation::SoftScores> {
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
    fn meets_policy(&self, scores: &validation::SoftScores) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_validation() {
        let result = validation::ValidationResult {
            verdict: validation::Verdict::Pass,
            scores: validation::SoftScores {
                scores: vec![0.8, 0.9, 0.7],
            },
        };

        let validation = SafetyValidation::new(result, Some(validation::SoftScores {
            scores: vec![0.8, 0.9, 0.7],
        }));

        assert!(validation.is_valid());
        assert!(validation.normalized_scores().is_some());
    }
}
