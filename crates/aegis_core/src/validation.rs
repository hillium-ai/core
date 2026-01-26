use serde::{Deserialize, Serialize};
use crate::soft_scores::SoftScores;

/// Result of a safety validation check
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall verdict of the validation
    pub verdict: Verdict,
    /// Soft scores for the validation result
    pub scores: SoftScores,
    /// Reason for the validation result
    pub reason: String,
}

/// Verdict of a safety validation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    /// Validation passed
    Approved,
    /// Validation failed
    Rejected,
    /// Validation requires human review
    RequiresHuman,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            verdict: Verdict::RequiresHuman,
            scores: SoftScores::default(),
            reason: "Default initialization".to_string(),
        }
    }
}
