use serde::{Deserialize, Serialize};

/// Result of a safety validation check
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall verdict of the validation
    pub verdict: Verdict,
    /// Soft scores for the validation result
    #[serde(rename = "scores")]
    pub scores: SoftScores,
}

/// Verdict of a safety validation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    /// Validation passed
    Pass,
    /// Validation failed
    Fail,
}

/// Soft scores for a validation result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftScores {
    /// The actual scores
    pub scores: Vec<f64>,
}

impl SoftScores {
    /// Check if the scores are valid (all scores are between 0 and 1)
    pub fn is_valid(&self) -> bool {
        self.scores.iter().all(|&score| score >= 0.0 && score <= 1.0)
    }

    /// Convert scores to normalized form (0-1 range)
    pub fn to_normalized(&self) -> Self {
        Self {
            scores: self.scores.iter().map(|&score| score.clamp(0.0, 1.0)).collect(),
        }
    }
}

impl Default for SoftScores {
    fn default() -> Self {
        Self { scores: vec![] }
    }
}
