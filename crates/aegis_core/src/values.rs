//! Value Scorecard for Gradient Safety
//!
//! Part of the Soft-Scoring Framework (v8.4+).
//! Enables gradient-based safety evaluation instead of binary pass/fail.

use serde::{Deserialize, Serialize};
use crate::soft_scores::SoftScores;

/// A scorecard representing a value assessment.
/// 
/// Instead of binary "Safe/Unsafe", we use gradient scores (0.0-1.0)
/// to allow the Solver to "climb the optimization hill".
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValueScorecard {
    /// Unique identifier for the value being scored
    pub value_id: String,
    
    /// Score between 0.0 (bad) and 1.0 (good)
    pub score: f32,
    
    /// Evidence/reasoning for the score
    pub evidence: String,
    
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl ValueScorecard {
    /// Create a new scorecard with current timestamp
    pub fn new(value_id: &str, score: f32, evidence: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        
        Self {
            value_id: value_id.to_string(),
            score: score.clamp(0.0, 1.0),
            evidence: evidence.to_string(),
            timestamp_ns,
        }
    }
    
    /// Check if score meets minimum threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
    
    /// Evaluate soft scores for a value
    pub fn evaluate_soft_scores(&self, scores: &SoftScores) -> SoftScores {
        // This method would be used to combine value scores with soft scores
        // For now, we return the scores as-is
        scores.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scorecard_creation() {
        let card = ValueScorecard::new("safety", 0.85, "No dangerous actions detected");
        assert_eq!(card.value_id, "safety");
        assert!(card.score >= 0.0 && card.score <= 1.0);
    }
    
    #[test]
    fn test_score_clamping() {
        let card = ValueScorecard::new("test", 1.5, "Over limit");
        assert_eq!(card.score, 1.0);
        
        let card2 = ValueScorecard::new("test", -0.5, "Under limit");
        assert_eq!(card2.score, 0.0);
    }
    
    #[test]
    fn test_threshold() {
        let card = ValueScorecard::new("efficiency", 0.7, "Moderate efficiency");
        assert!(card.meets_threshold(0.5));
        assert!(!card.meets_threshold(0.9));
    }
}