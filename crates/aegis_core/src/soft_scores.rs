// Enhanced soft scores implementation with security and threshold features

//! Soft Scores Framework for Gradient Safety Evaluation

use serde::{Deserialize, Serialize};

/// Soft scores for multi-dimensional safety evaluation
///
/// Uses u16 scale (0-1000) for deterministic behavior instead of f32
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SoftScores {
    /// Safety score (0-1000)
    pub safety: u16,
    /// Logical consistency score (0-1000)
    pub logic: u16,
    /// Efficiency score (0-1000)
    pub efficiency: u16,
    /// Ethical score (0-1000)
    pub ethics: u16,
    /// Threshold for determining if scores meet minimum requirements
    pub threshold: f32,
}

impl SoftScores {
    /// Create new SoftScores with all scores clamped to 0-1000 range
    pub fn new(safety: u16, logic: u16, efficiency: u16, ethics: u16, threshold: f32) -> Self {
        Self {
            safety: safety.min(1000),
            logic: logic.min(1000),
            efficiency: efficiency.min(1000),
            ethics: ethics.min(1000),
            threshold: threshold.min(1.0).max(0.0), // Clamp to 0.0-1.0 range
        }
    }

    /// Validate that all scores are within valid range
    pub fn is_valid(&self) -> bool {
        self.safety <= 1000 && 
        self.logic <= 1000 && 
        self.efficiency <= 1000 && 
        self.ethics <= 1000 &&
        self.threshold >= 0.0 &&
        self.threshold <= 1.0
    }

    /// Convert to normalized f32 scores (0.0-1.0)
    pub fn to_normalized(&self) -> NormalizedScores {
        NormalizedScores {
            safety: self.safety as f32 / 1000.0,
            logic: self.logic as f32 / 1000.0,
            efficiency: self.efficiency as f32 / 1000.0,
            ethics: self.ethics as f32 / 1000.0,
        }
    }

    /// Check if scores meet the minimum threshold
    pub fn meets_threshold(&self) -> bool {
        let normalized = self.to_normalized();
        let average = (normalized.safety + normalized.logic + normalized.efficiency + normalized.ethics) / 4.0;
        average >= self.threshold
    }
}

/// Normalized scores for compatibility with existing systems
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct NormalizedScores {
    pub safety: f32,
    pub logic: f32,
    pub efficiency: f32,
    pub ethics: f32,
}

/// Weights for aggregating soft scores
///
/// Must sum to 1.0 for proper weighting
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ScoreWeights {
    pub safety: f32,
    pub logic: f32,
    pub efficiency: f32,
    pub ethics: f32,
}

impl ScoreWeights {
    /// Create new ScoreWeights with validation
    pub fn new(safety: f32, logic: f32, efficiency: f32, ethics: f32) -> Result<Self, String> {
        let total = safety + logic + efficiency + ethics;
        
        // Check if weights sum to approximately 1.0 (allowing for floating point precision)
        if (total - 1.0).abs() > 0.001 {
            return Err(format!("Score weights must sum to 1.0, got {}", total));
        }
        
        Ok(Self { safety, logic, efficiency, ethics })
    }

    /// Get default weights (equal distribution)
    pub fn default_weights() -> Self {
        Self {
            safety: 0.25,
            logic: 0.25,
            efficiency: 0.25,
            ethics: 0.25,
        }
    }
}

/// Safety policy for minimum requirements
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SafetyPolicy {
    /// Minimum safety score
    pub min_safety: u16,
    /// Minimum logic score
    pub min_logic: u16,
    /// Minimum efficiency score
    pub min_efficiency: u16,
    /// Minimum ethics score
    pub min_ethics: u16,
    /// Minimum threshold for overall score
    pub min_threshold: f32,
}

impl SafetyPolicy {
    /// Create new safety policy
    pub fn new(min_safety: u16, min_logic: u16, min_efficiency: u16, min_ethics: u16, min_threshold: f32) -> Self {
        Self {
            min_safety,
            min_logic,
            min_efficiency,
            min_ethics,
            min_threshold: min_threshold.min(1.0).max(0.0),
        }
    }

    /// Check if scores meet policy requirements
    pub fn meets_policy(&self, scores: &SoftScores) -> bool {
        scores.safety >= self.min_safety &&
        scores.logic >= self.min_logic &&
        scores.efficiency >= self.min_efficiency &&
        scores.ethics >= self.min_ethics &&
        scores.threshold >= self.min_threshold
    }
}

/// Validation result with scores
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationResult {
    /// Validation passed
    Approved,
    /// Validation failed
    Rejected,
    /// Validation requires human review
    RequiresHumanReview,
}

/// Validation result with scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResultWithScores {
    /// The validation result
    pub verdict: ValidationResult,
    /// The soft scores used for validation
    pub scores: SoftScores,
    /// The confidence level of the validation
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_scores_creation() {
        let scores = SoftScores::new(500, 750, 250, 1000, 0.7);
        assert_eq!(scores.safety, 500);
        assert_eq!(scores.logic, 750);
        assert_eq!(scores.efficiency, 250);
        assert_eq!(scores.ethics, 1000);
        assert_eq!(scores.threshold, 0.7);
    }

    #[test]
    fn test_soft_scores_clamping() {
        let scores = SoftScores::new(1500, 500, 250, 1000, 1.5); // 1500 > 1000 and 1.5 > 1.0
        assert_eq!(scores.safety, 1000); // Should be clamped to 1000
        assert_eq!(scores.logic, 500);
        assert_eq!(scores.efficiency, 250);
        assert_eq!(scores.ethics, 1000);
        assert_eq!(scores.threshold, 1.0); // Should be clamped to 1.0
    }

    #[test]
    fn test_soft_scores_validity() {
        let scores = SoftScores::new(500, 750, 250, 1000, 0.7);
        assert!(scores.is_valid());
        
        let invalid_scores = SoftScores::new(500, 750, 250, 1500, 0.7); // 1500 > 1000
        assert!(!invalid_scores.is_valid());
        
        let invalid_scores = SoftScores::new(500, 750, 250, 1000, 1.5); // 1.5 > 1.0
        assert!(!invalid_scores.is_valid());
    }

    #[test]
    fn test_normalized_conversion() {
        let scores = SoftScores::new(500, 750, 250, 1000, 0.7);
        let normalized = scores.to_normalized();
        assert_eq!(normalized.safety, 0.5);
        assert_eq!(normalized.logic, 0.75);
        assert_eq!(normalized.efficiency, 0.25);
        assert_eq!(normalized.ethics, 1.0);
    }

    #[test]
    fn test_threshold_check() {
        let scores = SoftScores::new(500, 750, 250, 1000, 0.5); // Average = (0.5 + 0.75 + 0.25 + 1.0) / 4 = 0.5
        assert!(scores.meets_threshold());
        
        let scores = SoftScores::new(500, 750, 250, 1000, 0.6); // Average = 0.5 < 0.6
        assert!(!scores.meets_threshold());
    }

    #[test]
    fn test_score_weights_creation() {
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        assert_eq!(weights.safety, 0.25);
        assert_eq!(weights.logic, 0.25);
        assert_eq!(weights.efficiency, 0.25);
        assert_eq!(weights.ethics, 0.25);
    }

    #[test]
    fn test_score_weights_validation() {
        // This should fail - weights don't sum to 1.0
        let result = ScoreWeights::new(0.3, 0.3, 0.3, 0.3);
        assert!(result.is_err());
        
        // This should succeed - weights sum to 1.0
        let result = ScoreWeights::new(0.25, 0.25, 0.25, 0.25);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safety_policy() {
        let policy = SafetyPolicy::new(500, 500, 500, 500, 0.5);
        let scores = SoftScores::new(750, 750, 750, 750, 0.6);
        assert!(policy.meets_policy(&scores));
        
        let scores = SoftScores::new(250, 750, 750, 750, 0.6); // Safety below threshold
        assert!(!policy.meets_policy(&scores));
        
        let scores = SoftScores::new(750, 750, 750, 750, 0.8); // Threshold below requirement
        assert!(!policy.meets_policy(&scores));
    }
}