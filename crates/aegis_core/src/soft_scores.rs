//! Soft Scores Framework for Gradient Safety Evaluation

use serde::{Deserialize, Serialize};

/// Soft scores for multi-dimensional safety evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftScores {
    /// Safety score (0-1)
    pub safety: f32,
    /// Logical consistency score (0-1)
    pub logic: f32,
    /// Efficiency score (0-1)
    pub efficiency: f32,
    /// Ethical score (0-1)
    pub ethics: f32,
}

impl SoftScores {
    /// Create new SoftScores
    pub fn new(safety: f32, logic: f32, efficiency: f32, ethics: f32) -> Self {
        Self {
            safety,
            logic,
            efficiency,
            ethics,
        }
    }

    /// Check if the scores are valid (all scores are between 0 and 1)
    pub fn is_valid(&self) -> bool {
        self.safety >= 0.0 && self.safety <= 1.0 &&
        self.logic >= 0.0 && self.logic <= 1.0 &&
        self.efficiency >= 0.0 && self.efficiency <= 1.0 &&
        self.ethics >= 0.0 && self.ethics <= 1.0
    }

    /// Convert scores to normalized form (0-1 range)
    pub fn to_normalized(&self) -> Self {
        Self {
            safety: self.safety.clamp(0.0, 1.0),
            logic: self.logic.clamp(0.0, 1.0),
            efficiency: self.efficiency.clamp(0.0, 1.0),
            ethics: self.ethics.clamp(0.0, 1.0),
        }
    }

    /// Aggregate scores using weighted average
    pub fn aggregate(&self, weights: &ScoreWeights) -> f32 {
        self.safety * weights.safety +
        self.logic * weights.logic +
        self.efficiency * weights.efficiency +
        self.ethics * weights.ethics
    }
}

impl Default for SoftScores {
    fn default() -> Self {
        Self { 
            safety: 0.5,
            logic: 0.5,
            efficiency: 0.5,
            ethics: 0.5
        }
    }
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

/// Threshold-based decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPolicy {
    /// Minimum safety threshold
    pub safety_threshold: f32,
    /// Minimum logic threshold
    pub logic_threshold: f32,
    /// Minimum efficiency threshold
    pub efficiency_threshold: f32,
    /// Minimum ethics threshold
    pub ethics_threshold: f32,
    /// Overall score threshold for requiring human review
    pub overall_threshold: f32,
}

impl ThresholdPolicy {
    /// Create a new threshold policy
    pub fn new(safety_threshold: f32, logic_threshold: f32, efficiency_threshold: f32, ethics_threshold: f32, overall_threshold: f32) -> Self {
        Self {
            safety_threshold,
            logic_threshold,
            efficiency_threshold,
            ethics_threshold,
            overall_threshold,
        }
    }

    /// Check if scores meet policy requirements
    pub fn meets_policy(&self, scores: &SoftScores) -> bool {
        scores.safety >= self.safety_threshold &&
        scores.logic >= self.logic_threshold &&
        scores.efficiency >= self.efficiency_threshold &&
        scores.ethics >= self.ethics_threshold
    }

    /// Determine verdict based on scores and thresholds
    pub fn determine_verdict(&self, scores: &SoftScores) -> Verdict {
        // Check if any individual score is below threshold
        if scores.safety < self.safety_threshold ||
           scores.logic < self.logic_threshold ||
           scores.efficiency < self.efficiency_threshold ||
           scores.ethics < self.ethics_threshold {
            return Verdict::Rejected;
        }

        // Calculate overall score
        let weights = ScoreWeights::default_weights();
        let overall_score = scores.aggregate(&weights);

        // If overall score is below threshold, require human review
        if overall_score < self.overall_threshold {
            Verdict::RequiresHuman
        } else {
            Verdict::Approved
        }
    }
}

/// Validation result with scores
#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_scores_creation() {
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        assert_eq!(scores.safety, 0.8);
        assert_eq!(scores.logic, 0.9);
        assert_eq!(scores.efficiency, 0.7);
        assert_eq!(scores.ethics, 0.8);
    }

    #[test]
    fn test_soft_scores_validity() {
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        assert!(scores.is_valid());
        
        let invalid_scores = SoftScores::new(1.2, 0.9, 0.7, 0.8);
        assert!(!invalid_scores.is_valid());
    }

    #[test]
    fn test_soft_scores_aggregation() {
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        let weights = ScoreWeights::new(0.25, 0.25, 0.25, 0.25).unwrap();
        let aggregated = scores.aggregate(&weights);
        assert_eq!(aggregated, 0.8); // (0.8 + 0.9 + 0.7 + 0.8) / 4 = 0.8
    }

    #[test]
    fn test_threshold_policy() {
        let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.8);
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        let verdict = policy.determine_verdict(&scores);
        assert_eq!(verdict, Verdict::Approved);
        
        let scores = SoftScores::new(0.8, 0.9, 0.6, 0.8);
        let verdict = policy.determine_verdict(&scores);
        assert_eq!(verdict, Verdict::Rejected);
        
        let scores = SoftScores::new(0.8, 0.9, 0.7, 0.8);
        let policy = ThresholdPolicy::new(0.7, 0.7, 0.7, 0.7, 0.9);
        let verdict = policy.determine_verdict(&scores);
        assert_eq!(verdict, Verdict::RequiresHuman);
    }
}