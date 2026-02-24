//! Visual Validator Interface for Aegis Core

#[cfg(feature = "visual-validation")]
pub use restrav_detector::VisualValidator;

#[cfg(feature = "visual-validation")]
pub use restrav_detector::SyntheticDetectionResult;

#[cfg(feature = "visual-validation")]
pub use restrav_detector::DetectionThresholds;

#[cfg(feature = "visual-validation")]
pub use restrav_detector::ValidatorStats;

#[cfg(not(feature = "visual-validation"))]
pub trait VisualValidator {
    fn analyze(&mut self, _frames: &[Image]) -> SyntheticDetectionResult;
    fn set_thresholds(&mut self, _thresholds: DetectionThresholds);
    fn get_stats(&self) -> ValidatorStats;
}

#[cfg(not(feature = "visual-validation"))]
#[derive(Debug, Clone)]
pub struct SyntheticDetectionResult {
    pub is_synthetic: bool,
    pub curvature_score: f32,
    pub stepwise_distance: f32,
    pub confidence: f32,
    pub frame_anomalies: Vec<usize>,
}

#[cfg(not(feature = "visual-validation"))]
#[derive(Debug, Clone)]
pub struct DetectionThresholds {
    pub curvature_threshold: f32,
    pub stepwise_threshold: f32,
    pub confidence_threshold: f32,
}

#[cfg(not(feature = "visual-validation"))]
#[derive(Debug, Clone)]
pub struct ValidatorStats {
    pub total_frames_processed: u64,
    pub synthetic_frames_detected: u64,
    pub average_curvature_score: f32,
    pub average_distance_score: f32,
}

#[cfg(not(feature = "visual-validation"))]
impl Default for ValidatorStats {
    fn default() -> Self {
        Self {
            total_frames_processed: 0,
            synthetic_frames_detected: 0,
            average_curvature_score: 0.0,
            average_distance_score: 0.0,
        }
    }
}

#[cfg(not(feature = "visual-validation"))]
impl Default for DetectionThresholds {
    fn default() -> Self {
        Self {
            curvature_threshold: 0.5,
            stepwise_threshold: 0.3,
            confidence_threshold: 0.8,
        }
    }
}

#[cfg(not(feature = "visual-validation"))]
impl Default for SyntheticDetectionResult {
    fn default() -> Self {
        Self {
            is_synthetic: false,
            curvature_score: 0.0,
            stepwise_distance: 0.0,
            confidence: 0.0,
            frame_anomalies: Vec::new(),
        }
    }
}
