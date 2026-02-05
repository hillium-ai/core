// ReStraV Visual Validator - Detects AI-generated content in visual inputs

pub mod detector;
pub mod models;

pub use detector::{VisualValidator, ReStraVDetector, SyntheticDetectionResult, DetectionThresholds, ValidatorStats};
