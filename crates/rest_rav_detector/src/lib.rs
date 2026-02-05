// ReStraV Visual Validator - Detects AI-generated content in visual inputs

pub mod detector;
pub mod models;
pub mod detector_tests;

pub use detector::{VisualValidator, ReStraVDetector, SyntheticDetectionResult, DetectionThresholds, ValidatorStats};
