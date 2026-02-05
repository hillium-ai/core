// Visual validation module

//! This module provides the visual validation interface for detecting synthetic content
//! in visual inputs such as images and video frames.

pub mod validator;
pub use validator::ReStraVDetector;
#[cfg(test)]
mod tests;

/// Integration with CognitiveSafetyValidator using feature flag

/// When the `visual-validation` feature is enabled, this module integrates with
/// the CognitiveSafetyValidator to provide visual content validation capabilities.

/// # Example

///