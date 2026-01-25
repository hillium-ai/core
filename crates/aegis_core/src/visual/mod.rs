// Visual validation module

//! This module provides the visual validation interface for detecting synthetic content
//! in visual inputs such as images and video frames.

pub mod validator;
#[cfg(test)]
mod tests;

/// Integration with CognitiveSafetyValidator using feature flag
///
/// When the `visual-validation` feature is enabled, this module integrates with
/// the CognitiveSafetyValidator to provide visual content validation capabilities.
///
/// # Example
///
/// ```
/// use aegis_core::visual::validator::VisualValidator;
/// use aegis_core::visual::validator::NoOpValidator;
///
/// #[cfg(feature = "visual-validation")]
/// let mut validator = MyVisualValidator::new();
///
/// #[cfg(not(feature = "visual-validation"))]
/// let mut validator = NoOpValidator::new();
///
/// let frames = vec![];
/// let result = validator.analyze(&frames);
/// ```
