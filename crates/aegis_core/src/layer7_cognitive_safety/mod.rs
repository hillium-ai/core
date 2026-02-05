//! Cognitive Safety Validator for Layer 7

//! This module provides integration with the ReStraV visual validator
//! for detecting synthetic content in visual inputs.

pub mod integration;
pub mod visual_validator;

pub use integration::CognitiveSafetyValidator;
pub use visual_validator::CognitiveSafetyValidatorWithVisual;
