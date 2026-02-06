# ReStraV Visual Validator

This is the visual validator implementation for detecting AI-generated content (deepfakes, synthetic video) using perceptual straightening techniques.

## Implementation Details

This module provides:

1. `ReStraVDetector` struct implementing the `VisualValidator` trait
2. `analyze()` method for detecting synthetic content in video frames
3. `set_thresholds()` method for configuring detection parameters
4. `get_stats()` method for retrieving validator statistics

## Interface Contract

The implementation satisfies the interface contract defined in the test files:

- `analyze(&mut self, frames: &[Image]) -> SyntheticDetectionResult`
- `set_thresholds(&mut self, thresholds: DetectionThresholds)`
- `get_stats(&self) -> ValidatorStats`

## Feature Flag

This implementation is compatible with the `visual-validation` feature flag as specified in the documentation.

## Testing

All tests in `tests/test_restrav_implementation.py` and `tests/test_visual_validator.py` pass successfully.
