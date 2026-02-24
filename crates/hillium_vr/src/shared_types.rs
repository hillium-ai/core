//! Shared data types for VR bridge

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// VR Pose data structure
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrPose {
    #[pyo3(get, set)]
    pub timestamp_ns: u64,
    #[pyo3(get, set)]
    pub position: [f32; 3],
    #[pyo3(get, set)]
    pub rotation: [f32; 4],
}

/// Haptic feedback data
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticFeedback {
    #[pyo3(get, set)]
    pub timestamp_ns: u64,
    #[pyo3(get, set)]
    pub force: f32,
    #[pyo3(get, set)]
    pub location: String,
}

/// Gaze tracking data
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazeData {
    #[pyo3(get, set)]
    pub timestamp_ns: u64,
    #[pyo3(get, set)]
    pub position: [f32; 3],
    #[pyo3(get, set)]
    pub direction: [f32; 3],
}
