//! Mock data generators for VR Bridge testing

use crate::{VrPose, HapticFeedback, GazeData};

/// Generates a mock VR pose
pub fn mock_pose(timestamp_ns: u64) -> VrPose {
    VrPose {
        timestamp_ns,
        position: [0.0, 1.5, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
    }
}

/// Generates a mock haptic feedback
pub fn mock_haptic(timestamp_ns: u64, force: f32) -> HapticFeedback {
    HapticFeedback {
        timestamp_ns,
        force,
        location: "left_hand".to_string(),
    }
}

/// Generates a mock gaze data
pub fn mock_gaze(timestamp_ns: u64) -> GazeData {
    GazeData {
        timestamp_ns,
        position: [0.0, 0.0, 0.0],
        direction: [0.0, 0.0, -1.0],
    }
}