//! Integration tests for VR bridge functionality

use hillium_vr::{VrBridge, VrPose, HapticFeedback, GazeData};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vr_bridge_creation() {
        let bridge = VrBridge::new();
        assert!(!bridge.streaming);
    }

    #[test]
    fn test_vr_bridge_pose_serialization() {
        let pose = VrPose {
            timestamp_ns: 1234567890,
            position: [0.0, 1.5, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        
        // Test serialization/deserialization
        let serialized = bincode::serialize(&pose).unwrap();
        let deserialized: VrPose = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.timestamp_ns, pose.timestamp_ns);
    }

    #[test]
    fn test_vr_bridge_haptic_serialization() {
        let haptic = HapticFeedback {
            timestamp_ns: 1234567890,
            force: 0.5,
            location: "left_hand".to_string(),
        };
        
        // Test serialization/deserialization
        let serialized = bincode::serialize(&haptic).unwrap();
        let deserialized: HapticFeedback = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.force, haptic.force);
    }

    #[test]
    fn test_vr_bridge_gaze_serialization() {
        let gaze = GazeData {
            timestamp_ns: 1234567890,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
        };
        
        // Test serialization/deserialization
        let serialized = bincode::serialize(&gaze).unwrap();
        let deserialized: GazeData = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.position, gaze.position);
    }
}
