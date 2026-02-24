//! Integration tests for VR bridge functionality

use hillium_vr::{GazeData, HapticFeedback, VrBridge, VrPose};
use tempfile::NamedTempFile;

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

    #[test]
    fn test_hrec_writer_integration() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = hillium_vr::hrec_writer::HrecWriter::new(temp_file.path()).unwrap();
        writer.write_header().unwrap();

        let pose = VrPose {
            timestamp_ns: 1234567890,
            position: [0.0, 1.5, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        writer.write_pose(&pose).unwrap();

        let haptic = HapticFeedback {
            timestamp_ns: 1234567890,
            force: 0.5,
            location: "left_hand".to_string(),
        };
        writer.write_haptic(&haptic).unwrap();

        let gaze = GazeData {
            timestamp_ns: 1234567890,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
        };
        writer.write_gaze(&gaze).unwrap();

        writer.flush().unwrap();
        assert_eq!(writer.record_count(), 3);
    }

    #[test]
    fn test_mock_data_generators() {
        let pose = hillium_vr::mock_data::mock_pose(12345);
        assert_eq!(pose.timestamp_ns, 12345);
        assert_eq!(pose.position, [0.0, 1.5, 0.0]);

        let haptic = hillium_vr::mock_data::mock_haptic(12345, 0.5);
        assert_eq!(haptic.timestamp_ns, 12345);
        assert_eq!(haptic.force, 0.5);
        assert_eq!(haptic.location, "left_hand");

        let gaze = hillium_vr::mock_data::mock_gaze(12345);
        assert_eq!(gaze.timestamp_ns, 12345);
        assert_eq!(gaze.position, [0.0, 0.0, 0.0]);
        assert_eq!(gaze.direction, [0.0, 0.0, -1.0]);
    }
}
