//! Zenoh bridge for VR data streaming

use zenoh::config::Config;
use zenoh::Session;
use zenoh::prelude::sync::SyncResolve;
use bincode;
use crate::shared_types::{VrPose, HapticFeedback, GazeData};

/// Zenoh publisher for VR data
pub struct ZenohPublisher {
    session: Session,
}

impl ZenohPublisher {
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = Config::default();
        let session = zenoh::open(config).res()?;
        Ok(Self { session })
    }
    
    pub fn publish_pose(&self, pose: &VrPose) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/pose";
        let payload = bincode::serialize(pose)?;
        self.session.put(key, payload).res()?;
        Ok(())
    }

    pub fn publish_haptic(&self, haptic: &HapticFeedback) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/haptic";
        let payload = bincode::serialize(haptic)?;
        self.session.put(key, payload).res()?;
        Ok(())
    }
    
    pub fn publish_gaze(&self, gaze: &GazeData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/gaze";
        let payload = bincode::serialize(gaze)?;
        self.session.put(key, payload).res()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vr_pose_serialization() {
        let pose = VrPose {
            timestamp_ns: 1234567890,
            position: [0.0, 1.5, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        let serialized = bincode::serialize(&pose).unwrap();
        let deserialized: VrPose = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.timestamp_ns, pose.timestamp_ns);
    }

    #[test]
    fn test_haptic_serialization() {
        let haptic = HapticFeedback {
            timestamp_ns: 1234567890,
            force: 0.5,
            location: "left_hand".to_string(),
        };
        let serialized = bincode::serialize(&haptic).unwrap();
        let deserialized: HapticFeedback = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.force, haptic.force);
    }

    #[test]
    fn test_gaze_serialization() {
        let gaze = GazeData {
            timestamp_ns: 1234567890,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
        };
        let serialized = bincode::serialize(&gaze).unwrap();
        let deserialized: GazeData = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.position, gaze.position);
    }
}