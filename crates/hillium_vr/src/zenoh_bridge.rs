//! Zenoh bridge for VR data streaming

use zenoh::config::Config;
use zenoh::Session;
use serde::{Deserialize, Serialize};
use bincode;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Zenoh publisher for VR data
pub struct ZenohPublisher {
    session: Session,
}

impl ZenohPublisher {
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = Config::default();
        // Use blocking API for simplicity in this context
        let session = zenoh::open(config).res().unwrap();
        Ok(Self { session })
    }
    
    pub fn publish_pose(&self, pose: &VrPose) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/pose";
        let payload = bincode::serialize(pose)?;
        // Use blocking put operation
        self.session.put(key, payload).res().unwrap();
        Ok(())
    }
    
    pub fn publish_haptic(&self, haptic: &HapticFeedback) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/haptic";
        let payload = bincode::serialize(haptic)?;
        // Use blocking put operation
        self.session.put(key, payload).res().unwrap();
        Ok(())
    }
    
    pub fn publish_gaze(&self, gaze: &GazeData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/gaze";
        let payload = bincode::serialize(gaze)?;
        // Use blocking put operation
        self.session.put(key, payload).res().unwrap();
        Ok(())
    }
}

/// VR Pose data structure
#[derive(Serialize, Deserialize, Clone)]
pub struct VrPose {
    pub timestamp_ns: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

/// Haptic feedback data
#[derive(Serialize, Deserialize, Clone)]
pub struct HapticFeedback {
    pub timestamp_ns: u64,
    pub force: f32,
    pub location: String,
}

/// Gaze tracking data
#[derive(Serialize, Deserialize, Clone)]
pub struct GazeData {
    pub timestamp_ns: u64,
    pub position: [f32; 3],
    pub direction: [f32; 3],
}
