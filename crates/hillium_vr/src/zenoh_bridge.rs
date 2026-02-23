//! Zenoh bridge for VR data streaming

use zenoh::config::Config;
use zenoh::Session;
use bincode;
use crate::{VrPose, HapticFeedback, GazeData};

/// Zenoh publisher for VR data
pub struct ZenohPublisher {
    session: Session,
}

impl ZenohPublisher {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = Config::default();
        let session = zenoh::open(config).res().await?;
        Ok(Self { session })
    }
    
    pub async fn publish_pose(&self, pose: &VrPose) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/pose";
        let payload = bincode::serialize(pose)?;
        self.session.put(key, payload).await?;
        Ok(())
    }
    
    pub async fn publish_haptic(&self, haptic: &HapticFeedback) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/haptic";
        let payload = bincode::serialize(haptic)?;
        self.session.put(key, payload).await?;
        Ok(())
    }
    
    pub async fn publish_gaze(&self, gaze: &GazeData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let key = "hillium/vr/gaze";
        let payload = bincode::serialize(gaze)?;
        self.session.put(key, payload).await?;
        Ok(())
    }
}