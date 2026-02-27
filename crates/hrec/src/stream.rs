use serde::{Deserialize, Serialize};

/// Stream types supported by the .hrec format
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StreamType {
    BodyPose = 0,
    HandTracking = 1,
    Haptics = 2,
    Gaze = 3,
}

impl StreamType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(StreamType::BodyPose),
            1 => Some(StreamType::HandTracking),
            2 => Some(StreamType::Haptics),
            3 => Some(StreamType::Gaze),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            StreamType::BodyPose => 0,
            StreamType::HandTracking => 1,
            StreamType::Haptics => 2,
            StreamType::Gaze => 3,
        }
    }
}

/// Information about a single stream in the recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    pub name: String,
    pub stream_type: StreamType,
    pub frequency_hz: u16,
    pub sample_count: u64,
    pub offset: u64,
}

/// A single joint pose (position + rotation + confidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointPose {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub confidence: f32,
}

impl JointPose {
    pub fn new(position: [f32; 3], rotation: [f32; 4], confidence: f32) -> Self {
        JointPose {
            position,
            rotation,
            confidence,
        }
    }
}

/// A body pose sample with timestamp and joint poses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyPoseSample {
    pub timestamp_us: u64,
    pub joints: Vec<JointPose>,
}

impl BodyPoseSample {
    pub fn new(timestamp_us: u64, joints: Vec<JointPose>) -> Self {
        BodyPoseSample { timestamp_us, joints }
    }
}
