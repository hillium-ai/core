use serde::{Deserialize, Serialize};

/// Magic number for .hrec files: "HREC"
pub const HREC_MAGIC: u32 = 0x48524543;

/// Current format version
pub const HREC_VERSION: u16 = 1;

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

/// Header of the .hrec file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub magic: u32,
    pub version: u16,
    pub session_id: String,
    pub duration_ms: u64,
    pub streams: Vec<StreamInfo>,
}

impl Header {
    pub fn new(session_id: String, duration_ms: u64, streams: Vec<StreamInfo>) -> Self {
        Header {
            magic: HREC_MAGIC,
            version: HREC_VERSION,
            session_id,
            duration_ms,
            streams,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.magic == HREC_MAGIC && self.version == HREC_VERSION
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(HREC_MAGIC, 0x48524543);
    }

    #[test]
    fn test_version() {
        assert_eq!(HREC_VERSION, 1);
    }

    #[test]
    fn test_stream_type_from_u8() {
        assert_eq!(StreamType::from_u8(0), Some(StreamType::BodyPose));
        assert_eq!(StreamType::from_u8(1), Some(StreamType::HandTracking));
        assert_eq!(StreamType::from_u8(2), Some(StreamType::Haptics));
        assert_eq!(StreamType::from_u8(3), Some(StreamType::Gaze));
        assert_eq!(StreamType::from_u8(99), None);
    }

    #[test]
    fn test_stream_type_as_u8() {
        assert_eq!(StreamType::BodyPose.as_u8(), 0);
        assert_eq!(StreamType::HandTracking.as_u8(), 1);
        assert_eq!(StreamType::Haptics.as_u8(), 2);
        assert_eq!(StreamType::Gaze.as_u8(), 3);
    }

    #[test]
    fn test_header_is_valid() {
        let header = Header::new(String::from("test"), 1000, Vec::new());
        assert!(header.is_valid());
    }

    #[test]
    fn test_header_is_valid_with_future_version() {
        let mut header = Header::new(String::from("test"), 1000, Vec::new());
        header.version = 2;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_header_is_invalid_with_wrong_magic() {
        let mut header = Header::new(String::from("test"), 1000, Vec::new());
        header.magic = 0xDEADBEEF;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_joint_pose_new() {
        let pose = JointPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], 0.95);
        assert_eq!(pose.position, [1.0, 2.0, 3.0]);
        assert_eq!(pose.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(pose.confidence, 0.95);
    }

    #[test]
    fn test_body_pose_sample_new() {
        let joints = vec![JointPose::new([0.0; 3], [0.0; 4], 1.0)];
        let sample = BodyPoseSample::new(123456789, joints);
        assert_eq!(sample.timestamp_us, 123456789);
        assert_eq!(sample.joints.len(), 1);
    }
}
