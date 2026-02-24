//! HREC Writer for Project Mirror VR Bridge

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::GazeData;
use crate::HapticFeedback;
use crate::VrPose;

/// HREC file format version
const HREC_VERSION: u32 = 1;

/// HREC Writer for streaming VR data to binary format
pub struct HrecWriter {
    writer: BufWriter<File>,
    /// Number of records written
    record_count: u64,
}

impl HrecWriter {
    /// Creates a new HREC writer for the given file path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            record_count: 0,
        })
    }

    /// Writes the HREC header
    pub fn write_header(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let header = HrecHeader {
            version: HREC_VERSION,
            timestamp_ns: 0, // Will be filled in by actual start time
        };
        let header_bytes = bincode::serialize(&header)?;
        self.writer.write_all(&header_bytes)?;
        Ok(())
    }

    /// Writes a VR Pose record
    pub fn write_pose(&mut self, pose: &VrPose) -> Result<(), Box<dyn std::error::Error>> {
        let record = HrecRecord::Pose(pose.clone());
        let record_bytes = bincode::serialize(&record)?;
        self.writer.write_all(&record_bytes)?;
        self.record_count += 1;
        Ok(())
    }

    /// Writes a Haptic Feedback record
    pub fn write_haptic(
        &mut self,
        haptic: &HapticFeedback,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let record = HrecRecord::Haptic(haptic.clone());
        let record_bytes = bincode::serialize(&record)?;
        self.writer.write_all(&record_bytes)?;
        self.record_count += 1;
        Ok(())
    }

    /// Writes a Gaze Tracking record
    pub fn write_gaze(&mut self, gaze: &GazeData) -> Result<(), Box<dyn std::error::Error>> {
        let record = HrecRecord::Gaze(gaze.clone());
        let record_bytes = bincode::serialize(&record)?;
        self.writer.write_all(&record_bytes)?;
        self.record_count += 1;
        Ok(())
    }

    /// Flushes buffered data to disk
    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.writer.flush()?;
        Ok(())
    }

    /// Gets the number of records written so far
    pub fn record_count(&self) -> u64 {
        self.record_count
    }
}

/// HREC file header
#[derive(Serialize, Deserialize)]
struct HrecHeader {
    version: u32,
    timestamp_ns: u64,
}

/// HREC record types
#[derive(Serialize, Deserialize)]
enum HrecRecord {
    Pose(VrPose),
    Haptic(HapticFeedback),
    Gaze(GazeData),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use tempfile::NamedTempFile;

    #[test]
    fn test_write_header() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = HrecWriter::new(temp_file.path()).unwrap();
        writer.write_header().unwrap();
        assert_eq!(writer.record_count(), 0);
    }

    #[test]
    fn test_write_pose() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = HrecWriter::new(temp_file.path()).unwrap();
        writer.write_header().unwrap();
        let pose = crate::mock_data::mock_pose(12345);
        writer.write_pose(&pose).unwrap();
        assert_eq!(writer.record_count(), 1);
    }

    #[test]
    fn test_write_haptic() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = HrecWriter::new(temp_file.path()).unwrap();
        writer.write_header().unwrap();
        let haptic = crate::mock_data::mock_haptic(12345, 0.5);
        writer.write_haptic(&haptic).unwrap();
        assert_eq!(writer.record_count(), 1);
    }

    #[test]
    fn test_write_gaze() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = HrecWriter::new(temp_file.path()).unwrap();
        writer.write_header().unwrap();
        let gaze = crate::mock_data::mock_gaze(12345);
        writer.write_gaze(&gaze).unwrap();
        assert_eq!(writer.record_count(), 1);
    }

    #[test]
    fn test_flush() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut writer = HrecWriter::new(temp_file.path()).unwrap();
        writer.write_header().unwrap();
        let pose = crate::mock_data::mock_pose(12345);
        writer.write_pose(&pose).unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.record_count(), 1);
    }
}
