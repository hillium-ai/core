use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::env;

use lz4_flex::compress;

use crate::header::{BodyPoseSample, Header, StreamInfo};

#[cfg(test)]
use crate::reader::HrecReader;

/// HrecWriter handles writing .hrec files with LZ4 compression
/// HrecWriter handles writing .hrec files with LZ4 compression
pub struct HrecWriter {
    header: Header,
    file: BufWriter<File>,
}

impl HrecWriter {
    /// Create a new HrecWriter for the given file path
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);

        Ok(HrecWriter {
            header: Header::new(String::new(), 0, Vec::new()),
            file: buf_writer,
        })
    }

    /// Initialize the writer with header information
    pub fn init(
        mut self,
        session_id: String,
        streams: Vec<StreamInfo>,
    ) -> std::io::Result<Self> {
        self.header = Header::new(session_id, 0, streams);
        Ok(self)
    }

    /// Write a body pose sample to the body_pose stream
    pub fn write_body_pose(&mut self, sample: &BodyPoseSample) -> std::io::Result<()> {
        // Compress the sample using LZ4
        let data = bincode::serialize(sample)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let compressed = compress(&data);
        
        // Write compressed data with size prefix
        let size = compressed.len() as u32;
        self.file.write_all(&size.to_le_bytes())?;
        self.file.write_all(&compressed)?;
        
        Ok(())
    }

    /// Write a hand pose sample to the specified hand stream
    pub fn write_hand_pose(
        &mut self,
        _hand: &str,
        sample: &BodyPoseSample,
    ) -> std::io::Result<()> {
        // Compress the sample using LZ4
        let data = bincode::serialize(sample)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let compressed = compress(&data);
        
        // Write compressed data with size prefix
        let size = compressed.len() as u32;
        self.file.write_all(&size.to_le_bytes())?;
        self.file.write_all(&compressed)?;
        Ok(())
    }

    /// Write haptics data
    pub fn write_haptics(&mut self, _timestamp_us: u64, _data: &[f32]) -> std::io::Result<()> {
        // TODO: Implement haptics writing
        Ok(())
    }

    /// Write gaze data
    pub fn write_gaze(&mut self, _timestamp_us: u64, _gaze_dir: [f32; 3]) -> std::io::Result<()> {
        // TODO: Implement gaze writing
        Ok(())
    }

    /// Finalize the recording and write the header
    pub fn finalize(&mut self) -> std::io::Result<()> {
        // Flush buffered data first
        self.file.get_mut().sync_all()?;
        
        // Write header at the END (after all data)
        let header_data = bincode::serialize(&self.header)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        self.file.write_all(&header_data.len().to_le_bytes())?;
        self.file.write_all(&header_data)?;
        
        self.file.get_mut().sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{JointPose, StreamType};

    #[test]
    fn test_writer_new() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_writer_new.hrec");
        let writer = HrecWriter::new(&path);
        assert!(writer.is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_init() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_writer_init.hrec");
        let mut writer = HrecWriter::new(&path).unwrap();
        let streams = vec![StreamInfo {
            name: String::from("body_pose"),
            stream_type: StreamType::BodyPose,
            frequency_hz: 60,
            sample_count: 0,
            offset: 0,
        }];
        let result = writer.init(String::from("test_session"), streams);
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_finalize() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_writer_finalize.hrec");
        let mut writer = HrecWriter::new(&path).unwrap();
        let streams = vec![StreamInfo {
            name: String::from("body_pose"),
            stream_type: StreamType::BodyPose,
            frequency_hz: 60,
            sample_count: 0,
            offset: 0,
        }];
        writer = writer.init(String::from("test_session"), streams).unwrap();
        assert!(writer.finalize().is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_write_body_pose() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_writer_write_body_pose.hrec");
        let mut writer = HrecWriter::new(&path).unwrap();
        let streams = vec![StreamInfo {
            name: String::from("body_pose"),
            stream_type: StreamType::BodyPose,
            frequency_hz: 60,
            sample_count: 0,
            offset: 0,
        }];
        writer = writer.init(String::from("test_session"), streams).unwrap();
        
        let joints = vec![JointPose::new([0.0; 3], [0.0; 4], 1.0)];
        let sample = BodyPoseSample::new(123456789, joints);
        assert!(writer.write_body_pose(&sample).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_roundtrip() {
        let path = "test_writer_roundtrip.hrec";
        
        // Write
        {
            let mut writer = HrecWriter::new(path).unwrap();
            let streams = vec![StreamInfo {
                name: String::from("body_pose"),
                stream_type: StreamType::BodyPose,
                frequency_hz: 60,
                sample_count: 0,
                offset: 0,
            }];
            writer = writer.init(String::from("test_session"), streams).unwrap();
            
            let joints = vec![JointPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], 0.95)];
            let sample = BodyPoseSample::new(123456789, joints);
            writer.write_body_pose(&sample).unwrap();
            writer.finalize().unwrap();
        }
        
        // Verify file exists and can be read
        assert!(std::path::Path::new(path).exists(), "File does not exist: {}", path);
        
        // Check if file exists
        assert!(std::path::Path::new(path).exists(), "File does not exist: {}", path);
        
        // Read back
        let reader = match HrecReader::open(path) {
            Ok(r) => r,
            Err(e) => panic!("Failed to open file '{}': {}", path, e),
        };
        assert_eq!(reader.header().session_id, "test_session");
        assert_eq!(reader.header().streams.len(), 1);
        let _ = std::fs::remove_file(path);
    }
}
