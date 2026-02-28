use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use lz4_flex::compress;

use crate::header::{BodyPoseSample, Header, StreamInfo};

/// HrecWriter handles writing .hrec files with LZ4 compression
pub struct HrecWriter {
    header: Header,
    file: BufWriter<File>,
    stream_writers: Vec<StreamWriter>,
}

struct StreamWriter {
    writer: BufWriter<File>,
    sample_count: u64,
}

impl HrecWriter {
    /// Create a new HrecWriter for the given file path
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);

        Ok(HrecWriter {
            header: Header::new(String::new(), 0, Vec::new()),
            file: buf_writer,
            stream_writers: Vec::new(),
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
    pub fn finalize(mut self) -> std::io::Result<()> {
        // Write header at the END (after all data)
        let header_data = bincode::serialize(&self.header)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        self.file.write_all(&header_data.len().to_le_bytes())?;
        self.file.write_all(&header_data)?;
        
        self.file.flush()?;
        Ok(())
    }
}
