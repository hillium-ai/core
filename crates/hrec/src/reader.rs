use std::fs::File;
use std::path::Path;

use mmap_rs::Mmap;

use crate::header::{BodyPoseSample, Header, StreamInfo};

/// HrecReader handles reading .hrec files with memory-mapped seeks
pub struct HrecReader {
    header: Header,
    mmap: Option<Mmap>,
}

impl HrecReader {
    /// Open an .hrec file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();

        // Memory-map the file for fast random access
    }

    /// Read the header from the mmap
    fn read_header(mmap: &[u8], file_len: u64) -> std::io::Result<Header> {
        // Read header size from the end of the file
        if file_len < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small to contain header",
            ));
        }

        let header_size_start = (file_len - 8) as usize;
        let header_size_bytes = &mmap[header_size_start..header_size_start + 8];
        let header_size = u64::from_le_bytes(header_size_bytes.try_into().unwrap());

        // Read header data
        let header_start = (file_len - 8 - header_size) as usize;
        let header_bytes = &mmap[header_start..header_start + header_size as usize];

        bincode::deserialize(header_bytes).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }

    /// Seek to a specific timestamp and return samples
    pub fn seek_to_timestamp(&self, _timestamp_us: u64) -> std::io::Result<Vec<BodyPoseSample>> {
        // TODO: Implement seek-to-timestamp with <10ms latency
(Vec::new())
    }

    /// Get the header
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Get stream info by name
    pub fn get_stream_info(&self, name: &str) -> Option<&StreamInfo> {
        self.header.streams.iter().find(|s| s.name == name)
    }
}

impl std::fmt::Debug for HrecReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HrecReader")
            .field("header", &self.header)
            .finish()
    }
}
