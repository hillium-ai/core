use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use mmap_rs::Mmap;

use crate::header::{BodyPoseSample, Header, StreamInfo};

/// HrecReader handles reading .hrec files with memory-mapped seeks
pub struct HrecReader {
    header: Header,
    mmap: Mmap,
}

impl HrecReader {
    /// Open an .hrec file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();

        // Memory-map the file for fast random access
        let mmap = unsafe { Mmap::map(&file)? };

        // Read and parse header
        let header = Self::read_header(&mmap)?;

        Ok(HrecReader { header, mmap })
    }

    /// Read the header from the mmap
    fn read_header(mmap: &[u8]) -> std::io::Result<Header> {
        // TODO: Parse header from mmap
        Ok(Header::new(String::new(), 0, Vec::new()))
    }

    /// Seek to a specific timestamp and return samples
    pub fn seek_to_timestamp(&self, _timestamp_us: u64) -> std::io::Result<Vec<BodyPoseSample>> {
        // TODO: Implement seek-to-timestamp with <10ms latency
        Ok(Vec::new())
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
