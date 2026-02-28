use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use lz4_flex::decompress;

use crate::header::{BodyPoseSample, Header};

/// HrecReader handles reading .hrec files with memory-mapped I/O
pub struct HrecReader {
    file: BufReader<File>,
    header: Header,
}

impl HrecReader {
    /// Open an .hrec file and parse its header
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut buf_reader = BufReader::new(file);

        // Read header from end of file
        let header = Self::read_header(&mut buf_reader)?;

        Ok(HrecReader {
            file: buf_reader,
            header,
        })
    }

    /// Read header from the end of the file
    fn read_header(reader: &mut BufReader<File>) -> std::io::Result<Header> {
        // Get file size
        let file_len = reader.seek(SeekFrom::End(0))?;
        if file_len < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small to contain header",
            ));
        }

        // Read header size (8 bytes at end)
        reader.seek(SeekFrom::End(-8))?;
        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let header_size = u64::from_le_bytes(size_bytes) as usize;

        // Read header data
        reader.seek(SeekFrom::End(-(header_size as i64 + 8)))?;
        let mut header_data = vec![0u8; header_size];
        reader.read_exact(&mut header_data)?;

        let header = bincode::deserialize(&header_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(header)
    }

    /// Seek to the nearest sample to the given timestamp
    pub fn seek(&mut self, _timestamp_us: u64) -> std::io::Result<()> {
        // TODO: Implement timestamp-based seeking using index
        Ok(())
    }

    /// Read the next body pose sample
    pub fn read_body_pose(&mut self) -> std::io::Result<Option<BodyPoseSample>> {
        // Read compressed data size
        let mut size_bytes = [0u8; 4];
        match self.file.read_exact(&mut size_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let data_size = u32::from_le_bytes(size_bytes) as usize;
        let mut compressed = vec![0u8; data_size];
        self.file.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = decompress(&compressed, data_size)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Deserialize
        let sample = bincode::deserialize(&decompressed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
Ok(Some(sample))
    }

    /// Get the header
    pub fn header(&self) -> &Header {
        &self.header
    }
}
