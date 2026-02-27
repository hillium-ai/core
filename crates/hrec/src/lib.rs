pub mod header;
pub mod writer;
pub mod reader;
pub mod quality;

pub use header::{Header, StreamInfo, StreamType};
pub use writer::HrecWriter;
pub use reader::HrecReader;
pub use quality::{JitterScorer, CoverageMetric};
