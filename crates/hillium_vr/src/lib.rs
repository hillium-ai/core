//! Hillium VR Bridge - Real-time VR data streaming for Project Mirror

use pyo3::prelude::*;
use pyo3::pyclass;

// Declare modules
pub mod haptic_bridge;
pub mod hrec_writer;
pub mod mock_data;
pub mod openxr_bridge;
pub mod shared_types;
pub mod vr_bridge;
pub mod webrtc_bridge;
pub mod zenoh_bridge;

pub use shared_types::*;

/// Main VR Bridge struct
#[pyclass]
pub struct VrBridge {
    /// OpenXR bridge for headset connectivity
    #[pyo3(get)]
    pub openxr_bridge: openxr_bridge::OpenXrBridge,
    /// Zenoh publisher for data streaming
    #[pyo3(get)]
    pub zenoh_publisher: zenoh_bridge::ZenohPublisher,
    /// Haptic bridge for glove feedback
    #[pyo3(get)]
    pub haptic_bridge: haptic_bridge::HapticBridge,
    /// WebRTC server for NAT traversal
    #[pyo3(get)]
    pub webrtc_server: webrtc_bridge::WebRtcServer,
    /// Flag indicating if streaming is active
    #[pyo3(get)]
    pub streaming: bool,
}

impl Default for VrBridge {
    fn default() -> Self {
        Self::new()
    }
}

