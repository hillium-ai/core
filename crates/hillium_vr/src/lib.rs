//! Hillium VR Bridge - Real-time VR data streaming for Project Mirror

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// VR Pose data structure
#[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub struct VrPose {
    pub timestamp_ns: u64,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
}

/// Haptic feedback data
#[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub struct HapticFeedback {
    pub timestamp_ns: u64,
    pub force: f32,
    pub location: String,
}

/// Gaze tracking data
#[pyclass]
#[derive(Serialize, Deserialize, Clone)]
pub struct GazeData {
    pub timestamp_ns: u64,
    pub position: [f32; 3],
    pub direction: [f32; 3],
}

// Declare modules
pub mod openxr_bridge;
pub mod haptic_bridge;
pub mod webrtc_bridge;
pub mod zenoh_bridge;
pub mod hrec_writer;
pub mod mock_data;
pub mod vr_bridge;

/// Main VR Bridge struct
#[pyclass]
pub struct VrBridge {
    /// OpenXR bridge for headset connectivity
    openxr_bridge: openxr_bridge::OpenXrBridge,
    /// Zenoh publisher for data streaming
    zenoh_publisher: zenoh_bridge::ZenohPublisher,
    /// Haptic bridge for glove feedback
    haptic_bridge: haptic_bridge::HapticBridge,
    /// WebRTC server for NAT traversal
    webrtc_server: webrtc_bridge::WebRtcServer,
    /// Flag indicating if streaming is active
    streaming: bool,
}

#[pymethods]
impl VrBridge {
    #[new]
    pub fn new() -> Self {
        Self {
            openxr_bridge: openxr_bridge::OpenXrBridge::new(),
            zenoh_publisher: zenoh_bridge::ZenohPublisher::new().unwrap(),
            haptic_bridge: haptic_bridge::HapticBridge::new(),
            webrtc_server: webrtc_bridge::WebRtcServer::new(),
            streaming: false,
        }
    }

    /// Start VR data streaming
    pub fn start_streaming(&mut self) -> PyResult<()> {
        // Initialize OpenXR
        if let Err(e) = self.openxr_bridge.initialize() {
            eprintln!("Failed to initialize OpenXR: {}", e);
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to initialize OpenXR: {}", e)));
        }

        // Connect to headset
        if let Err(e) = self.openxr_bridge.connect_headset() {
            eprintln!("Failed to connect to headset: {}", e);
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to connect to headset: {}", e)));
        }

        // Start WebRTC signaling
        if let Err(e) = self.webrtc_server.start_signaling() {
            eprintln!("Failed to start WebRTC signaling: {}", e);
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to start WebRTC signaling: {}", e)));
        }

        self.streaming = true;
        Ok(())
    }

    /// Stop VR data streaming
    pub fn stop_streaming(&mut self) -> PyResult<()> {
        // Disconnect from headset
        if let Err(e) = self.openxr_bridge.disconnect_headset() {
            eprintln!("Failed to disconnect from headset: {}", e);
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to disconnect from headset: {}", e)));
        }

        self.streaming = false;
        Ok(())
    }

    /// Get current pose data
    pub fn get_pose(&self) -> PyResult<VrPose> {
        if !self.streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not active".to_string()));
        }

        match self.openxr_bridge.capture_pose() {
            Ok(pose) => Ok(pose),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to capture pose: {}", e))),
        }
    }

    /// Get haptic feedback data
    pub fn get_haptic(&self) -> PyResult<HapticFeedback> {
        if !self.streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not active".to_string()));
        }

        // For now, return mock haptic data
        Ok(HapticFeedback {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            force: 0.5,
            location: "hand".to_string(),
        })
    }

    /// Get gaze tracking data
    pub fn get_gaze(&self) -> PyResult<GazeData> {
        if !self.streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not active".to_string()));
        }

        // For now, return mock gaze data
        Ok(GazeData {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
        })
    }
}
