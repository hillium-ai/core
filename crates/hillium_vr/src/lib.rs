//! Hillium VR Bridge - Real-time VR data streaming for Project Mirror

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// Declare modules
pub mod shared_types;
pub mod openxr_bridge;
pub mod haptic_bridge;
pub mod webrtc_bridge;
pub mod zenoh_bridge;
pub mod hrec_writer;
pub mod mock_data;
pub mod vr_bridge;

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
            Ok(pose) => {
                // Publish pose data via Zenoh
                if let Err(e) = self.zenoh_publisher.publish_pose(&pose) {
                    eprintln!("Failed to publish pose: {}", e);
                }
                Ok(pose)
            },
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to capture pose: {}", e))),
        }
    }

    /// Get haptic feedback data
    pub fn get_haptic(&self) -> PyResult<HapticFeedback> {
        if !self.streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not active".to_string()));
        }

        // For now, return mock haptic data
        let haptic = HapticFeedback {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            force: 0.5,
            location: "hand".to_string(),
        };
        
        // Publish haptic data via Zenoh
        if let Err(e) = self.zenoh_publisher.publish_haptic(&haptic) {
            eprintln!("Failed to publish haptic: {}", e);
        }
        
        Ok(haptic)
    }

    /// Get gaze tracking data
    pub fn get_gaze(&self) -> PyResult<GazeData> {
        if !self.streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not active".to_string()));
        }

        // For now, return mock gaze data
        let gaze = GazeData {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
        };
        
        // Publish gaze data via Zenoh
        if let Err(e) = self.zenoh_publisher.publish_gaze(&gaze) {
            eprintln!("Failed to publish gaze: {}", e);
        }
        
        Ok(gaze)
    }
}
