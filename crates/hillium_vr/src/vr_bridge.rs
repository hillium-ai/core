//! Main VR Bridge implementation for HilliumOS Project Mirror

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::zenoh_bridge::ZenohPublisher;
use crate::openxr_bridge::OpenXrBridge;
use crate::haptic_bridge::HapticBridge;
use crate::hrec_writer::HrecWriter;

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

/// Main VR Bridge struct
#[pyclass]
pub struct VrBridge {
    zenoh_publisher: Arc<Mutex<ZenohPublisher>>,
    openxr_bridge: Arc<Mutex<OpenXrBridge>>,
    haptic_bridge: Arc<Mutex<HapticBridge>>,
    hrec_writer: Arc<Mutex<HrecWriter>>,
    is_streaming: bool,
}

#[pymethods]
impl VrBridge {
    #[new]
    pub fn new() -> Self {
        let zenoh_publisher = Arc::new(Mutex::new(ZenohPublisher::new().unwrap()));
        let openxr_bridge = Arc::new(Mutex::new(OpenXrBridge::new()));
        let haptic_bridge = Arc::new(Mutex::new(HapticBridge::new()));
        let hrec_writer = Arc::new(Mutex::new(HrecWriter::new()));
        
        Self {
            zenoh_publisher,
            openxr_bridge,
            haptic_bridge,
            hrec_writer,
            is_streaming: false,
        }
    }

    /// Start VR data streaming
    pub fn start_streaming(&mut self) -> PyResult<()> {
        // Connect to VR hardware
        self.openxr_bridge.blocking_lock().connect_headset()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.haptic_bridge.blocking_lock().connect_glove()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.is_streaming = true;
        Ok(())
    }

    /// Stop VR data streaming
    pub fn stop_streaming(&mut self) -> PyResult<()> {
        self.is_streaming = false;
        Ok(())
    }

    /// Get current pose data
    pub fn get_pose(&self) -> PyResult<VrPose> {
        if !self.is_streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not started"));
        }
        
        let pose = self.openxr_bridge.blocking_lock().capture_pose()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(pose)
    }

    /// Get haptic feedback data
    pub fn get_haptic(&self) -> PyResult<HapticFeedback> {
        if !self.is_streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not started"));
        }
        
        // In a real implementation, this would get actual haptic data
        Ok(HapticFeedback {
            timestamp_ns: 0,
            force: 0.0,
            location: "unknown".to_string(),
        })
    }

    /// Get gaze tracking data
    pub fn get_gaze(&self) -> PyResult<GazeData> {
        if !self.is_streaming {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Streaming not started"));
        }
        
        // In a real implementation, this would get actual gaze data
        Ok(GazeData {
            timestamp_ns: 0,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 0.0],
        })
    }
}
