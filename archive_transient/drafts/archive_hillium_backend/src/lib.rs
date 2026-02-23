use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use shared_memory::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use hipposerver::shm::{HippoState, IntentState, CONVERSATION_BUFFER_SIZE};

/// Shared memory handle for Python
struct ShmHandle {
    shm: Shmem,
    ptr: *mut HippoState,
}

unsafe impl Send for ShmHandle {}
unsafe impl Sync for ShmHandle {}

impl ShmHandle {
    fn open(name: &str) -> PyResult<Self> {
        let shm = ShmemConf::new()
            .os_id(name)
            .open()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to open shared memory: {:?}", e)))?;
        
        let ptr = shm.as_ptr() as *mut HippoState;
        Ok(Self { shm, ptr })
    }

    fn state(&self) -> &HippoState {
        unsafe { &*self.ptr }
    }
}

/// Python wrapper for HippoServer connection
#[pyclass]
pub struct HippoLink {
    shm: Arc<ShmHandle>,
}

#[pymethods]
impl HippoLink {
    /// Connect to HippoServer shared memory
    #[new]
    fn new(shm_name: Option<String>) -> PyResult<Self> {
        let name = shm_name.unwrap_or_else(|| "/hillium_v1".to_string());
        let shm = ShmHandle::open(&name)?;
        Ok(Self {
            shm: Arc::new(shm),
        })
    }

    /// Get singleton instance (convenience for loqus_core)
    #[staticmethod]
    fn get_instance() -> PyResult<Self> {
        Self::new(None)
    }

    /// Get current intent state
    fn get_intent(&self) -> String {
        let intent = self.shm.state().get_intent();
        format!("{:?}", intent)
    }

    /// Set intent state
    fn set_intent(&self, intent: &str) -> PyResult<()> {
        let intent_enum = match intent {
            "Idle" => IntentState::Idle,
            "Listening" => IntentState::Listening,
            "Thinking" => IntentState::Thinking,
            "Speaking" => IntentState::Speaking,
            "Acting" => IntentState::Acting,
            "Error" => IntentState::Error,
            _ => return Err(PyRuntimeError::new_err(format!("Invalid intent: {}", intent))),
        };

        self.shm.state().set_intent(intent_enum);
        Ok(())
    }

    /// Write to conversation buffer (Level 1)
    fn write_conversation(&self, text: &str) -> PyResult<()> {
        let bytes = text.as_bytes();
        let max_len = CONVERSATION_BUFFER_SIZE;

        if bytes.len() > max_len {
            return Err(PyRuntimeError::new_err(format!(
                "Text too long: {} > {} bytes",
                bytes.len(),
                max_len
            )));
        }

        let state = self.shm.state();

        unsafe {
            // SAFETY: We have exclusive or atomic access guaranteed by the architectural contract
            let buffer_ptr = state.conversation_buffer.as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer_ptr, bytes.len());
        }

        state.conversation_length.store(bytes.len() as u64, Ordering::Release);
        Ok(())
    }

    /// Read conversation buffer (Level 1)
    fn read_conversation(&self) -> String {
        let state = self.shm.state();
        let len = state.conversation_length.load(Ordering::Acquire) as usize;

        if len == 0 {
            return String::new();
        }

        let bytes = &state.conversation_buffer[..len];
        String::from_utf8_lossy(bytes).to_string()
    }

    /// Update heartbeat
    fn heartbeat(&self, component: &str) -> PyResult<()> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let state = self.shm.state();

        match component {
            "loqus" => state.hb_loqus_ns.store(now, Ordering::Release),
            "aegis" => state.hb_aegis_ns.store(now, Ordering::Release),
            "aura" => state.hb_aura_ns.store(now, Ordering::Release),
            _ => return Err(PyRuntimeError::new_err(format!("Unknown component: {}", component))),
        }

        Ok(())
    }

    /// Check if emergency stop is active
    fn is_estopped(&self) -> bool {
        self.shm.state().is_safety_locked()
    }

    /// Get boot timestamp
    fn get_boot_time(&self) -> u64 {
        self.shm.state().boot_timestamp_ns.load(Ordering::Relaxed)
    }
}

/// Python module
#[pymodule]
fn hillium_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HippoLink>()?;
    Ok(())
}