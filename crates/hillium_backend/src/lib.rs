// hillium_backend/src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;

// Mock hipposerver for now since we don't have the crate dependency set up perfectly in this restoration
// Real implementation using workspace crates
use hipposerver::shm::ShmHandle;
use working_memory::WorkingMemory;
use associative_core::AssociativeCore;

#[pyclass]
#[derive(Debug)]
pub struct HippoLink {
    shm_handle: Arc<ShmHandle>,
    working_memory: Arc<WorkingMemory>,
    associative_core: Arc<AssociativeCore>,
}

#[pymethods]
impl HippoLink {
    #[new]
    fn new(shm_name: &str, db_path:Option<String>) -> PyResult<Self> {
        // Connect to Shared Memory (Try Open, then Create)
        let shm_handle = match ShmHandle::open(shm_name) {
            Ok(h) => Arc::new(h),
            Err(_) => {
                // Fallback to creation for testing/standalone
               Arc::new(ShmHandle::create(shm_name).map_err(|e| {
                   PyRuntimeError::new_err(format!("Failed to create/open SHM handle: {}", e))
               })?)
            }
        };
        
        // Initialize Working Memory (Sled)
        let path = db_path.unwrap_or_else(|| "/tmp/hillium_sled".to_string());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let wm = rt.block_on(async {
            WorkingMemory::new(&path).await
        }).map_err(|e| PyRuntimeError::new_err(format!("Failed to open Working Memory: {}", e)))?;

        // Initialize Associative Core (Dummy params for now as per WP-003 defaults)
        let ac = AssociativeCore::new(128, 64, 0.9, 0.01, 0.01);

        Ok(HippoLink {
            shm_handle,
            working_memory: Arc::new(wm),
            associative_core: Arc::new(ac),
        })
    }

    // Level 1 (Sensory) - Conversation & Intent
    fn get_intent(&self) -> PyResult<String> {
        let state = self.shm_handle.state();
        // Just return idle for now unless we implement full enum mapping
        Ok(format!("{:?}", state.get_intent()))
    }

    fn set_intent(&self, intent_code: u8) -> PyResult<()> {
        let state = self.shm_handle.state();
        // Warning: Direct unsafe cast for MVP speed, needs validation in prod
        unsafe {
            let intent_enum: hipposerver::layout::IntentState = std::mem::transmute(intent_code);
            state.set_intent(intent_enum);
        }
        Ok(())
    }

    // Level 2 (Working Memory) - Async via Tokio runtime block_on
    fn store_note(&self, note_id: &str, content: &str) -> PyResult<()> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            self.working_memory.write_note(&working_memory::StructuredNote {
                id: note_id.to_string(),
                content: content.to_string(),
                timestamp: 0, 
                domain: "default".to_string(),
                tags: vec![],
            }).await
        }).map_err(|e| PyRuntimeError::new_err(format!("WM Write Error: {}", e)))
    }

    fn get_note(&self, note_id: &str) -> PyResult<Option<String>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let note = rt.block_on(async {
            self.working_memory.read_note(note_id).await
        }).map_err(|e| PyRuntimeError::new_err(format!("WM Read Error: {}", e)))?;
        
        Ok(note.map(|n| n.content))
    }

    // Level 2.5 (Associative)
    fn get_associative_weights(&self) -> PyResult<Vec<f32>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            self.associative_core.export_weights().await
        }).map_err(|e| PyRuntimeError::new_err(format!("AC Export Error: {}", e)))
    }

    // Utility methods
    fn get_telemetry(&self) -> PyResult<String> {
        let state = self.shm_handle.state();
        Ok(format!(
            "Telemetry: cpu={:.1}%, mem={:.1}%, temp={:.1}C, batt={:.1}%",
            state.telemetry.cpu_usage,
            state.telemetry.memory_usage,
            state.telemetry.temperature_c,
            state.telemetry.battery_percentage
        ))
    }

    fn is_estopped(&self) -> PyResult<bool> {
        Ok(self.shm_handle.state().is_safety_locked())
    }
}

#[pymodule]
fn hillium_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HippoLink>()?;
    Ok(())
}
