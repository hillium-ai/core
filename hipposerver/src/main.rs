use std::time::{SystemTime, UNIX_EPOCH, Duration};
use shared_memory::*;
use tracing::{info, warn, error};
use anyhow::{Context, Result};

mod shm;
use shm::{HippoState, StateManager};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting HippoServer v8.0 Orchestrator...");

    // Shared memory configuration
    let shm_path = "/hillium_v1";
    let shm_size = 512 * 1024 * 1024; // 512MB

    // Try to open existing or create new shared memory
    let shm = match ShmemConf::new()
        .size(shm_size)
        .os_id(shm_path)
        .create() {
            Ok(s) => {
                info!("Created new shared memory mapping at {}", shm_path);
                s
            },
            Err(ShmemError::LinkExists) => {
                info!("Shared memory mapping already exists, opening {}", shm_path);
                ShmemConf::new()
                    .os_id(shm_path)
                    .open()
                    .context("Failed to open existing shared memory")?
            },
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to create shared memory: {:?}", e));
            }
        };

    let shm_ptr = shm.as_ptr() as *mut HippoState;
    let manager = StateManager::new(shm_ptr);

    // Initialize state if it's a fresh mapping
    if !manager.validate() {
        info!("Initializing HippoState magic and default values...");
        manager.initialize().map_err(|e| anyhow::anyhow!(e))?;
    }

    // Set boot timestamp
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("Time went backwards")?
        .as_nanos() as u64;
    manager.set_boot_timestamp(now);

    info!("HippoServer initialized. Entering heartbeat loop...");

    // Main heartbeat loop
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    loop {
        interval.tick().await;
        
        // Safety check
        unsafe {
            let state = &*shm_ptr;
            if state.is_safety_locked() {
                warn!("Aegis Safety Lock ACTIVE. Restricting operations.");
            }
        }
        
        // Heartbeat log for debug
        // info!("Heartbeat tick...");
    }
}