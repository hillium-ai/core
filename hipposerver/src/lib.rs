// Main library file for HippoServer
// Provides the core shared memory structure and state management

pub mod shm {
    pub use layout::*;
    pub use state::*;
}

pub use shm::{HippoState, IntentState, RobotTelemetry, SeqLock, StateManager};
