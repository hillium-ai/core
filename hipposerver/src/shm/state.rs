// State management for HippoServer
// This module provides additional state-related functionality
// that doesn't fit in the core layout module

pub mod layout;

/// State initialization and management
pub struct StateManager {
    state: *mut layout::HippoState,
}

impl StateManager {
    pub fn new(state_ptr: *mut layout::HippoState) -> Self {
        Self { state: state_ptr }
    }

    pub fn initialize(&self) -> Result<(), &'static str> {
        unsafe {
            let state = &mut *self.state;
            
            // Initialize with default values
            state.magic.store(layout::HILLIUM_MAGIC, std::sync::atomic::Ordering::Relaxed);
            state.current_intent.store(layout::IntentState::Idle as u8, std::sync::atomic::Ordering::Relaxed);
            state.safety_lock.store(0, std::sync::atomic::Ordering::Relaxed);
            state.boot_timestamp_ns.store(0, std::sync::atomic::Ordering::Relaxed);
            
            // Initialize all pointers to null
            state.conversation_buffer_ptr = std::ptr::null_mut();
            state.conversation_buffer_size = 0;
            state.audio_buffer_ptr = std::ptr::null_mut();
            state.audio_buffer_size = 0;
            state.sled_db_path = std::ptr::null_mut();
            state.sled_db_path_len = 0;
            state.fast_weights_ptr = std::ptr::null_mut();
            state.fast_weights_size = 0;
            state.associative_update_count.store(0, std::sync::atomic::Ordering::Relaxed);
            state.qdrant_collection_ptr = std::ptr::null_mut();
            state.qdrant_collection_len = 0;
            state.last_consolidation_ns.store(0, std::sync::atomic::Ordering::Relaxed);
            
            // Initialize causal clock
            for i in 0..8 {
                state.causal_clock[i] = 0;
            }
            
            // Initialize telemetry
            state.telemetry = layout::RobotTelemetry::default();
            
            Ok(())
        }
    }

    pub fn validate(&self) -> bool {
        unsafe {
            let state = &*self.state;
            state.magic.load(std::sync::atomic::Ordering::Relaxed) == layout::HILLIUM_MAGIC
        }
    }

    pub fn set_boot_timestamp(&self, timestamp: u64) {
        unsafe {
            let state = &*self.state;
            state.boot_timestamp_ns.store(timestamp, std::sync::atomic::Ordering::Release);
        }
    }

    pub fn get_boot_timestamp(&self) -> u64 {
        unsafe {
            let state = &*self.state;
            state.boot_timestamp_ns.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    pub fn increment_associative_updates(&self) {
        unsafe {
            let state = &*self.state;
            let current = state.associative_update_count.load(std::sync::atomic::Ordering::Relaxed);
            state.associative_update_count.store(current + 1, std::sync::atomic::Ordering::Release);
        }
    }

    pub fn increment_consolidation_timestamp(&self, timestamp: u64) {
        unsafe {
            let state = &*self.state;
            state.last_consolidation_ns.store(timestamp, std::sync::atomic::Ordering::Release);
        }
    }

    pub fn get_consolidation_timestamp(&self) -> u64 {
        unsafe {
            let state = &*self.state;
            state.last_consolidation_ns.load(std::sync::atomic::Ordering::Acquire)
        }
    }
}

/// Unit tests for state management
#[cfg(test)]
mod tests {
    use super::*;
    use layout::*;

    #[test]
    fn test_state_manager() {
        // Create a HippoState instance
        let mut state = layout::HippoState::new();
        let state_ptr = &mut state as *mut layout::HippoState;
        
        // Create state manager
        let manager = StateManager::new(state_ptr);
        
        // Test initialization
        assert!(manager.initialize().is_ok());
        assert!(manager.validate());
        
        // Test boot timestamp
        manager.set_boot_timestamp(1234567890);
        assert_eq!(manager.get_boot_timestamp(), 1234567890);
        
        // Test associative updates
        manager.increment_associative_updates();
        manager.increment_associative_updates();
        unsafe {
            let state = &*state_ptr;
            assert_eq!(state.associative_update_count.load(std::sync::atomic::Ordering::Relaxed), 2);
        }
        
        // Test consolidation timestamp
        manager.increment_consolidation_timestamp(9876543210);
        assert_eq!(manager.get_consolidation_timestamp(), 9876543210);
    }
}
