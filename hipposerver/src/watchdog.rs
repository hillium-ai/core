//! Watchdog monitors component heartbeats and triggers protective actions
//! if any component freezes or crashes.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use log::{error, info, warn};

use crate::shm::{ShmHandle, HippoState};

/// Watchdog monitors component heartbeats
pub struct Watchdog {
    hippo: Arc<ShmHandle>,
    timeout_ms: u64,
}

impl Watchdog {
    /// Create new watchdog instance
    pub fn new(hippo: Arc<ShmHandle>) -> Self {
        Self {
            hippo,
            timeout_ms: 500, // 500ms timeout threshold
        }
    }

    /// Set timeout threshold in milliseconds
    pub fn set_timeout(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms;
    }

    /// Check all heartbeats and trigger actions if timeout occurs
    pub fn check(&self) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let state = self.hippo.state();
        let hb_aegis_ns = state.hb_aegis_ns.load(Ordering::Acquire);
        let age_ms = (now_ns - hb_aegis_ns) / 1_000_000;

        // Check Aegis heartbeat
        if age_ms > self.timeout_ms {
            error!("Aegis heartbeat timeout! Last update: {}ms ago (threshold: {}ms)", 
                   age_ms, self.timeout_ms);
            
            // Trigger emergency stop
            self.trigger_emergency_stop();
        } else {
            info!("Aegis heartbeat OK: {}ms ago", age_ms);
        }
    }

    /// Main watchdog monitoring loop
    pub fn monitor_loop(&self) {
        info!("Watchdog monitoring loop started (check interval: 100ms)");
        
        loop {
            self.check();
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Trigger emergency stop when component fails
    fn trigger_emergency_stop(&self) {
        warn!("Triggering emergency stop due to heartbeat timeout");
        let state = self.hippo.state();
        state.safety_lock.store(true, Ordering::Release);
        
        // Also trigger Layer 7 emergency brake
        state.layer7_emergency_brake.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_watchdog_timeout() {
        let shm = Arc::new(ShmHandle::create("/hillium_test_watchdog").unwrap());
        let watchdog = Watchdog::new(shm.clone());
        
        // Set very short timeout for testing
        let mut watchdog = watchdog;
        watchdog.set_timeout(100); // 100ms timeout
        
        // Don't update heartbeat - should timeout
        thread::sleep(Duration::from_millis(200));
        
        // Check should detect timeout
        watchdog.check();
        
        // Verify E-Stop was triggered
        assert!(shm.state().safety_lock.load(Ordering::Acquire));
        assert!(shm.state().layer7_emergency_brake.load(Ordering::Acquire));
        
        // Cleanup
        drop(shm);
        crate::shm::unlink("/hillium_test_watchdog").unwrap();
    }

    #[test]
    fn test_watchdog_no_timeout() {
        let shm = Arc::new(ShmHandle::create("/hillium_test_watchdog2").unwrap());
        let watchdog = Watchdog::new(shm.clone());
        
        // Set short timeout
        let mut watchdog = watchdog;
        watchdog.set_timeout(100);
        
        // Update heartbeat
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        shm.state().hb_aegis_ns.store(now, Ordering::Release);
        
        // Should not timeout
        thread::sleep(Duration::from_millis(50));
        watchdog.check();
        
        // Verify E-Stop was NOT triggered
        assert!(!shm.state().safety_lock.load(Ordering::Acquire));
        
        // Cleanup
        drop(shm);
        crate::shm::unlink("/hillium_test_watchdog2").unwrap();
    }
}
