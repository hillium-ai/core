use hipposerver::shm::ShmHandle;
pub mod visual;
use crate::values::ValueScorecard;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

/// Safety check result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyResult {
    Allowed,
    Blocked { reason: &'static str },
}

/// Aegis safety validator (MVP version)
pub struct AegisCore {
    hippo: Arc<ShmHandle>,
    /// Value scorecards from latest evaluation
    pub value_scores: Vec<ValueScorecard>,
}

impl AegisCore {
    /// Create new Aegis instance
    pub fn new(hippo: Arc<ShmHandle>) -> Self {
        Self { 
            hippo,
            value_scores: Vec::new()
        }
    }

    /// Validate command (MVP: just check E-Stop)
    pub fn validate_command(&self, _command: &str) -> SafetyResult {
        // Check E-Stop
        if self.is_estopped() {
            return SafetyResult::Blocked {
                reason: "Emergency stop active"
            };
        }

        // 🆕 v8.0: Check Layer 7 emergency brake
        if self.is_layer7_brake_active() {
            return SafetyResult::Blocked {
                reason: "Layer 7 cognitive safety brake active"
            };
        }

        // MVP: Allow everything if both checks pass
        SafetyResult::Allowed
    }

    /// Trigger emergency stop
    pub fn trigger_estop(&self) {
        log::warn!("EMERGENCY STOP TRIGGERED");
        self.hippo.state().safety_lock.store(true, Ordering::Release);
    }

    /// Clear emergency stop
    pub fn clear_estop(&self) {
        log::info!("Emergency stop cleared");
        self.hippo.state().safety_lock.store(false, Ordering::Release);
    }

    /// Check if E-Stop is active
    pub fn is_estopped(&self) -> bool {
        self.hippo.state().safety_lock.load(Ordering::Acquire)
    }

    /// Check if Layer 7 emergency brake is active
    pub fn is_layer7_brake_active(&self) -> bool {
        self.hippo.state().layer7_emergency_brake.load(Ordering::Acquire)
    }

    /// Clear both manual E-Stop and Layer 7 brake
    pub fn clear_all_safety_locks(&self) {
        // Clear manual E-Stop
        self.hippo.state().safety_lock.store(false, Ordering::Release);

        // Clear Layer 7 brake
        self.hippo.state().layer7_emergency_brake.store(false, Ordering::Release);

        log::info!("All safety locks cleared");
    }

    /// Stub: Returns constant scores for MVP
    pub fn evaluate_values(&mut self) -> Vec<ValueScorecard> {
        let scores = vec![
            ValueScorecard::new("safety", 1.0, "Placeholder: always safe"),
            ValueScorecard::new("efficiency", 0.5, "Placeholder: neutral"),
        ];
        self.value_scores = scores.clone();
        scores
    }

    /// Start heartbeat thread (10Hz = 100ms interval)
    pub fn start_heartbeat_thread(&self) {
        let hippo_clone = self.hippo.clone();
        thread::spawn(move || {
            log::info!("Aegis heartbeat thread started (10Hz)");
            loop {
                // Update heartbeat timestamp
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
                hippo_clone.state().hb_aegis_ns.store(now, Ordering::Release);
                
                // Sleep for 100ms (10Hz)
                thread::sleep(Duration::from_millis(100));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use hipposerver::shm::ShmHandle;

    #[test]
    fn test_estop() {
        let shm = Arc::new(ShmHandle::create("/hillium_test_aegis").unwrap());
        let aegis = AegisCore::new(shm.clone());

        // Initially not stopped
        assert!(!aegis.is_estopped());
        assert!(!aegis.is_layer7_brake_active());
        assert_eq!(aegis.validate_command("test"), SafetyResult::Allowed);

        // Trigger E-Stop
        aegis.trigger_estop();
        assert!(aegis.is_estopped());
        assert_eq!(
            aegis.validate_command("test"),
            SafetyResult::Blocked { reason: "Emergency stop active" }
        );

        // Clear E-Stop
        aegis.clear_estop();
        assert!(!aegis.is_estopped());

        // Test Layer 7 emergency brake functionality
        // (Note: In production, this would be set by Layer 7 cognitive safety system)
        // For testing purposes, we verify the method exists and works correctly
        assert!(!aegis.is_layer7_brake_active());
        
        // Verify the clear_all_safety_locks method works
        aegis.clear_all_safety_locks();
        assert!(!aegis.is_estopped());
        assert!(!aegis.is_layer7_brake_active());
        assert_eq!(aegis.validate_command("test"), SafetyResult::Allowed);

        // Test Layer 7 emergency brake blocking behavior
        // Manually set the Layer 7 brake (simulating cognitive safety system)
        shm.state().layer7_emergency_brake.store(true, Ordering::Release);
        assert!(aegis.is_layer7_brake_active());
        assert_eq!(
            aegis.validate_command("test"),
            SafetyResult::Blocked { reason: "Layer 7 cognitive safety brake active" }
        );

        // Clear the Layer 7 brake
        shm.state().layer7_emergency_brake.store(false, Ordering::Release);
        assert!(!aegis.is_layer7_brake_active());
        assert_eq!(aegis.validate_command("test"), SafetyResult::Allowed);

        // Cleanup
        drop(shm);
        hipposerver::shm::unlink("/hillium_test_aegis").unwrap();
    }
}
