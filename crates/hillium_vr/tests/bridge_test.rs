//! Tests for VR bridge functionality

use hillium_vr::VrBridge;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vr_bridge_creation() {
        let bridge = VrBridge::new();
        assert!(!bridge.streaming);
    }

    #[test]
    fn test_vr_bridge_fields_access() {
        let bridge = VrBridge::new();
        // Test that we can access the fields directly (this was failing before)
        let _ = &bridge.streaming;
        let _ = &bridge.openxr_bridge;
        let _ = &bridge.zenoh_publisher;
        let _ = &bridge.haptic_bridge;
        let _ = &bridge.webrtc_server;
    }
}
