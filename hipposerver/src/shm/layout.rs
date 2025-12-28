use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::ffi::c_char;
use std::time::Duration;

/// Magic number for HippoState validation
pub const HILLIUM_MAGIC: u32 = 0x48494C4C;

/// Intent state for the robot
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentState {
    Idle = 0,
    Listening = 1,
    Thinking = 2,
    Speaking = 3,
    Acting = 4,
    Emergency = 255,
}

/// Robot telemetry data structure
#[repr(C)]
#[derive(Debug, Default)]
pub struct RobotTelemetry {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub temperature_c: f32,
    pub battery_percentage: f32,
    pub timestamp_ns: u64,
}

/// Sequence lock for lock-free reading
#[repr(C)]
#[derive(Debug, Default)]
pub struct SeqLock {
    seq: AtomicU64,
}

impl SeqLock {
    pub fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
        }
    }

    pub fn begin_read(&self) -> u64 {
        self.seq.load(Ordering::Acquire)
    }

    pub fn end_read(&self, seq: u64) -> bool {
        let current = self.seq.load(Ordering::Relaxed);
        current == seq
    }

    pub fn begin_write(&self) -> u64 {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq + 1, Ordering::Release);
        seq
    }

    pub fn end_write(&self, _seq: u64) {
        // Write is complete, no need to verify
    }
}

/// Main HippoState structure for shared memory
/// This defines the memory contract for v8.0
#[repr(C, align(64))]
#[derive(Debug)]
pub struct HippoState {
    /// Magic number for validation
    pub magic: AtomicU32,
    
    /// Current intent state
    pub current_intent: AtomicU8,
    
    /// Safety lock flag
    pub safety_lock: AtomicBool,
    
    /// Boot timestamp in nanoseconds
    pub boot_timestamp_ns: AtomicU64,
    
    /// Sequence lock for non-atomic fields
    pub seq_lock: SeqLock,
    
    /// Level 1: Sensory Buffer
    /// Conversation ring buffer pointer
    pub conversation_buffer_ptr: *mut u8,
    /// Conversation buffer size
    pub conversation_buffer_size: usize,
    /// Audio ring buffer pointer
    pub audio_buffer_ptr: *mut u8,
    /// Audio buffer size
    pub audio_buffer_size: usize,
    
    /// Level 2: Working Memory
    /// Sled database path
    pub sled_db_path: *mut c_char,
    /// Sled database path length
    pub sled_db_path_len: usize,
    
    /// Level 2.5: Associative Core
    /// Fast weights pointer
pub fast_weights_ptr: *mut f32,
    /// Fast weights size
    pub fast_weights_size: usize,
    /// Associative update count
    pub associative_update_count: AtomicU64,
    
    /// Level 3: Episodic Store Metadata
    /// Qdrant collection name pointer
    pub qdrant_collection: *mut c_char,
    /// Qdrant collection name length
    pub qdrant_collection_len: usize,
    /// Last consolidation timestamp in nanoseconds
    pub last_consolidation_ns: AtomicU64,
    
    /// Causal clock for distributed consistency
    pub causal_clock: [u64; 8],
    
    /// Robot telemetry data
    pub telemetry: RobotTelemetry,
    
    /// Padding to ensure 64-byte alignment
    pub _padding: [u8; 64],
}

impl HippoState {
    pub fn new() -> Self {
        Self {
            magic: AtomicU32::new(HILLIUM_MAGIC),
            current_intent: AtomicU8::new(IntentState::Idle as u8),
            safety_lock: AtomicBool::new(false),
            boot_timestamp_ns: AtomicU64::new(0),
            seq_lock: SeqLock::new(),
            conversation_buffer_ptr: std::ptr::null_mut(),
            conversation_buffer_size: 0,
            audio_buffer_ptr: std::ptr::null_mut(),
            audio_buffer_size: 0,
            sled_db_path: std::ptr::null_mut(),
            sled_db_path_len: 0,
            fast_weights_ptr: std::ptr::null_mut(),
            fast_weights_size: 0,
            associative_update_count: AtomicU64::new(0),
            qdrant_collection: std::ptr::null_mut(),
            qdrant_collection_len: 0,
            last_consolidation_ns: AtomicU64::new(0),
            causal_clock: [0; 8],
            telemetry: RobotTelemetry::default(),
            _padding: [0; 64],
        }
    }

    pub fn validate(&self) -> bool {
        self.magic.load(Ordering::Relaxed) == HILLIUM_MAGIC
    }

    pub fn set_intent(&self, intent: IntentState) {
        self.current_intent.store(intent as u8, Ordering::Release);
    }

    pub fn get_intent(&self) -> IntentState {
        let val = self.current_intent.load(Ordering::Acquire);
        unsafe { std::mem::transmute(val) }
    }

    pub fn lock_safety(&self) {
        self.safety_lock.store(true, Ordering::Release);
    }

    pub fn unlock_safety(&self) {
        self.safety_lock.store(false, Ordering::Release);
    }

    pub fn is_safety_locked(&self) -> bool {
        self.safety_lock.load(Ordering::Acquire)
    }
}

/// Unit tests for HippoState layout
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_layout() {
        // Verify size is within 4KB
        let size = std::mem::size_of::<HippoState>();
        assert!(size <= 4096, "HippoState size {} exceeds 4KB limit", size);

        // Verify alignment is 64 bytes
        let align = std::mem::align_of::<HippoState>();
        assert_eq!(align, 64, "HippoState alignment should be 64 bytes, got {}", align);

        // Verify magic number
        let state = HippoState::new();
        assert!(state.validate(), "Magic number validation failed");
        assert_eq!(state.magic.load(Ordering::Relaxed), HILLIUM_MAGIC);

        // Verify atomic fields
        assert_eq!(state.current_intent.load(Ordering::Relaxed), IntentState::Idle as u8);
        assert_eq!(state.safety_lock.load(Ordering::Relaxed), false);
        assert_eq!(state.boot_timestamp_ns.load(Ordering::Relaxed), 0);

        // Test intent state transitions
        state.set_intent(IntentState::Listening);
        assert_eq!(state.get_intent(), IntentState::Listening);

        state.set_intent(IntentState::Thinking);
        assert_eq!(state.get_intent(), IntentState::Thinking);

        // Test safety lock
        assert!(!state.is_safety_locked());
        state.lock_safety();
        assert!(state.is_safety_locked());
        state.unlock_safety();
        assert!(!state.is_safety_locked());

        // Test sequence lock
        let seq = state.seq_lock.begin_read();
        assert!(state.seq_lock.end_read(seq));
        let write_seq = state.seq_lock.begin_write();
        state.seq_lock.end_write(write_seq);

        println!("HippoState size: {} bytes", size);
        println!("HippoState alignment: {} bytes", align);
    }

    #[test]
    fn test_telemetry_layout() {
        let size = std::mem::size_of::<RobotTelemetry>();
        assert!(size <= 64, "RobotTelemetry size {} exceeds 64 bytes", size);
        
        let align = std::mem::align_of::<RobotTelemetry>();
        assert_eq!(align, 8, "RobotTelemetry alignment should be 8 bytes, got {}", align);
    }

    #[test]
    fn test_seq_lock() {
        let lock = SeqLock::new();
        
        // Test read sequence
        let seq1 = lock.begin_read();
        assert!(lock.end_read(seq1));
        
        // Test write sequence
        let write_seq = lock.begin_write();
        lock.end_write(write_seq);
        
        // Verify sequence increments
        let seq2 = lock.begin_read();
        assert_eq!(seq2, write_seq + 1);
    }
}
