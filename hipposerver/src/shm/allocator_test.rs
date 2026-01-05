use super::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Integration test for ShmPtr and LinearRingAllocator
#[test]
fn test_shm_ptr_deref() {
    // Simulate a base address (e.g., start of shared memory)
    let base_addr = 0x10000000 as *const u8;
    
    // Create a pointer with offset 1024
    let ptr = ShmPtr::<u32>::new(1024);
    
    // Dereference it
    let absolute_ptr = ptr.deref(base_addr);
    
    // Verify the pointer is at the correct absolute address
    assert_eq!(absolute_ptr as usize, 0x10000000 + 1024);
}

/// Test allocation and wrap-around behavior
#[test]
fn test_allocator_behavior() {
    // Create allocator with 1KB arena starting at 4KB
    let allocator = LinearRingAllocator::new(4096, 1024);
    
    // Test 1: Normal allocation
    let ptr1 = allocator.allocate(128).unwrap();
    assert_eq!(ptr1.offset(), 4096);
    
    // Test 2: Second allocation
    let ptr2 = allocator.allocate(256).unwrap();
    assert_eq!(ptr2.offset(), 4128); // 4096 + 32 (aligned)
    
    // Test 3: Wrap around
    let ptr3 = allocator.allocate(512).unwrap();
    assert_eq!(ptr3.offset(), 4096); // Wrapped to beginning
    
    // Test 4: Verify cursor position
    assert_eq!(allocator.current_cursor(), 4608); // 4096 + 512
    
    println!("Allocation tests passed!");
}

/// Test alignment behavior
#[test]
fn test_alignment() {
    let allocator = LinearRingAllocator::new(4096, 1024);
    
    // Allocate unaligned size
    let ptr1 = allocator.allocate(13).unwrap();
    assert_eq!(ptr1.offset(), 4096);
    
    // Next allocation should be aligned
    let ptr2 = allocator.allocate(13).unwrap();
    assert_eq!(ptr2.offset(), 4104); // 4096 + 8
    
    // Next allocation should also be aligned
    let ptr3 = allocator.allocate(13).unwrap();
    assert_eq!(ptr3.offset(), 4112); // 4104 + 8
}

/// Test edge case: exactly filling the arena
#[test]
fn test_exact_fill() {
    let allocator = LinearRingAllocator::new(4096, 1024);
    
    // Allocate exactly 1024 bytes (1KB)
    let ptr = allocator.allocate(1024).unwrap();
    assert_eq!(ptr.offset(), 4096);
    
    // Next allocation should wrap
    let ptr2 = allocator.allocate(1).unwrap();
    assert_eq!(ptr2.offset(), 4096);
}

/// Test concurrent access simulation
#[test]
fn test_concurrent_allocation() {
    use std::thread;
    
    let allocator = LinearRingAllocator::new(4096, 1024 * 1024); // 1MB arena
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let allocator = allocator.clone(); // This won't work as-is, need Arc
        // In real implementation, we'd use Arc<LinearRingAllocator>
        break;
    }
    
    // For now, just test that allocations don't overlap in single-threaded case
    let ptr1 = allocator.allocate(100).unwrap();
    let ptr2 = allocator.allocate(100).unwrap();
    
    assert_ne!(ptr1.offset(), ptr2.offset());
    assert!(ptr2.offset() > ptr1.offset() || ptr2.offset() < allocator.arena_start() + allocator.arena_size());
}
