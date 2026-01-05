use std::sync::atomic::{AtomicU32, Ordering};
use std::marker::PhantomData;

/// A pointer type that stores an offset from the start of the shared memory segment.
/// This allows for safe inter-process referencing without storing absolute pointers.
/// The offset is a u32 to ensure it fits within a 4GB shared memory segment.
#[derive(Debug, Clone, Copy)]
pub struct ShmPtr<T> {
    offset: u32,
    _marker: PhantomData<T>,
}

impl<T> ShmPtr<T> {
    /// Creates a new `ShmPtr` from an offset.
    pub fn new(offset: u32) -> Self {
        Self {
            offset,
            _marker: PhantomData,
        }
    }

    /// Resolves the pointer to an absolute address given the base address of the SHM segment.
    /// This converts the relative offset into an absolute pointer that can be used in the
    /// current process's address space.
    pub fn deref(&self, base_addr: *const u8) -> *mut T {
        unsafe { base_addr.add(self.offset as usize) as *mut T }
    }

    /// Returns the offset value stored in this pointer.
    pub fn offset(&self) -> u32 {
        self.offset
    }
}

/// A simple linear ring allocator for dynamic memory in shared memory.
/// This allocator uses a write cursor that wraps around when it reaches the end of the arena.
/// Old data is overwritten when new allocations are made (agnostic broadcasting pattern).
/// Consumers must be fast enough to read data before it gets overwritten.
pub struct LinearRingAllocator {
    write_cursor: AtomicU32,
    arena_start: u32,
    arena_size: u32,
}

impl LinearRingAllocator {
    /// Creates a new allocator with the given arena bounds.
    /// The arena starts at `arena_start` offset and has `arena_size` bytes available.
    pub fn new(arena_start: u32, arena_size: u32) -> Self {
        Self {
            write_cursor: AtomicU32::new(arena_start),
            arena_start,
            arena_size,
        }
    }

    /// Allocates a chunk of memory of the specified size.
    /// Returns `None` if allocation would exceed the arena bounds even after wrap-around.
    /// The size is automatically aligned to 8 bytes for compatibility with Python memoryview.
    pub fn allocate(&self, size: usize) -> Option<ShmPtr<u8>> {
        // Align size to 8-byte boundary (required for Python memoryview compatibility)
        let aligned_size = (size + 7) & !7;
        
        // Get current cursor position and atomically advance it
        let cursor = self.write_cursor.fetch_add(aligned_size as u32, Ordering::Relaxed);
        let end = cursor + aligned_size as u32;
        
        // Check if we need to wrap around
        if end > self.arena_start + self.arena_size {
            // Calculate remaining space from current position to end
            let remaining = (self.arena_start + self.arena_size) - cursor;
            
            if remaining >= aligned_size as u32 {
                // Wrap around is safe - we can fit at the beginning
                // Update cursor to point to the start of the allocation at beginning
                self.write_cursor.store(
                    self.arena_start + aligned_size as u32,
                    Ordering::Relaxed,
                );
                // Return pointer to beginning of arena
                return Some(ShmPtr::new(self.arena_start));
            } else {
                // Not enough space even at the beginning
                return None;
            }
        } else {
            // Normal allocation within current space
            self.write_cursor.store(end, Ordering::Relaxed);
            return Some(ShmPtr::new(cursor));
        }
    }

    /// Returns the current write cursor position.
    /// Useful for debugging and monitoring allocator state.
    pub fn current_cursor(&self) -> u32 {
        self.write_cursor.load(Ordering::Relaxed)
    }

    /// Returns the arena size.
    pub fn arena_size(&self) -> u32 {
        self.arena_size
    }

    /// Returns the arena start offset.
    pub fn arena_start(&self) -> u32 {
        self.arena_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shm_ptr_creation() {
        let ptr = ShmPtr::<u8>::new(1024);
        assert_eq!(ptr.offset(), 1024);
    }
    
    #[test]
    fn test_allocator_normal_allocation() {
        let allocator = LinearRingAllocator::new(4096, 1024 * 1024); // 1MB arena starting at 4KB
        
        // Allocate some memory
        let ptr1 = allocator.allocate(100).unwrap();
        assert_eq!(ptr1.offset(), 4096);
        
        let ptr2 = allocator.allocate(200).unwrap();
        assert_eq!(ptr2.offset(), 4196);
        
        // Verify cursor advanced
        assert_eq!(allocator.current_cursor(), 4396);
    }
    
    #[test]
    fn test_allocator_wrap_around() {
        let allocator = LinearRingAllocator::new(4096, 1024); // Small 1KB arena
        
        // Fill the arena
        let ptr1 = allocator.allocate(500).unwrap();
        assert_eq!(ptr1.offset(), 4096);
        
        // This should wrap around
        let ptr2 = allocator.allocate(600).unwrap();
        assert_eq!(ptr2.offset(), 4096); // Wrapped to beginning
        
        // Verify cursor is at end of allocation
        assert_eq!(allocator.current_cursor(), 5096);
    }
    
    #[test]
    fn test_allocator_alignment() {
        let allocator = LinearRingAllocator::new(4096, 1024);
        
        // Allocate odd size
        let ptr = allocator.allocate(13).unwrap();
        assert_eq!(ptr.offset(), 4096);
        
        // Next allocation should start at aligned position
        let ptr2 = allocator.allocate(13).unwrap();
        assert_eq!(ptr2.offset(), 4104); // 4096 + 8 (aligned)
    }
    
    #[test]
    fn test_allocator_full() {
        let allocator = LinearRingAllocator::new(4096, 100); // Tiny arena
        
        // Fill it up
        let ptr1 = allocator.allocate(50).unwrap();
        assert_eq!(ptr1.offset(), 4096);
        
        let ptr2 = allocator.allocate(50).unwrap();
        assert_eq!(ptr2.offset(), 4146); // 4096 + 46 (aligned)
        
        // Should fail - not enough space
        let ptr3 = allocator.allocate(10).unwrap();
        assert_eq!(ptr3.offset(), 4096); // Wrapped
        
        // Now it's full
        let ptr4 = allocator.allocate(1).unwrap();
        assert_eq!(ptr4.offset(), 4104); // 4096 + 8
        
        // Should fail - no space left
        assert!(allocator.allocate(1000).is_none());
    }
}
