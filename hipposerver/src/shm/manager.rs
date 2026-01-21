use std::ffi::CString;
use std::ptr::{self, NonNull};
use libc::{c_char, c_int, c_void, shm_open, shm_unlink, ftruncate, mmap, munmap, close, O_CREAT, O_RDWR, PROT_READ, PROT_WRITE, MAP_SHARED, MAP_FAILED, S_IRUSR, S_IWUSR, mode_t};
use anyhow::{Result, bail};
use crate::shm::layout::HippoState;
use std::marker::{PhantomData, Send};

const SHM_SIZE: usize = 512 * 1024 * 1024; // 512 MB
const SHM_PERMISSIONS: mode_t = S_IRUSR | S_IWUSR; // 0o600

#[derive(Debug)]
pub struct ShmHandle {
    name: CString,
    ptr: NonNull<c_void>,
    fd: c_int,
    _send_marker: PhantomData<Box<dyn Send + Sync>>,
}

unsafe impl Send for ShmHandle {}
unsafe impl Sync for ShmHandle {}

impl ShmHandle {
    /// Creates a new shared memory segment with the given name.
    pub fn create(name: &str) -> Result<Self> {
        let c_name = CString::new(name)?;
        let fd = unsafe {
            shm_open(
                c_name.as_ptr(),
                O_CREAT | O_RDWR,
                SHM_PERMISSIONS,
            )
        };
        if fd == -1 {
            bail!("Failed to create shared memory segment");
        }

        // Set the size of the shared memory segment
        let result = unsafe { ftruncate(fd, SHM_SIZE as i64) };
        if result == -1 {
            unsafe { close(fd) };
            bail!("Failed to set shared memory size");
        }

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                SHM_SIZE,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            unsafe { close(fd) };
            bail!("Failed to map shared memory segment");
        }

        Ok(ShmHandle {
            name: c_name,
            ptr: NonNull::new(ptr).unwrap(), // SAFETY: mmap succeeded
            fd,
            _send_marker: PhantomData,
        })
    }

    /// Opens an existing shared memory segment with the given name.
    pub fn open(name: &str) -> Result<Self> {
        let c_name = CString::new(name)?;
        let fd = unsafe {
            shm_open(
                c_name.as_ptr(),
                O_RDWR,
                SHM_PERMISSIONS,
            )
        };
        if fd == -1 {
            bail!("Failed to open shared memory segment");
        }

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                SHM_SIZE,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            unsafe { close(fd) };
            bail!("Failed to map shared memory segment");
        }

        Ok(ShmHandle {
            name: c_name,
            ptr: NonNull::new(ptr).unwrap(), // SAFETY: mmap succeeded
            fd,
            _send_marker: PhantomData,
        })
    }

    /// Returns a raw pointer to the start of the shared memory segment.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    /// Returns a mutable raw pointer to the start of the shared memory segment.
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    /// Initializes HippoState at the start of the shared memory segment.
    pub fn initialize_state(&mut self) -> Result<()> {
        let state_ptr = self.ptr.as_ptr() as *mut HippoState;
        unsafe {
            // Zero out the memory first
            std::ptr::write_bytes(state_ptr, 0, 1);
            // Initialize HippoState
            (*state_ptr).magic.store(0x48494C4C, std::sync::atomic::Ordering::Relaxed);
            (*state_ptr).current_intent.store(0, std::sync::atomic::Ordering::Relaxed);
            (*state_ptr).safety_lock.store(false, std::sync::atomic::Ordering::Relaxed);
            (*state_ptr).boot_timestamp_ns.store(0, std::sync::atomic::Ordering::Relaxed);
            (*state_ptr).layer7_emergency_brake.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Gets a reference to the HippoState in shared memory.
    pub fn state(&self) -> &HippoState {
        unsafe { &*(self.ptr.as_ptr() as *const HippoState) }
    }

    /// Gets a mutable reference to the HippoState in shared memory.
    pub fn state_mut(&mut self) -> &mut HippoState {
        unsafe { &mut *(self.ptr.as_ptr() as *mut HippoState) }
    }
}

impl Drop for ShmHandle {
    /// Unmaps the shared memory segment and closes the file descriptor.
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr.as_ptr(), SHM_SIZE);
            close(self.fd);
        }
    }
}

/// Unlinks (deletes) a shared memory segment with the given name.
pub fn unlink(name: &str) -> Result<()> {
    let c_name = CString::new(name)?;
    let result = unsafe { shm_unlink(c_name.as_ptr()) };
    if result == -1 {
        bail!("Failed to unlink shared memory segment");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shm_creation() {
        let mut handle = ShmHandle::create("/test_shm_creation").unwrap();
        assert_eq!(handle.as_ptr() as usize % 4096, 0); // Page-aligned
        assert_eq!(unsafe { *(handle.as_ptr() as *const u32) }, 0); // Should be zero-initialized

        // Test HippoState initialization
        handle.initialize_state().unwrap();
        assert_eq!(handle.state().magic.load(std::sync::atomic::Ordering::Relaxed), 0x48494C4C);
        assert_eq!(handle.state().current_intent.load(std::sync::atomic::Ordering::Relaxed), 0);

        // Test that we can write to it
        let test_val: u32 = 0xDEADBEEF;
        unsafe {
            *(handle.as_mut_ptr() as *mut u32) = test_val;
        }
        assert_eq!(unsafe { *(handle.as_ptr() as *const u32) }, test_val);

        drop(handle); // This should unmap the segment

        // Reopen and check if it's still there (should be zeroed)
        let handle2 = ShmHandle::open("/test_shm_creation").unwrap();
        assert_eq!(unsafe { *(handle2.as_ptr() as *const u32) }, 0); // Zeroed again
        drop(handle2);

        unlink("/test_shm_creation").unwrap();
    }
}
