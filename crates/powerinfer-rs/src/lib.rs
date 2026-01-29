//! PowerInfer backend for HilliumOS
//!
//! This crate provides FFI bindings to the PowerInfer C++ library
//! for hybrid CPU/GPU sparse inference.

pub mod ffi;
pub mod backend;

pub use backend::PowerInferBackend;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_powerinfer_backend_exists() {
        assert!(true); // Placeholder test
    }
}