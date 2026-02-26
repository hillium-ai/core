//! PowerInfer backend implementation

use std::collections::HashMap;
use std::ffi::CString;


use serde::{Deserialize, Serialize};

use crate::ffi::{PowerInferModel, powerinfer_load_model, powerinfer_generate};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            stop_sequences: vec![],
            seed: None,
        }
    }
}

/// Result from text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResult {
    pub text: String,
    pub tokens_generated: u32,
    pub latency_ms: f64,
    pub finish_reason: String,
}

/// PowerInferBackend implementation
pub struct PowerInferBackend {
    model: Option<PowerInferModel>,
    is_model_loaded: bool,
}

impl PowerInferBackend {
    /// Create a new PowerInferBackend
    pub fn new() -> Self {
        Self {
            model: None,
            is_model_loaded: false,
        }
    }
}

impl Default for PowerInferBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of InferenceBackend trait
impl PowerInferBackend {
    /// Load a model from disk
    pub fn load_model(&mut self, path: &str, config: &HashMap<String, serde_json::Value>) -> Result<(), Box<dyn std::error::Error>> {
        // Convert path to C string
        let c_path = CString::new(path)?;
        
        // Convert config to JSON string
        let config_json = serde_json::to_string(config)?;
        let c_config = CString::new(config_json)?;
        
        // Call C function to load model
        let model_handle = powerinfer_load_model(c_path.as_ptr(), c_config.as_ptr());
        
        if model_handle.is_null() {
            return Err("Failed to load model".into());
        }
        
        self.model = Some(PowerInferModel { handle: model_handle });
        self.is_model_loaded = true;
        
        Ok(())
    }
    
    /// Generate text from prompt
    pub fn generate(&self, prompt: &str, params: &GenerateParams) -> Result<GenerateResult, Box<dyn std::error::Error>> {
        if self.model.is_none() {
            return Err("Model not loaded".into());
        }
        
        // Convert prompt to C string
        let c_prompt = CString::new(prompt)?;
        
        // Convert params to JSON string
        let params_json = serde_json::to_string(params)?;
        let c_params = CString::new(params_json)?;
        
        // Call C function to generate
        let result_ptr = powerinfer_generate(
            self.model.as_ref().unwrap().handle,
            c_prompt.as_ptr(),
            c_params.as_ptr(),
        );
        
        if result_ptr.is_null() {
            return Err("Generation failed".into());
        }
        
        // Convert result back to Rust
        let result_str = unsafe { std::ffi::CStr::from_ptr(result_ptr) };
        let result_json = result_str.to_str()?;
        
        // Parse JSON result
        let result: GenerateResult = serde_json::from_str(result_json)?;
        
        Ok(result)
    }
    
    /// Unload model and release resources
    pub fn unload(&mut self) {
        self.model = None;
        self.is_model_loaded = false;
    }
    
    /// Check if model is currently loaded
    pub fn is_loaded(&self) -> bool {
        self.is_model_loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_backend_creation() {
        let backend = PowerInferBackend::new();
        assert!(!backend.is_loaded());
    }
    
    #[test]
    fn test_default_params() {
        let params = GenerateParams::default();
        assert_eq!(params.max_tokens, 512);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 40);
    }
}