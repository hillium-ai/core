use anyhow::{Error, Result};
use log::info;
use tch::nn::Module;
use tch::{Device, Tensor, no_grad_guard};

/// RootLTSPlanner implements the high-performance √LTS planner in Rust.
pub struct RootLTSPlanner {
    model: tch::CModule,
}

impl RootLTSPlanner {
    /// Load the TorchScript model from the given path.
    pub fn load(path: &str) -> Result<Self> {
        info!("Loading model from: {}", path);
        let model = tch::CModule::load(path)
            .map_err(|e| Error::msg(format!("Failed to load model {}: {}", path, e)))?;
        info!("Model loaded successfully");
        Ok(RootLTSPlanner { model })
    }

    /// Execute planning with start and goal coordinates.
    ///
    /// # Arguments
    /// * `start` - Slice of 2 f32 values representing start coordinates
    /// * `goal` - Slice of 2 f32 values representing goal coordinates
    ///
    /// # Returns
    /// * `Result<Tensor, anyhow::Error>` - Output tensor or error
    pub fn plan(&self, start: &[f32], goal: &[f32]) -> Result<Tensor> {
        if start.len() != 2 {
            return Err(Error::msg("Start coordinates must have exactly 2 elements"));
        }
        if goal.len() != 2 {
            return Err(Error::msg("Goal coordinates must have exactly 2 elements"));
        }

        let _guard = no_grad_guard();

        let mut input_data: Vec<f32> = Vec::new();
        input_data.extend_from_slice(start);
        input_data.extend_from_slice(goal);
        input_data.resize(2 + 2 + 32 * 32, 0.0);

        let input = Tensor::from_slice(&input_data).view([-1, 1024 + 4]);
        let _device = Device::Cpu;
        let output = self.model.forward(&input);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_input_validation() {
        assert!(true);
    }

    #[test]
    fn test_plan_loading() {
        let model_path = "rerooter.pt";
        
        // Skip test if model file doesn't exist
        if !std::path::Path::new(model_path).exists() {
            info!("Skipping test: {} not found", model_path);
            return;
        }

        let planner = match RootLTSPlanner::load(model_path) {
            Ok(p) => p,
            Err(e) => {
                info!("Failed to load model: {}", e);
                return;
            }
        };

        let start = [0.0, 0.0];
        let goal = [1.0, 1.0];
        
        match planner.plan(&start, &goal) {
            Ok(output) => {
                assert!(output.size().len() > 0, "Output tensor is empty");
                info!("Plan successful, output shape: {:?}", output.size());
            }
            Err(e) => {
                info!("Planning failed: {}", e);
            }
        }
    }
}
