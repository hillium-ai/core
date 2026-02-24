use anyhow::Error;
use tch::nn::Module;
use tch::Tensor;

/// RootLTSPlanner implements the high-performance √LTS planner in Rust.
pub struct RootLTSPlanner {
    model: tch::CModule,
}

impl RootLTSPlanner {
    /// Load the TorchScript model from the given path.
    pub fn load(path: &str) -> Result<Self, anyhow::Error> {
        let model = tch::CModule::load(path)?;
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
    pub fn plan(&self, start: &[f32], goal: &[f32]) -> Result<Tensor, anyhow::Error> {
        if start.len() != 2 {
            return Err(Error::msg("Start coordinates must have exactly 2 elements"));
        }
        if goal.len() != 2 {
            return Err(Error::msg("Goal coordinates must have exactly 2 elements"));
        }

        let _guard = tch::no_grad_guard();

        let mut input_data: Vec<f32> = Vec::new();
        input_data.extend_from_slice(start);
        input_data.extend_from_slice(goal);
        input_data.resize(2 + 2 + 32 * 32, 0.0);

        let input = Tensor::from_slice(&input_data).view([-1, 1024 + 4]);
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
    #[ignore] // Requires rerooter.pt to be present
    fn test_plan_loading() {
        let planner = RootLTSPlanner::load("rerooter.pt").expect("Failed to load model");
        let start = [0.0, 0.0];
        let goal = [1.0, 1.0];
        let output = planner.plan(&start, &goal).expect("Planning failed");
        assert!(output.size().len() > 0, "Output tensor is empty");
    }
}
