use ndarray::prelude::*;
use rand::Rng;
use std::sync::{Arc, RwLock};
use anyhow::{Result, anyhow};

#[derive(Debug)]
pub struct AssociativeCore {
    fast_weights: Arc<RwLock<Array2<f32>>>,
    momentum_buffer: Arc<RwLock<Array2<f32>>>,
    beta: f32,            // Momentum coefficient
    surprise_threshold: f32,
    decay_rate: f32,
}

impl AssociativeCore {
    pub fn new(
        rows: usize,
        cols: usize,
        beta: f32,
        surprise_threshold: f32,
        decay_rate: f32,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let fast_weights = Array2::<f32>::zeros((rows, cols)).mapv(|_: f32| rng.gen_range(-0.01..0.01));
        let momentum_buffer = Array2::<f32>::zeros((rows, cols));

        Self {
            fast_weights: Arc::new(RwLock::new(fast_weights)),
            momentum_buffer: Arc::new(RwLock::new(momentum_buffer)),
            beta,
            surprise_threshold,
            decay_rate,
        }
    }

    /// Predicts the output vector given a context vector using matrix multiplication.
    pub fn predict(&self, context: &Array1<f32>) -> Result<Array1<f32>> {
        let weights = self.fast_weights.read().map_err(|_| anyhow!("Lock poisoned"))?;
        // Matrix-vector multiplication: W * x
        let prediction = weights.dot(context);
        Ok(prediction)
    }

    /// Updates fast weights online if the surprise exceeds the threshold.
    /// Returns true if an update occurred.
    pub fn update_online(&self, context: &Array1<f32>, target: &Array1<f32>) -> Result<bool> {
        let prediction = self.predict(context)?;
        
        // Calculate error (surprise)
        let error = target - &prediction; // Element-wise subtraction
        let mse = error.mapv(|x| x.powi(2)).mean().unwrap_or(0.0);

        if mse > self.surprise_threshold {
            // Apply delta rule with momentum and decay
            let mut weights = self.fast_weights.write().map_err(|_| anyhow!("Lock poisoned"))?;
            let mut momentum = self.momentum_buffer.write().map_err(|_| anyhow!("Lock poisoned"))?;

            // Delta rule: dW = learning_rate * error * context^T
            // Here we use Outer Product of error and context
            // outer(a, b)_ij = a_i * b_j
            // So we want error * context^T -> Shape (rows, cols)
            // Error is (rows,), Context is (cols,)
            // We need to confirm shapes. Assuming strict shapes for now.
            
            // Simplified update logic inspired by Titans:
            // W_new = W_old * (1 - decay) + momentum
            
            // Applying decay
            weights.mapv_inplace(|x| x * (1.0 - self.decay_rate));

            // Calculate gradient (simplified)
            // grad = -error * input
            // But for online update we add:
            // W += beta * momentum + (1-beta) * gradient
            // This is complex to do blindly without full spec specs.
            // Implementing basic Hebbian-like update for MVP as per spec hint:
            // "update weights based on surprise"
            
            // Let's assume a learning rate (eta) of 0.1 for now.
            let eta = 0.1;
            
            // Outer product for gradient
            // We need 2D view for dot product to produce matrix? No, output product is manual in ndarray usually?
            // Or explicit loop.
            
            let rows = weights.nrows();
            let cols = weights.ncols();
            
            for i in 0..rows {
                for j in 0..cols {
                    let grad = error[i] * context[j];
                    momentum[[i,j]] = self.beta * momentum[[i,j]] + (1.0 - self.beta) * grad;
                    weights[[i,j]] += eta * momentum[[i,j]];
                }
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_associative_core_initialization() {
        let core = AssociativeCore::new(10, 10, 0.9, 0.01, 0.01);
        let context = Array1::from_elem(10, 0.5);
        let prediction = core.predict(&context).unwrap();
        assert_eq!(prediction.len(), 10);
    }

    #[test]
    fn test_surprise_gating_update() {
        let core = AssociativeCore::new(5, 5, 0.9, 0.0001, 0.01); // Low threshold
        let context = Array1::from_elem(5, 1.0);
        let target = Array1::from_elem(5, 10.0); // High target -> High error -> High surprise
        
        let updated = core.update_online(&context, &target).unwrap();
        assert!(updated, "Should have updated due to high surprise");
    }

    #[test]
    fn test_no_update_low_surprise() {
        let core = AssociativeCore::new(5, 5, 0.9, 100.0, 0.01); // High threshold
        let context = Array1::from_elem(5, 1.0);
        let target = Array1::from_elem(5, 1.0); // Target close to zero-init prediction -> Low error
        
        // Actually prediction is near 0. Target is 1. Error is 1. MSE is 1.
        // Threshold is 100. 1 < 100.
        
        let updated = core.update_online(&context, &target).unwrap();
        assert!(!updated, "Should NOT have updated due to low surprise");
    }
}
