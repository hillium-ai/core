use anyhow::Result;
use log::{info, warn};
use tokio::time::{interval, Duration};
use associative_core::AssociativeCore;
use working_memory::WorkingMemory;

pub struct ConsolidationScheduler {
    interval_duration: Duration,
    core: AssociativeCore,
    memory: WorkingMemory,
}

impl ConsolidationScheduler {
    pub fn new(core: AssociativeCore, memory: WorkingMemory) -> Self {
        Self {
            interval_duration: Duration::from_secs(30), // Default 30s
            core,
            memory,
        }
    }

    pub fn with_interval(mut self, duration: Duration) -> Self {
        self.interval_duration = duration;
        self
    }

    pub async fn run(&mut self) -> Result<()> {
        let mut ticker = interval(self.interval_duration);
        loop {
            ticker.tick().await;
            info!("Starting consolidation cycle");

            // Extract weights from AssociativeCore
            let weights = self.core.export_weights().await?;
            let knowledge_items = weights_to_knowledge(weights);

            // Simulate batch transfer to Episodic Storage
            if !knowledge_items.is_empty() {
                info!("Extracted {} knowledge items for consolidation", knowledge_items.len());
                // In Phase 3: Push to Qdrant/Neo4j
                // For now, log only
                for item in &knowledge_items {
                    info!("Would push to Episodic: {:?}", item);
                }
            } else {
                info!("No high-weight connections found to consolidate.");
            }
        }
    }
}

// Heuristic: filter weights > 0.5
fn weights_to_knowledge(weights: Vec<(String, f32)>) -> Vec<(String, f32)> {
    weights
        .into_iter()
        .filter(|(_, weight)| *weight > 0.5)
        .collect()
}
