//! Mock robot driver for testing.

use crate::traits::{RobotDriver, SensorData, JointAngles, JointTorques};
use async_trait::async_trait;
use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct MockDriver {
    num_joints: usize,
    positions: JointAngles,
}

impl MockDriver {
    pub fn new(num_joints: usize) -> Self {
        Self {
            num_joints,
            positions: vec![0.0; num_joints],
        }
    }
}

#[async_trait]
impl RobotDriver for MockDriver {
    async fn init(&mut self) -> Result<()> {
        log::info!("MockDriver initialized with {} joints", self.num_joints);
        Ok(())
    }

    async fn set_target_positions(&mut self, positions: JointAngles) -> Result<()> {
        if positions.len() != self.num_joints {
            anyhow::bail!("Invalid positions length: {} != {}", positions.len(), self.num_joints);
        }

        // Simulate instant movement
        self.positions = positions;
        Ok(())
    }

    async fn set_target_torques(&mut self, _torques: JointTorques) -> Result<()> {
        // Mock: do nothing
        Ok(())
    }

    async fn read_sensors(&mut self) -> Result<SensorData> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos() as u64;

        Ok(SensorData {
            joint_positions: self.positions.clone(),
            joint_velocities: vec![0.0; self.num_joints],
            joint_torques: vec![0.0; self.num_joints],
            timestamp_ns: now,
        })
    }

    async fn emergency_stop(&mut self) -> Result<()> {
        log::warn!("MockDriver: EMERGENCY STOP");
        self.positions = vec![0.0; self.num_joints];
        Ok(())
    }

    fn num_joints(&self) -> usize {
        self.num_joints
    }

    fn name(&self) -> &'static str {
        "Mock"
    }

    async fn shutdown(&mut self) -> Result<()> {
        log::info!("MockDriver shutdown");
        Ok(())
    }
}
