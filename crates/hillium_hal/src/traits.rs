//! Hardware Abstraction Layer traits.
//! Defines common interface for different robot backends.

use async_trait::async_trait;
use anyhow::Result;

/// Joint angles (radians)
pub type JointAngles = Vec<f32>;

/// Joint torques (Newton-meters)
pub type JointTorques = Vec<f32>;

/// Sensor data from robot
#[derive(Debug, Clone)]
pub struct SensorData {
    pub joint_positions: JointAngles,
    pub joint_velocities: JointAngles,
    pub joint_torques: JointTorques,
    pub timestamp_ns: u64,
}

/// Robot driver trait
#[async_trait]
pub trait RobotDriver: Send + Sync {
    /// Initialize driver
    async fn init(&mut self) -> Result<()>;

    /// Set target joint positions
    async fn set_target_positions(&mut self, positions: JointAngles) -> Result<()>;

    /// Set target joint torques
    async fn set_target_torques(&mut self, torques: JointTorques) -> Result<()>;

    /// Read current sensor data
    async fn read_sensors(&mut self) -> Result<SensorData>;

    /// Emergency stop (halts all motion)
    async fn emergency_stop(&mut self) -> Result<()>;

    /// Get number of joints
    fn num_joints(&self) -> usize;

    /// Get driver name
    fn name(&self) -> &'static str;

    /// Shutdown driver
    async fn shutdown(&mut self) -> Result<()>;
}
