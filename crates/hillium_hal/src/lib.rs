//! HilliumHAL - Hardware Abstraction Layer

pub mod traits;
pub mod drivers;

use traits::RobotDriver;
use drivers::{mock::MockDriver, mujoco::MuJoCoDriver};
use anyhow::Result;

/// Driver types
#[derive(Debug, Clone, Copy)]
pub enum DriverType {
    Mock,
    MuJoCo,
    // Future: Isaac, ROS2, Real
}

/// Create driver
pub fn create_driver(driver_type: DriverType, num_joints: usize) -> Result<Box<dyn RobotDriver>> {
    match driver_type {
        DriverType::Mock => {
            Ok(Box::new(MockDriver::new(num_joints)))
        }
        DriverType::MuJoCo => {
            Ok(Box::new(MuJoCoDriver::new(num_joints)))
        }
    }
}
