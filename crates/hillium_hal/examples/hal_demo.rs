//! Demo of HilliumHAL usage

use hillium_hal::{create_driver, DriverType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Create mock driver
    let mut driver = create_driver(DriverType::Mock, 6)?;

    // Initialize
    driver.init().await?;

    println!("Driver: {}, Joints: {}", driver.name(), driver.num_joints());

    // Set positions
    let positions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    driver.set_target_positions(positions.clone()).await?;

    // Read sensors
    let sensors = driver.read_sensors().await?;
    println!("Positions: {:?}", sensors.joint_positions);

    // E-Stop
    driver.emergency_stop().await?;

    // Read sensors after E-Stop
    let sensors = driver.read_sensors().await?;
    println!("Positions after E-Stop: {:?}", sensors.joint_positions);

    // Shutdown
    driver.shutdown().await?;

    Ok(())
}
