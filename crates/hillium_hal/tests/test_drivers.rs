use hillium_hal::{create_driver, DriverType};

#[tokio::test]
async fn test_mock_driver_lifecycle() {
    let mut driver = create_driver(DriverType::Mock, 6).unwrap();

    // Test initialization
    driver.init().await.unwrap();
    assert_eq!(driver.num_joints(), 6);

    // Test setting positions
    let positions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    driver.set_target_positions(positions.clone()).await.unwrap();

    // Test reading sensors
    let sensors = driver.read_sensors().await.unwrap();
    assert_eq!(sensors.joint_positions, positions);
    assert_eq!(sensors.joint_positions.len(), 6);
    assert_eq!(sensors.joint_velocities.len(), 6);
    assert_eq!(sensors.joint_torques.len(), 6);

    // Test emergency stop
    driver.emergency_stop().await.unwrap();

    // After E-Stop, positions should be reset to zero
    let sensors = driver.read_sensors().await.unwrap();
    assert_eq!(sensors.joint_positions, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Test shutdown
    driver.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_mock_driver_invalid_positions() {
    let mut driver = create_driver(DriverType::Mock, 6).unwrap();
    driver.init().await.unwrap();

    // Test with wrong number of joints
    let invalid_positions = vec![0.1, 0.2];
    let result = driver.set_target_positions(invalid_positions).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_driver_factory() {
    // Test Mock driver creation
    let mock_driver = create_driver(DriverType::Mock, 6).unwrap();
    assert_eq!(mock_driver.name(), "Mock");
    assert_eq!(mock_driver.num_joints(), 6);

    // Test MuJoCo driver creation
    let mujoco_driver = create_driver(DriverType::MuJoCo, 7).unwrap();
    assert_eq!(mujoco_driver.name(), "MuJoCo");
    assert_eq!(mujoco_driver.num_joints(), 7);
}
