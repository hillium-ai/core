// Integration test for telemetry JSONL output

use hipposerver::telemetry::{LogEntry, create_log_entry};
use serde_json;

#[test]
fn test_jsonl_format() {
    // Create a log entry
    let entry = create_log_entry(
        "test-correlation-123".to_string(),
        "test_component",
        "INFO",
        "This is a test message"
    );

    // Serialize to JSON
    let json = serde_json::to_string(&entry).unwrap();
    
    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    
    // Verify all required fields are present
    assert!(parsed.get("correlation_id").is_some());
    assert!(parsed.get("component").is_some());
    assert!(parsed.get("level").is_some());
    assert!(parsed.get("message").is_some());
    assert!(parsed.get("timestamp").is_some());
    
    // Verify values
    assert_eq!(parsed["correlation_id"], "test-correlation-123");
    assert_eq!(parsed["component"], "test_component");
    assert_eq!(parsed["level"], "INFO");
    assert_eq!(parsed["message"], "This is a test message");
    
    // Verify timestamp format (should be ISO 8601)
    let timestamp = &parsed["timestamp"];
    assert!(timestamp.as_str().unwrap().contains('T'));
    assert!(timestamp.as_str().unwrap().contains('Z') || timestamp.as_str().unwrap().contains('+'));
}

#[test]
fn test_log_entry_struct() {
    // Test that LogEntry struct has all required fields
    let entry = LogEntry {
        correlation_id: "test-id".to_string(),
        component: "test_comp".to_string(),
        level: "DEBUG".to_string(),
        message: "Debug message".to_string(),
        timestamp: "2023-01-01T00:00:00Z".to_string(),
    };
    
    // Verify serialization
    let json = serde_json::to_string(&entry).unwrap();
    assert!(json.contains("\"correlation_id\":\"test-id\""));
    assert!(json.contains("\"component\":\"test_comp\""));
    assert!(json.contains("\"level\":\"DEBUG\""));
    assert!(json.contains("\"message\":\"Debug message\""));
    assert!(json.contains("\"timestamp\":\"2023-01-01T00:00:00Z\""));
}
