// Telemetry and structured logging implementation for HippoServer
//
// This module provides:
// - LogEntry struct matching the RobotTelemetry spec
// - JSONL formatting for all logs
// - Correlation ID injection for distributed tracing

use serde::{Deserialize, Serialize};
use tracing_subscriber::{fmt, EnvFilter};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// LogEntry struct matching the RobotTelemetry specification
/// Contains all required fields for the Cognitive Council auditing system
#[derive(Debug, Serialize, Deserialize)]
pub struct LogEntry {
    /// Unique identifier for correlating logs across components
    pub correlation_id: String,
    /// The component emitting the log (e.g., "hipposerver", "robot_controller")
    pub component: String,
    /// Log level (INFO, WARN, ERROR, DEBUG, TRACE)
    pub level: String,
    /// The actual log message
    pub message: String,
    /// Timestamp in ISO 8601 format
    pub timestamp: String,
}

/// Initialize the tracing subscriber with JSONL formatting
/// This should be called once at application startup
pub fn init() {
    let subscriber = fmt::Subscriber::builder()
        .json() // Output as JSON
        .flatten_event(true) // Flatten fields
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .finish();

    // In a real production system we might wrap this to inject more context
    tracing::subscriber::set_global_default(subscriber)
        .expect("Unable to set global default subscriber");
}

/// Helper function to create a new LogEntry with current timestamp
pub fn create_log_entry(
    correlation_id: String,
    component: &str,
    level: &str,
    message: &str,
) -> LogEntry {
    use chrono::Utc;
    let timestamp = Utc::now().to_rfc3339();
    
    LogEntry {
        correlation_id,
        component: component.to_string(),
        level: level.to_string(),
        message: message.to_string(),
        timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_entry_serialization() {
        let entry = LogEntry {
            correlation_id: "abc123".to_string(),
            component: "test_component".to_string(),
            level: "INFO".to_string(),
            message: "Test message".to_string(),
            timestamp: "2023-01-01T00:00:00Z".to_string(),
        };

        let serialized = serde_json::to_string(&entry).unwrap();
        assert!(serialized.contains("\"correlation_id\":\"abc123\""));
        assert!(serialized.contains("\"component\":\"test_component\""));
        assert!(serialized.contains("\"level\":\"INFO\""));
        assert!(serialized.contains("\"message\":\"Test message\""));
        assert!(serialized.contains("\"timestamp\":\"2023-01-01T00:00:00Z\""));
    }
}
