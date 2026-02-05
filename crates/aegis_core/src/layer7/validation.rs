/// Validation result types for Aegis Layer 7

/// Result of validation
#[derive(Debug, Clone)]
pub enum ValidationResult {
    Approved,
    Rejected { 
        reason: String,
        evidence: String
    }
}

/// Synthetic input detected error
pub struct SyntheticInputDetected;
