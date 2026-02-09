//! Golden ratio constants

/// The golden ratio φ (phi)
pub const PHI: f64 = 1.618033988749895;

/// The inverse of the golden ratio 1/φ
pub const INV_PHI: f64 = 0.6180339887498949;

/// The square root of 5
pub const SQRT_5: f64 = 2.23606797749979;

/// Validates that the constants are mathematically correct
pub fn validate_constants() -> bool {
    (PHI - 1.0) == INV_PHI &&
    (PHI * PHI - PHI - 1.0).abs() < 1e-10 &&
    (SQRT_5 * SQRT_5 - 5.0).abs() < 1e-10
}
