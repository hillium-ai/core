//! Golden Ratio Constants for Fibonacci Math Library

/// The Golden Ratio (φ)
pub const PHI: f64 = 1.618033988749895;

/// The inverse of the Golden Ratio (1/φ)
pub const INV_PHI: f64 = 0.6180339887498949; // 1/PHI = PHI - 1

/// The square root of 5
pub const SQRT_5: f64 = 2.23606797749979;

/// Mathematical proof that INV_PHI = PHI - 1
pub fn inv_phi_proof() -> f64 {
    PHI - 1.0
}

/// Verify that INV_PHI is indeed 1/PHI
pub fn verify_inverse() -> bool {
    (INV_PHI - 1.0 / PHI).abs() < 1e-15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_constant() {
        assert!((PHI - 1.618033988749895).abs() < 1e-15);
    }

    #[test]
    fn test_inv_phi_constant() {
        assert!((INV_PHI - 0.6180339887498949).abs() < 1e-15);
    }

    #[test]
    fn test_sqrt_5_constant() {
        assert!((SQRT_5 - 2.23606797749979).abs() < 1e-15);
    }

    #[test]
    fn test_inv_phi_proof() {
        assert!((inv_phi_proof() - INV_PHI).abs() < 1e-15);
    }

    #[test]
    fn test_verify_inverse() {
        assert!(verify_inverse());
    }
}