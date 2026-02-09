// Golden Ratio constants

/// The Golden Ratio (φ)
pub const PHI: f64 = 1.618033988749895;

/// The inverse of the Golden Ratio (1/φ)
pub const INV_PHI: f64 = 0.6180339887498949; // 1/PHI = PHI - 1

/// The square root of 5
pub const SQRT_5: f64 = 2.23606797749979;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_golden_constants() {
        assert_eq!(PHI * INV_PHI, 1.0);
        assert_eq!(PHI - 1.0, INV_PHI);
        assert_eq!(SQRT_5 * SQRT_5, 5.0);
    }
}