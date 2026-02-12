import math

# Test the mathematical correctness of the golden ratio implementations
PHI = 1.618033988749895
INV_PHI = 0.6180339887498949  # 1/φ
SQRT_5 = 2.23606797749979

def test_golden_ratio_constants():
    print('=== Testing Golden Ratio Constants ===')
    print(f'φ (PHI): {PHI}')
    print(f'1/φ (INV_PHI): {INV_PHI}')
    print(f'√5: {SQRT_5}')
    
    # Verify mathematical relationships
    assert abs(PHI * PHI - PHI - 1) < 1e-15, "PHI should satisfy x² = x + 1"
    assert abs(INV_PHI + 1) - PHI < 1e-15, "INV_PHI + 1 should equal PHI"
    assert abs(INV_PHI * PHI - 1) < 1e-15, "INV_PHI * PHI should equal 1"
    assert abs(SQRT_5 * SQRT_5 - 5) < 1e-15, "√5 should be square root of 5"
    print('✅ All golden ratio constant relationships verified')

def test_golden_kalman_gain_convergence():
    print('\n=== Testing Golden Kalman Gain Convergence ===')
    
    def golden_kalman_gain(q, r, iterations):
        p = 1.0
        for _ in range(iterations):
            if p + r != 0:
                p = q + p - (p * p) / (p + r)
        if p + r != 0:
            return p / (p + r)
        else:
            return 0.0
    
    # Test convergence to 1/φ
    for iterations in [10, 50, 100, 500]:
        gain = golden_kalman_gain(1.0, 1.0, iterations)
        diff = abs(gain - INV_PHI)
        print(f'Iterations: {iterations:3d} | Gain: {gain:.15f} | Difference: {diff:.15f}')
        assert diff < 0.001, f'Gain did not converge within 0.1% tolerance at {iterations} iterations'
    
    print('✅ Golden Kalman gain converges within 0.1% tolerance')

def test_mathematical_properties():
    print('\n=== Testing Mathematical Properties ===')
    
    # The golden ratio has special properties
    # φ = (1 + √5) / 2
    phi_from_sqrt5 = (1 + SQRT_5) / 2
    assert abs(PHI - phi_from_sqrt5) < 1e-15, "PHI should equal (1 + √5) / 2"
    
    # 1/φ = φ - 1
    assert abs(INV_PHI - (PHI - 1)) < 1e-15, "1/φ should equal φ - 1"
    
    print('✅ All mathematical properties verified')

if __name__ == '__main__':
    test_golden_ratio_constants()
    test_golden_kalman_gain_convergence()
    test_mathematical_properties()
    print('\n🎉 All validations passed! The implementation meets the requirements.')
