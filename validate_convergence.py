import math

# Constants
PHI = 1.618033988749895
INV_PHI = 0.6180339887498949  # 1/PHI
SQRT_5 = 2.23606797749979

def golden_kalman_gain(q, r, iterations):
    '''
    Golden Kalman gain function that should converge to 1/φ
    '''
    p = 1.0
    for _ in range(iterations):
        if p + r != 0:
            p = q + p - (p * p) / (p + r)
    if p + r != 0:
        return p / (p + r)
    else:
        return 0.0

def test_convergence():
    print(f'Expected value (1/φ): {INV_PHI}')
    print(f'Expected value (φ): {PHI}')
    print(f'SQRT_5: {SQRT_5}')
    
    # Test with different iteration counts
    for iterations in [10, 50, 100, 500]:
        gain = golden_kalman_gain(1.0, 1.0, iterations)
        diff = abs(gain - INV_PHI)
        print(f'Iterations: {iterations:3d} | Gain: {gain:.15f} | Difference: {diff:.15f} | Within 0.1%: {diff < 0.001}')

if __name__ == '__main__':
    test_convergence()
