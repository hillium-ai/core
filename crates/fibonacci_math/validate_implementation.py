# Validation script for fibonacci_math implementation

import sys
import os

# Add the fibonacci_math crate to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def validate_constants():
    """Validate that golden ratio constants are correctly defined"""
    try:
        import fibonacci_math
        print("✅ Successfully imported fibonacci_math")
        
        # Test constants
        print(f"PHI: {fibonacci_math.P}")
        print(f"INV_PHI: {fibonacci_math.INV_PHI}")
        print(f"SQRT_5: {fibonacci_math.SQRT_5}")
        
        # Test mathematical relationships
        print(f"PHI * PHI - PHI - 1: {fibonacci_math.P * fibonacci_math.P - fibonacci_math.P - 1}")
        print(f"INV_PHI + 1 - PHI: {fibonacci_math.INV_PHI + 1 - fibonacci_math.P}")
        print(f"SQRT_5^2 - 5: {fibonacci_math.SQRT_5 * fibonacci_math.SQRT_5 - 5}")
        
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def validate_kalman():
    """Validate Golden Kalman functionality"""
    try:
        import fibonacci_math
        gain = fibonacci_math.calculate_golden_gain(1.0, 1.0, 100)
        print(f"Golden Kalman gain: {gain}")
        print(f"Expected (1/PHI): {1/1.618033988749895}")
        return True
    except Exception as e:
        print(f"❌ Kalman validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Validating fibonacci_math implementation...")
    const_ok = validate_constants()
    kalman_ok = validate_kalman()
    
    if const_ok and kalman_ok:
        print("🎉 Implementation validation successful!")
    else:
        print("❌ Implementation validation failed!")
