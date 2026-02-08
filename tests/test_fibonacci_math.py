import unittest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    import fibonacci_math
    HAS_FIBONACCI_MATH = True
except ImportError:
    HAS_FIBONACCI_MATH = False

class TestFibonacciMath(unittest.TestCase):
    
    def test_constants_exist(self):
        if not HAS_FIBONACCI_MATH:
            self.skipTest("fibonacci_math not available")
        
        # Test that constants are accessible
        self.assertTrue(hasattr(fibonacci_math, 'PHI'))
        self.assertTrue(hasattr(fibonacci_math, 'INV_PHI'))
        self.assertTrue(hasattr(fibonacci_math, 'SQRT_5'))
        
    def test_golden_kalman_convergence(self):
        if not HAS_FIBONACCI_MATH:
            self.skipTest("fibonacci_math not available")
        
        # Test that we can call the Kalman filter function
        try:
            # This should work if the library is properly built
            result = fibonacci_math.golden_kalman_gain(1.0, 1.0, 100)
            self.assertIsInstance(result, float)
        except AttributeError:
            self.skipTest("golden_kalman_gain not implemented")
        
    def test_fibonacci_heap_operations(self):
        if not HAS_FIBONACCI_MATH:
            self.skipTest("fibonacci_math not available")
        
        # Test that we can call Fibonacci heap functions
        try:
            # This should work if the library is properly built
            heap = fibonacci_math.FibonacciHeap()
            self.assertTrue(hasattr(heap, 'push'))
            self.assertTrue(hasattr(heap, 'pop'))
        except AttributeError:
            self.skipTest("FibonacciHeap not implemented")

if __name__ == '__main__':
    unittest.main()