import sys
sys.path.insert(0, '.')

from loqus_core.observability.query_engine import QueryEngine, ReadOnlyViolationError
import tempfile
import os

def test_basic_functionality():
    print("Testing basic query engine functionality...")
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        engine = QueryEngine(tmp_dir)
        
        # Test that we can import and instantiate
        print("✓ QueryEngine instantiated successfully")
        
        # Test read-only enforcement with a simple write operation
        try:
            engine.query_logs("INSERT INTO test VALUES (1)")
            print("✗ Read-only enforcement failed - should have raised exception")
        except ReadOnlyViolationError:
            print("✓ Read-only enforcement working correctly")
        except Exception as e:
            print(f"? Unexpected exception: {e}")
        
        # Test valid SELECT query
        try:
            result = engine.query_logs("SELECT 1 as test")
            print("✓ Valid SELECT query executed successfully")
        except Exception as e:
            print(f"? Valid query failed: {e}")
            
    print("Basic functionality test completed.")

if __name__ == "__main__":
    test_basic_functionality()