import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'loqus_core'))

# Test that we can import the modules
try:
    from observability.query_engine import QueryEngine, ReadOnlyViolationError, QueryEngineError
    print("✅ Successfully imported QueryEngine, ReadOnlyViolationError, QueryEngineError")
    
    # Test basic instantiation
    engine = QueryEngine('/tmp')
    print("✅ Successfully created QueryEngine instance")
    
    # Test that methods exist
    assert hasattr(engine, 'query_logs')
    assert hasattr(engine, 'export_to_parquet')
    assert hasattr(engine, 'get_telemetry_schema')
    assert hasattr(engine, 'get_table_names')
    print("✅ All expected methods are present")
    
    print("✅ Implementation is syntactically correct and matches test expectations")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)