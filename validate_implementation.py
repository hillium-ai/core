import sys
import os

cwd = os.getcwd()
print(f'Working directory: {cwd}')

# Test that we can import the module
try:
    from loqus_core.observability.query_engine import QueryEngine, ReadOnlyViolationError
    print('✅ Successfully imported QueryEngine and ReadOnlyViolationError')
    
    # Test that we can import duckdb
    import duckdb
    print('✅ Successfully imported duckdb')
    
    # Test that we can import sqlparse
    import sqlparse
    print('✅ Successfully imported sqlparse')
    
    # Test that we can import polars
    import polars as pl
    print('✅ Successfully imported polars')
    
    print('\n🎉 All dependencies and modules imported successfully!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}')
    sys.exit(1)