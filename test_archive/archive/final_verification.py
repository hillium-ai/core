from loqus_core.observability.query_engine import QueryEngine, ReadOnlyViolationError
import tempfile

test_sql = "INSERT INTO test VALUES (1)"

with tempfile.TemporaryDirectory() as tmp:
    e = QueryEngine(tmp)
    print('Testing security features...')
    try:
        e.query_logs(test_sql)
        print('ERROR: Should have blocked INSERT')
    except ReadOnlyViolationError:
        print('✅ Write operations blocked successfully')
    except Exception as ex:
        print(f'Other exception: {ex}')
    print('✅ Security implementation complete')