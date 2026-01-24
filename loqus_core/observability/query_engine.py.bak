import duckdb
import logging
from typing import Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connection variable
_connection = None


class QueryEngineError(Exception):
    """Custom exception for query engine errors."""
    pass

class ReadOnlyViolationError(Exception):
    """Custom exception for read-only violations."""
    pass


def _initialize_connection() -> duckdb.Connection:
    """Initialize DuckDB connection with read-only mode."""
    global _connection
    if _connection is None:
        try:
            # Connect to DuckDB in read-only mode
            _connection = duckdb.connect(database=':memory:', read_only=True)
            logger.info("DuckDB connection initialized in read-only mode")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB connection: {e}")
            raise QueryEngineError(f"Failed to initialize DuckDB: {e}")
    return _connection


def _validate_sql(sql: str) -> None:
    """Validate SQL to ensure it's read-only."""
    # Convert to uppercase for case-insensitive comparison
    upper_sql = sql.strip().upper()
    
    # List of write operations to block
    write_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
        'RENAME', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT'
    ]
    
    # Check for any write keywords
    for keyword in write_keywords:
        if keyword in upper_sql:
            logger.warning(f"Write operation detected in SQL: {keyword}")
            raise ReadOnlyViolationError(f"Write operations are not allowed: {keyword}")


def query_logs(sql: str) -> Dict[str, Any]:
    """Execute a SQL query on telemetry logs and return results as JSON."""
    try:
        # Validate SQL
        _validate_sql(sql)
        
        # Initialize connection if needed
        conn = _initialize_connection()
        
        # Execute query
        result = conn.execute(sql)
        
        # Fetch results
        rows = result.fetchall()
        
        # Get column names
        columns = result.description
        
        # Convert to list of dictionaries
        data = []
        for row in rows:
            row_dict = dict(zip([col[0] for col in columns], row))
            data.append(row_dict)
        
        logger.info(f"Query executed successfully: {sql}")
        return {
            "success": True,
            "data": data,
            "count": len(data)
        }
        
    except ReadOnlyViolationError:
        logger.error("Read-only violation detected")
        raise
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise QueryEngineError(f"Query execution failed: {e}")


def export_to_parquet(data: Dict[str, Any], filename: str) -> None:
    """Export telemetry data to Parquet format."""
    try:
        import pandas as pd
        
        # Convert data to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Export to Parquet
        df.to_parquet(filename)
        
        logger.info(f"Data exported to Parquet: {filename}")
    except Exception as e:
        logger.error(f"Parquet export failed: {e}")
        raise QueryEngineError(f"Parquet export failed: {e}")


def get_telemetry_schema() -> Dict[str, Any]:
    """Get the schema of the telemetry data."""
    try:
        conn = _initialize_connection()
        result = conn.execute("PRAGMA table_info(logs)")
        columns = result.fetchall()
        
        schema = {
            "columns": [
                {"name": col[1], "type": col[2]} for col in columns
            ]
        }
        
        return schema
    except Exception as e:
        logger.error(f"Failed to get schema: {e}")
        raise QueryEngineError(f"Failed to get schema: {e}")