import duckdb
import logging
from typing import Dict, Any, List
from pathlib import Path
import re
import sqlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngineError(Exception):
    """Custom exception for query engine errors."""
    pass

class ReadOnlyViolationError(Exception):
    """Custom exception for read-only violations."""
    pass

class QueryEngine:
    """DuckDB-based query engine for observability."""
    
    def __init__(self, data_directory: str):
        """Initialize the QueryEngine with a data directory."""
        self.data_directory = Path(data_directory)
        self._connection = None
        
    def _initialize_connection(self) -> Any:
        """Initialize DuckDB connection and register Parquet files."""
        if self._connection is None:
            try:
                # Connect to DuckDB in read-only mode using a memory database
                self._connection = duckdb.connect(database=':memory:', read_only=False)
                logger.info("DuckDB connection initialized (Memory Mode)")
                
                # Scan data directory for Parquet files and register them as views
                self._register_parquet_files()
                
            except Exception as e:
                logger.error(f"Failed to initialize DuckDB connection: {e}")
                raise QueryEngineError(f"Failed to initialize DuckDB: {e}")
        return self._connection

    def _register_parquet_files(self) -> None:
        """Scan the data directory and register all .parquet files as tables/views."""
        if not self.data_directory.exists():
            logger.warning(f"Data directory {self.data_directory} does not exist. No files registered.")
            return

        parquet_files = list(self.data_directory.glob("**/*.parquet"))
        if not parquet_files:
            logger.info(f"No parquet files found in {self.data_directory}")
            return

        for p_file in parquet_files:
            table_name = p_file.stem
            try:
                # Register the parquet file as a view
                self._connection.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{p_file}')")
                logger.info(f"Registered Parquet file: {p_file} as table '{table_name}'")
            except Exception as e:
                logger.error(f"Failed to register parquet file {p_file}: {e}")

    def _validate_sql(self, sql: str) -> None:
        """Validate SQL to ensure it's read-only using multiple techniques."""
        if not sql or not isinstance(sql, str):
            raise ReadOnlyViolationError("Invalid SQL query")
        
        # Convert to uppercase for case-insensitive comparison
        upper_sql = sql.strip().upper()
        
        # List of write operations to block
        write_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
            'RENAME', 'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT',
            'WITH', 'EXECUTE', 'CALL', 'MERGE', 'REPLACE'
        ]
        
        # Check for any write keywords
        for keyword in write_keywords:
            # Use regex to match whole words only to avoid false positives
            if re.search(rf'\b{keyword}\b', upper_sql):
                logger.warning(f"Write operation detected in SQL: {keyword}")
                raise ReadOnlyViolationError(f"Write operations are not allowed: {keyword}")
        
        # Use sqlparse to parse and validate SQL structure
        try:
            parsed = sqlparse.parse(sql)[0]
            # Check if the statement type is a write operation
            statement_type = parsed.get_type()
            if statement_type in ('INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE'):
                logger.warning(f"Write operation detected by sqlparse: {statement_type}")
                raise ReadOnlyViolationError(f"Write operations are not allowed: {statement_type}")
        except Exception as e:
            logger.warning(f"SQL parsing failed: {e}")
            # If parsing fails, we still check keywords, but don't fail completely
            pass

    def query_logs(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query on telemetry logs and return results as JSON."""
        try:
            # Validate SQL
            self._validate_sql(sql)
            
            # Initialize connection if needed
            conn = self._initialize_connection()
            
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

    def export_to_parquet(self, data: Dict[str, Any], filename: str) -> None:
        """Export telemetry data to Parquet format."""
        try:
            # Try to import polars first (preferred)
            try:
                import polars as pl
                df = pl.DataFrame(data["data"])
                df.write_parquet(filename)
                logger.info(f"Data exported to Parquet using polars: {filename}")
            except ImportError:
                # Fallback to pandas
                import pandas as pd
                df = pd.DataFrame(data["data"])
                df.to_parquet(filename)
                logger.info(f"Data exported to Parquet using pandas: {filename}")
                
        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            raise QueryEngineError(f"Parquet export failed: {e}")

    def get_telemetry_schema(self) -> Dict[str, Any]:
        """Get the schema of the telemetry data."""
        try:
            conn = self._initialize_connection()
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

    def get_table_names(self) -> List[str]:
        """Get list of table/view names in the database."""
        try:
            conn = self._initialize_connection()
            result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'")
            tables = result.fetchall()
            return [table[0] for table in tables]
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            raise QueryEngineError(f"Failed to get table names: {e}")
