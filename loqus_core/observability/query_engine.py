"""
DuckDB Query Engine for Agentic Observability

SAFETY: This engine is READ-ONLY. All write attempts are blocked.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import sqlparse

logger = logging.getLogger(__name__)


class QueryEngineError(Exception):
    """Base exception for query engine errors."""
    pass

class ReadOnlyViolationError(QueryEngineError):
    """Raised when a write operation is attempted."""
    pass

class QueryEngine:
    """
    Read-only DuckDB query engine for observability.
    
    Example:
        engine = QueryEngine("/path/to/telemetry")
        result = engine.query_logs("SELECT * FROM sensors LIMIT 10")
    """
    
    # SQL keywords that indicate write operations
    WRITE_KEYWORDS = frozenset([
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'REPLACE', 'MERGE', 'UPSERT', 'WITH'
    ])
    
    def __init__(self, data_dir: str, memory_limit: str = "256MB"):
        """
        Initialize query engine.
        
        Args:
            data_dir: Directory containing Parquet files
            memory_limit: DuckDB memory limit (default 256MB)
        """
        self.data_dir = Path(data_dir)
        self.memory_limit = memory_limit
        
        try:
            # Create connection
            self.conn = duckdb.connect()
            self.conn.execute(f"SET memory_limit='{memory_limit}'")
            
            # Register Parquet files as tables
            self._register_tables()
            
            logger.info(f"QueryEngine initialized with data_dir={data_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize QueryEngine: {e}")
            raise QueryEngineError(f"Failed to initialize QueryEngine: {e}")
    
    def _register_tables(self):
        """Auto-register Parquet files as tables."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return
        
        for parquet_file in self.data_dir.glob("*.parquet"):
            table_name = parquet_file.stem  # filename without extension
            try:
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {table_name} AS 
                    SELECT * FROM read_parquet('{parquet_file}')
                """)
                logger.debug(f"Registered table: {table_name}")
            except Exception as e:
                logger.error(f"Failed to register table {table_name}: {e}")
                # Continue with other tables
    
    def _validate_read_only(self, sql: str):
        """
        Validate that SQL is read-only.
        
        Args:
            sql: SQL query string
            
        Raises:
            ReadOnlyViolationError: If write operation detected
        """
        # Normalize SQL for checking
        sql_normalized = sql.strip()
        if not sql_normalized:
            return  # Empty query is allowed
        
        # Check for write operations using multiple approaches
        sql_upper = sql_normalized.upper()
        
        # Check for direct write keywords
        for keyword in self.WRITE_KEYWORDS:
            # Check if keyword appears at the beginning or after whitespace
            if sql_upper.startswith(keyword + ' ') or keyword + ' ' in sql_upper:
                logger.error(f"Write operation blocked: {sql[:50]}...")
                raise ReadOnlyViolationError(
                    f"Write operations are not allowed. Detected: {keyword}"
                )
        
        # Check for specific patterns that indicate write operations
        write_patterns = [
            'INSERT INTO', 'UPDATE SET', 'DELETE FROM', 'DROP TABLE',
            'CREATE TABLE', 'ALTER TABLE', 'TRUNCATE TABLE', 'REPLACE TABLE',
            'MERGE INTO', 'UPSERT INTO'
        ]
        
        for pattern in write_patterns:
            if pattern in sql_upper:
                logger.error(f"Write operation blocked: {sql[:50]}...")
                raise ReadOnlyViolationError(
                    f"Write operations are not allowed. Detected pattern: {pattern}"
                )
        
        # Additional check for common SQL injection patterns
        injection_patterns = [
            "';", "--", "/*", "*/", "UNION SELECT"
        ]
        
        for pattern in injection_patterns:
            if pattern in sql_upper:
                logger.error(f"Potential injection attempt blocked: {sql[:50]}...")
                raise ReadOnlyViolationError(
                    f"Potential injection attempt blocked: {pattern}"
                )
    
    def query_logs(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query.
        
        Args:
            sql: SQL SELECT query
            
        Returns:
            List of dictionaries with query results
            
        Raises:
            ReadOnlyViolationError: If write operation attempted
            QueryEngineError: For other query errors
        """
        # Validate read-only
        self._validate_read_only(sql)
        
        try:
            # Execute the query
            result = self.conn.execute(sql).fetchall()
            columns = [desc[0] for desc in self.conn.description]
            
            return [dict(zip(columns, row)) for row in result]
            
        except duckdb.Error as e:
            logger.error(f"Query failed: {e}")
            raise QueryEngineError(f"Query failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise QueryEngineError(f"Unexpected error: {e}")
    
    def get_table_names(self) -> List[str]:
        """List available tables."""
        try:
            result = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("QueryEngine connection closed")