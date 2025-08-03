"""
Database connection management for retro-peft adapters.
"""

import os
import sqlite3
import json
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
from datetime import datetime
import threading
from dataclasses import dataclass, asdict


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str = "sqlite"  # sqlite, postgresql, mysql
    db_path: str = "./data/retro_peft.db"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "retro_peft"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class DatabaseManager:
    """
    Database connection manager with support for SQLite and PostgreSQL.
    
    Handles connection pooling, transactions, and schema management.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._connection_pool = {}
        self._lock = threading.Lock()
        self._initialized = False
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        if self._initialized:
            return
        
        if self.config.db_type == "sqlite":
            self._initialize_sqlite()
        elif self.config.db_type == "postgresql":
            self._initialize_postgresql()
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
        
        self._initialized = True
    
    def _initialize_sqlite(self):
        """Initialize SQLite database"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create adapter configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adapter_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    adapter_type TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create training runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adapter_config_id INTEGER NOT NULL,
                    run_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    hyperparameters_json TEXT NOT NULL,
                    metrics_json TEXT DEFAULT '{}',
                    checkpoint_path TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (adapter_config_id) REFERENCES adapter_configs (id)
                )
            """)
            
            # Create retrieval indices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    backend_type TEXT NOT NULL,
                    index_path TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    num_documents INTEGER NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_run_id INTEGER,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER,
                    epoch INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (training_run_id) REFERENCES training_runs (id)
                )
            """)
            
            # Create model versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adapter_config_id INTEGER NOT NULL,
                    version TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    performance_summary_json TEXT DEFAULT '{}',
                    tags_json TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (adapter_config_id) REFERENCES adapter_configs (id),
                    UNIQUE(adapter_config_id, version)
                )
            """)
            
            # Create indices for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_adapter ON training_runs(adapter_config_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_run ON performance_metrics(training_run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_adapter ON model_versions(adapter_config_id)")
            
            conn.commit()
    
    def _initialize_postgresql(self):
        """Initialize PostgreSQL database"""
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
        except ImportError:
            raise ImportError("PostgreSQL support requires psycopg2: pip install psycopg2-binary")
        
        # Create connection pool
        self._pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=self.config.pool_size,
            host=self.config.host,
            port=self.config.port,
            user=self.config.username,
            password=self.config.password,
            database=self.config.database
        )
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables with PostgreSQL syntax
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adapter_configs (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    adapter_type VARCHAR(100) NOT NULL,
                    base_model VARCHAR(255) NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id SERIAL PRIMARY KEY,
                    adapter_config_id INTEGER NOT NULL,
                    run_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    hyperparameters_json TEXT NOT NULL,
                    metrics_json TEXT DEFAULT '{}',
                    checkpoint_path TEXT,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (adapter_config_id) REFERENCES adapter_configs (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_indices (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    backend_type VARCHAR(50) NOT NULL,
                    index_path TEXT NOT NULL,
                    embedding_model VARCHAR(255) NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    num_documents INTEGER NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    training_run_id INTEGER,
                    metric_type VARCHAR(100) NOT NULL,
                    metric_name VARCHAR(255) NOT NULL,
                    metric_value REAL NOT NULL,
                    step INTEGER,
                    epoch INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (training_run_id) REFERENCES training_runs (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    adapter_config_id INTEGER NOT NULL,
                    version VARCHAR(100) NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    performance_summary_json TEXT DEFAULT '{}',
                    tags_json TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (adapter_config_id) REFERENCES adapter_configs (id),
                    UNIQUE(adapter_config_id, version)
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_adapter ON training_runs(adapter_config_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_run ON performance_metrics(training_run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_adapter ON model_versions(adapter_config_id)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        if self.config.db_type == "sqlite":
            conn = sqlite3.connect(
                self.config.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()
        
        elif self.config.db_type == "postgresql":
            conn = self._pool.getconn()
            try:
                yield conn
            finally:
                self._pool.putconn(conn)
        
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager"""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Union[tuple, dict]] = None,
        fetch: str = "none"  # none, one, all
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Execute SQL query with optional parameter binding.
        
        Args:
            query: SQL query string
            params: Query parameters (tuple or dict)
            fetch: Result fetch mode ("none", "one", "all")
            
        Returns:
            Query results based on fetch mode
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch == "one":
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch == "all":
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                conn.commit()
                return cursor.lastrowid if cursor.lastrowid else None
    
    def create_adapter_config(
        self,
        name: str,
        adapter_type: str,
        base_model: str,
        config: Dict[str, Any]
    ) -> int:
        """Create new adapter configuration"""
        query = """
            INSERT INTO adapter_configs (name, adapter_type, base_model, config_json)
            VALUES (?, ?, ?, ?)
        """
        params = (name, adapter_type, base_model, json.dumps(config))
        return self.execute_query(query, params)
    
    def get_adapter_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get adapter configuration by name"""
        query = "SELECT * FROM adapter_configs WHERE name = ?"
        result = self.execute_query(query, (name,), fetch="one")
        if result:
            result['config'] = json.loads(result['config_json'])
            del result['config_json']
        return result
    
    def create_training_run(
        self,
        adapter_config_id: int,
        run_name: str,
        hyperparameters: Dict[str, Any]
    ) -> int:
        """Create new training run"""
        query = """
            INSERT INTO training_runs (adapter_config_id, run_name, status, hyperparameters_json)
            VALUES (?, ?, 'started', ?)
        """
        params = (adapter_config_id, run_name, json.dumps(hyperparameters))
        return self.execute_query(query, params)
    
    def update_training_run_status(
        self,
        run_id: int,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None
    ):
        """Update training run status and metrics"""
        if status == "completed":
            query = """
                UPDATE training_runs 
                SET status = ?, metrics_json = ?, checkpoint_path = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            params = (status, json.dumps(metrics or {}), checkpoint_path, run_id)
        else:
            query = "UPDATE training_runs SET status = ? WHERE id = ?"
            params = (status, run_id)
        
        self.execute_query(query, params)
    
    def log_performance_metric(
        self,
        training_run_id: Optional[int],
        metric_type: str,
        metric_name: str,
        metric_value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log performance metric"""
        query = """
            INSERT INTO performance_metrics 
            (training_run_id, metric_type, metric_name, metric_value, step, epoch, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            training_run_id, metric_type, metric_name, metric_value,
            step, epoch, json.dumps(metadata or {})
        )
        self.execute_query(query, params)
    
    def create_retrieval_index(
        self,
        name: str,
        backend_type: str,
        index_path: str,
        embedding_model: str,
        embedding_dim: int,
        num_documents: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create retrieval index record"""
        query = """
            INSERT INTO retrieval_indices 
            (name, backend_type, index_path, embedding_model, embedding_dim, num_documents, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            name, backend_type, index_path, embedding_model,
            embedding_dim, num_documents, json.dumps(metadata or {})
        )
        return self.execute_query(query, params)
    
    def get_retrieval_index(self, name: str) -> Optional[Dict[str, Any]]:
        """Get retrieval index by name"""
        query = "SELECT * FROM retrieval_indices WHERE name = ?"
        result = self.execute_query(query, (name,), fetch="one")
        if result:
            result['metadata'] = json.loads(result['metadata_json'])
            del result['metadata_json']
        return result
    
    def save_model_version(
        self,
        adapter_config_id: int,
        version: str,
        checkpoint_path: str,
        performance_summary: Dict[str, Any],
        tags: List[str] = None
    ) -> int:
        """Save model version"""
        query = """
            INSERT INTO model_versions 
            (adapter_config_id, version, checkpoint_path, performance_summary_json, tags_json)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (
            adapter_config_id, version, checkpoint_path,
            json.dumps(performance_summary), json.dumps(tags or [])
        )
        return self.execute_query(query, params)
    
    def get_training_history(self, adapter_config_id: int) -> List[Dict[str, Any]]:
        """Get training history for adapter"""
        query = """
            SELECT * FROM training_runs 
            WHERE adapter_config_id = ? 
            ORDER BY started_at DESC
        """
        results = self.execute_query(query, (adapter_config_id,), fetch="all")
        
        for result in results:
            result['hyperparameters'] = json.loads(result['hyperparameters_json'])
            result['metrics'] = json.loads(result['metrics_json'])
            del result['hyperparameters_json']
            del result['metrics_json']
        
        return results
    
    def get_performance_metrics(
        self,
        training_run_id: int,
        metric_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get performance metrics for training run"""
        if metric_type:
            query = """
                SELECT * FROM performance_metrics 
                WHERE training_run_id = ? AND metric_type = ?
                ORDER BY timestamp
            """
            params = (training_run_id, metric_type)
        else:
            query = """
                SELECT * FROM performance_metrics 
                WHERE training_run_id = ?
                ORDER BY timestamp
            """
            params = (training_run_id,)
        
        results = self.execute_query(query, params, fetch="all")
        
        for result in results:
            result['metadata'] = json.loads(result['metadata_json'])
            del result['metadata_json']
        
        return results
    
    def cleanup_old_metrics(self, days: int = 30):
        """Clean up old performance metrics"""
        if self.config.db_type == "sqlite":
            query = """
                DELETE FROM performance_metrics 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days)
        else:
            query = """
                DELETE FROM performance_metrics 
                WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '{} days'
            """.format(days)
        
        self.execute_query(query)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        # Count records in each table
        tables = ['adapter_configs', 'training_runs', 'retrieval_indices', 'performance_metrics', 'model_versions']
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query, fetch="one")
            stats[f"{table}_count"] = result['count'] if result else 0
        
        # Get database size (SQLite only)
        if self.config.db_type == "sqlite":
            stats['database_size_bytes'] = os.path.getsize(self.config.db_path)
        
        return stats