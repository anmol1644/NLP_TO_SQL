import asyncio
import time
import threading
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import pool, OperationalError
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """
    Manages database connection pools for multiple workspaces with automatic cleanup
    """
    
    def __init__(self, 
                 min_connections: int = 1, 
                 max_connections: int = 10,
                 inactivity_timeout: int = 600):  # 10 minutes in seconds
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.inactivity_timeout = inactivity_timeout
        
        # Dictionary to store connection pools for each workspace
        # Format: {workspace_id: {'pool': pool_object, 'last_used': timestamp, 'db_config': config, 'db_analyzer': analyzer, 'schema_analyzed': bool}}
        self.workspace_pools: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Background cleanup task
        self._cleanup_task = None
        self._stop_cleanup = False
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        def cleanup_worker():
            while not self._stop_cleanup:
                try:
                    self._cleanup_inactive_connections()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in cleanup worker: {e}")
        
        self._cleanup_task = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_task.start()
        logger.info("Database connection cleanup task started")
    
    def _cleanup_inactive_connections(self):
        """Clean up inactive connection pools"""
        current_time = time.time()
        inactive_workspaces = []
        
        with self._lock:
            for workspace_id, pool_info in self.workspace_pools.items():
                last_used = pool_info['last_used']
                if current_time - last_used > self.inactivity_timeout:
                    inactive_workspaces.append(workspace_id)
            
            # Close inactive pools
            for workspace_id in inactive_workspaces:
                try:
                    pool_info = self.workspace_pools[workspace_id]
                    pool_obj = pool_info['pool']
                    pool_obj.closeall()
                    del self.workspace_pools[workspace_id]
                    logger.info(f"Closed inactive connection pool for workspace {workspace_id}")
                except Exception as e:
                    logger.error(f"Error closing pool for workspace {workspace_id}: {e}")
    
    def create_workspace_pool(self, workspace_id: str, db_config: Dict[str, Any], analyze_schema: bool = True) -> bool:
        """
        Create a connection pool for a workspace
        
        Args:
            workspace_id: Unique identifier for the workspace
            db_config: Database configuration dict with keys: host, port, db_name, username, password
            analyze_schema: Whether to analyze the database schema immediately
            
        Returns:
            bool: True if pool created successfully, False otherwise
        """
        with self._lock:
            try:
                # Close existing pool if it exists
                if workspace_id in self.workspace_pools:
                    self.close_workspace_pool(workspace_id)
                
                # Create new connection pool
                connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=self.min_connections,
                    maxconn=self.max_connections,
                    host=db_config['host'],
                    port=db_config['port'],
                    database=db_config['db_name'],
                    user=db_config['username'],
                    password=db_config['password'],
                    connect_timeout=10
                )
                
                # Test the pool by getting a connection
                test_conn = connection_pool.getconn()
                if test_conn:
                    cursor = test_conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    cursor.close()
                    connection_pool.putconn(test_conn)
                
                # Create database analyzer with connection manager
                from db_analyzer import DatabaseAnalyzer
                db_analyzer = DatabaseAnalyzer(
                    db_config['db_name'],
                    db_config['username'],
                    db_config['password'],
                    db_config['host'],
                    db_config['port'],
                    connection_manager=self,
                    workspace_id=workspace_id
                )
                
                # Store pool info
                self.workspace_pools[workspace_id] = {
                    'pool': connection_pool,
                    'last_used': time.time(),
                    'db_config': db_config.copy(),
                    'db_analyzer': db_analyzer,
                    'schema_analyzed': False
                }
                
                # Analyze schema if requested
                if analyze_schema:
                    try:
                        logger.info(f"Analyzing database schema for workspace {workspace_id}")
                        schema_info = db_analyzer.analyze_schema()
                        self.workspace_pools[workspace_id]['schema_analyzed'] = True
                        self.workspace_pools[workspace_id]['schema_info'] = schema_info
                        logger.info(f"Schema analysis completed for workspace {workspace_id}")
                    except Exception as e:
                        logger.error(f"Error analyzing schema for workspace {workspace_id}: {e}")
                        # Don't fail the connection creation if schema analysis fails
                
                logger.info(f"Created connection pool for workspace {workspace_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create connection pool for workspace {workspace_id}: {e}")
                return False
    
    @contextmanager
    def get_connection(self, workspace_id: str):
        """
        Get a database connection from the workspace pool using context manager
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Yields:
            psycopg2.connection: Database connection
            
        Raises:
            Exception: If workspace pool doesn't exist or connection fails
        """
        if workspace_id not in self.workspace_pools:
            raise Exception(f"No connection pool found for workspace {workspace_id}")
        
        connection = None
        try:
            with self._lock:
                pool_info = self.workspace_pools[workspace_id]
                pool_obj = pool_info['pool']
                pool_info['last_used'] = time.time()  # Update last used time
            
            # Get connection from pool
            connection = pool_obj.getconn()
            
            if connection is None:
                raise Exception(f"Failed to get connection from pool for workspace {workspace_id}")
            
            # Test connection
            if connection.closed:
                raise Exception("Connection is closed")
            
            yield connection
            
        except Exception as e:
            logger.error(f"Error with database connection for workspace {workspace_id}: {e}")
            # If connection is bad, try to close it
            if connection and not connection.closed:
                try:
                    connection.close()
                except:
                    pass
            raise
        finally:
            # Return connection to pool
            if connection and workspace_id in self.workspace_pools:
                try:
                    pool_obj = self.workspace_pools[workspace_id]['pool']
                    pool_obj.putconn(connection)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
    
    def close_workspace_pool(self, workspace_id: str) -> bool:
        """
        Close connection pool for a workspace
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Returns:
            bool: True if closed successfully, False otherwise
        """
        with self._lock:
            if workspace_id in self.workspace_pools:
                try:
                    pool_obj = self.workspace_pools[workspace_id]['pool']
                    pool_obj.closeall()
                    del self.workspace_pools[workspace_id]
                    logger.info(f"Closed connection pool for workspace {workspace_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error closing pool for workspace {workspace_id}: {e}")
                    return False
            return True  # Already closed
    
    def get_database_analyzer(self, workspace_id: str):
        """
        Get the database analyzer for a workspace
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Returns:
            DatabaseAnalyzer instance or None if workspace not found
        """
        with self._lock:
            if workspace_id in self.workspace_pools:
                pool_info = self.workspace_pools[workspace_id]
                pool_info['last_used'] = time.time()  # Update last used time
                return pool_info['db_analyzer']
            return None
    
    def is_schema_analyzed(self, workspace_id: str) -> bool:
        """
        Check if schema has been analyzed for a workspace
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Returns:
            True if schema has been analyzed, False otherwise
        """
        with self._lock:
            if workspace_id in self.workspace_pools:
                return self.workspace_pools[workspace_id].get('schema_analyzed', False)
            return False
    
    def ensure_schema_analyzed(self, workspace_id: str) -> bool:
        """
        Ensure schema is analyzed for a workspace, analyze if not already done
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Returns:
            True if schema is analyzed (or was successfully analyzed), False otherwise
        """
        with self._lock:
            if workspace_id not in self.workspace_pools:
                return False
            
            pool_info = self.workspace_pools[workspace_id]
            
            # If already analyzed, return True
            if pool_info.get('schema_analyzed', False):
                return True
            
            # Try to analyze schema
            try:
                logger.info(f"Analyzing database schema for workspace {workspace_id}")
                db_analyzer = pool_info['db_analyzer']
                schema_info = db_analyzer.analyze_schema()
                pool_info['schema_analyzed'] = True
                pool_info['schema_info'] = schema_info
                pool_info['last_used'] = time.time()  # Update last used time
                logger.info(f"Schema analysis completed for workspace {workspace_id}")
                return True
            except Exception as e:
                logger.error(f"Error analyzing schema for workspace {workspace_id}: {e}")
                return False
    
    def get_workspace_status(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a workspace connection pool
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Returns:
            Dict with status info or None if workspace not found
        """
        with self._lock:
            if workspace_id not in self.workspace_pools:
                return None
            
            pool_info = self.workspace_pools[workspace_id]
            pool_obj = pool_info['pool']
            
            # Get table count from schema info if available
            table_count = 0
            if pool_info.get('schema_analyzed', False) and 'schema_info' in pool_info:
                schema_info = pool_info['schema_info']
                if schema_info and 'tables' in schema_info:
                    table_count = len(schema_info['tables'])
            
            return {
                'workspace_id': workspace_id,
                'status': 'connected',
                'last_used': datetime.fromtimestamp(pool_info['last_used']).isoformat(),
                'min_connections': pool_obj.minconn,
                'max_connections': pool_obj.maxconn,
                'schema_analyzed': pool_info.get('schema_analyzed', False),
                'database_info': {
                    'name': pool_info['db_config']['db_name'],
                    'host': pool_info['db_config']['host'],
                    'port': pool_info['db_config']['port'],
                    'table_count': table_count
                }
            }
    
    def get_all_workspace_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all active workspace pools"""
        with self._lock:
            status = {}
            for workspace_id in self.workspace_pools.keys():
                status[workspace_id] = self.get_workspace_status(workspace_id)
            return status
    
    def refresh_connection_pool(self, workspace_id: str) -> bool:
        """
        Refresh a connection pool (close and recreate)
        
        Args:
            workspace_id: Unique identifier for the workspace
            
        Returns:
            bool: True if refreshed successfully, False otherwise
        """
        with self._lock:
            if workspace_id not in self.workspace_pools:
                return False
            
            # Get the existing config
            db_config = self.workspace_pools[workspace_id]['db_config']
            
            # Close existing pool
            self.close_workspace_pool(workspace_id)
            
            # Create new pool
            return self.create_workspace_pool(workspace_id, db_config)
    
    def shutdown(self):
        """Shutdown the connection manager and close all pools"""
        logger.info("Shutting down database connection manager")
        
        # Stop cleanup task
        self._stop_cleanup = True
        if self._cleanup_task and self._cleanup_task.is_alive():
            self._cleanup_task.join(timeout=5)
        
        # Close all pools
        with self._lock:
            workspace_ids = list(self.workspace_pools.keys())
            for workspace_id in workspace_ids:
                self.close_workspace_pool(workspace_id)
        
        logger.info("Database connection manager shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except:
            pass


# Global connection manager instance
db_connection_manager = DatabaseConnectionManager()

# Cleanup function for graceful shutdown
def cleanup_db_connections():
    """Cleanup function to be called on application shutdown"""
    global db_connection_manager
    if db_connection_manager:
        db_connection_manager.shutdown() 