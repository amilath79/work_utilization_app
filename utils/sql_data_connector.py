# utils/sql_data_connector.py
"""
SQL Server data extraction and parquet storage utilities.
"""
import pandas as pd
import pyodbc
import os
import logging
import traceback
from datetime import datetime
from config import DATA_DIR, CACHE_TTL, CHUNK_SIZE

# Configure logger
logger = logging.getLogger(__name__)

@pd.api.extensions.register_dataframe_accessor("sql_data")
class SQLDataConnector:
    """
    Utility class to extract data from SQL Server and store it as parquet files.
    Implemented as a pandas accessor for better integration with existing code.
    """
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
    @staticmethod
    def connect_to_sql(server, database, username=None, password=None, trusted_connection=True):
        """
        Create a connection to SQL Server
        
        Parameters:
        -----------
        server : str
            SQL Server name
        database : str
            Database name
        username : str, optional
            SQL Server username (if not using trusted connection)
        password : str, optional
            SQL Server password (if not using trusted connection)
        trusted_connection : bool
            Whether to use Windows authentication
            
        Returns:
        --------
        pyodbc.Connection or None
            Connection object or None if connection fails
        """
        try:
            # Generate connection string
            if trusted_connection:
                conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
            else:
                conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
            
            # Create connection
            conn = pyodbc.connect(conn_str)
            return conn
        
        except Exception as e:
            logger.error(f"Error connecting to SQL Server: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def extract_to_df(query, conn, chunk_size=CHUNK_SIZE):
        """
        Extract data from SQL Server and return as DataFrame
        
        Parameters:
        -----------
        query : str
            SQL query to execute
        conn : pyodbc.Connection
            Connection to SQL Server
        chunk_size : int
            Number of rows to fetch in each chunk
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing the query results or None if error occurs
        """
        try:
            # Use pandas read_sql with chunksize for memory efficiency
            chunks = []
            for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
                chunks.append(chunk)
                logger.info(f"Fetched chunk of {len(chunk)} rows")
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Total rows fetched: {len(df)}")
                return df
            else:
                logger.warning("No data returned from query")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def save_to_parquet(self, file_path, partition_cols=None, compression='snappy'):
        """
        Save DataFrame to parquet file, optionally with partitioning
        
        Parameters:
        -----------
        file_path : str
            Path to save the parquet file
        partition_cols : list, optional
            Columns to partition by
        compression : str
            Compression codec to use
            
        Returns:
        --------
        bool
            True if save is successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if partition_cols:
                # Save with partitioning for more efficient queries later
                self._obj.to_parquet(
                    file_path, 
                    engine='pyarrow',
                    compression=compression,
                    partition_cols=partition_cols
                )
            else:
                self._obj.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression=compression
                )
                
            logger.info(f"Data saved to {file_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
            logger.error(traceback.format_exc())
            return False


def extract_sql_data(server, database, query, username=None, password=None, 
                     trusted_connection=True, chunk_size=CHUNK_SIZE):
    """
    Extract data from SQL Server
    
    Parameters:
    -----------
    server : str
        SQL Server name
    database : str
        Database name
    query : str
        SQL query to execute
    username : str, optional
        SQL Server username (if not using trusted connection)
    password : str, optional
        SQL Server password (if not using trusted connection)
    trusted_connection : bool
        Whether to use Windows authentication
    chunk_size : int
        Number of rows to fetch in each chunk
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing the query results or None if error occurs
    """
    try:
        logger.info(f"Connecting to {server}/{database}")
        conn = SQLDataConnector.connect_to_sql(
            server=server,
            database=database,
            username=username,
            password=password,
            trusted_connection=trusted_connection
        )
        
        if conn is None:
            return None
        
        # Extract data to DataFrame
        df = SQLDataConnector.extract_to_df(query, conn, chunk_size)
        
        # Close connection
        conn.close()
        
        return df
        
    except Exception as e:
        logger.error(f"Error in extract_sql_data: {str(e)}")
        logger.error(traceback.format_exc())
        return None