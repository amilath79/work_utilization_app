# utils/sql_data_connector.py
"""
SQL Server data extraction and parquet storage utilities.
Also includes prediction database operations.
"""
import pandas as pd
import pyodbc
import os
import logging
import traceback
from datetime import datetime
from config import DATA_DIR, CACHE_TTL, CHUNK_SIZE, SQL_SERVER, SQL_TRUSTED_CONNECTION, SQL_DATABASE_LIVE, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD

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
    

def load_demand_forecast_data(server=SQL_SERVER, database=SQL_DATABASE, trusted_connection=SQL_TRUSTED_CONNECTION, username=None, password=None, chunk_size=CHUNK_SIZE):
    logger.info("Loading Data")
    """
    Load demand forecast data from SQL Server.

    Parameters:
    -----------
    server : str
        SQL Server name.
    database : str
        Database name.
    trusted_connection : bool
        Whether to use Windows authentication.
    username : str, optional
        SQL Server username (if not using trusted connection).
    password : str, optional
        SQL Server password (if not using trusted connection).
    chunk_size : int
        Number of rows to fetch in each chunk.

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing the forecast data or None if an error occurs.
    """
    query = """
        DECLARE @StartDate DATE = CAST(GETDATE() AS DATE);
        DECLARE @EndDate DATE = CAST(GETDATE() + 14 AS DATE);            
        SELECT 
            Date AS PlanDate, 
            (SELECT KPIValue 
             FROM KPIData k 
             WHERE k.PunchCode = p.PunchCode AND k.KPIDate = p.Date) * Hours AS Quantity,
            PunchCode Punchcode
        FROM PredictionData p
        WHERE Date BETWEEN @StartDate AND @EndDate;
    """

    try:
        logger.info(f"Connecting to {server}/{database} to load forecast data.")
        
        conn = SQLDataConnector.connect_to_sql(server=server, database=database, username=username, password=password, trusted_connection=trusted_connection)

        if conn is None:
            logger.error(f"Failed to connect to database {database}")
            return None

        forecast_df = SQLDataConnector.extract_to_df(query, conn, chunk_size)
        conn.close()
        if forecast_df is not None:
            logger.info(f"Successfully loaded forecast data. Shape: {forecast_df.shape}")

        return forecast_df

    except Exception as e:
        logger.error(f"Error loading forecast data: {e}")
        logger.error(traceback.format_exc())
        return None


# ================ PREDICTION DATABASE FUNCTIONS ================

def save_predictions_to_db(predictions_dict, hours_dict, username, server=None, database=None):
    """
    Save predictions to the database using stored procedure
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary of predictions with dates as keys and worktype predictions as values
    hours_dict : dict
        Dictionary of hour predictions with dates as keys and worktype hours as values
    username : str
        Username who made the prediction
    server : str, optional
        SQL Server name (defaults to configured SQL_SERVER)
    database : str, optional
        Database name (defaults to configured SQL_DATABASE)
        
    Returns:
    --------
    bool
        True if successful, False if error occurred
    """
    try:
        # Get configured values if not provided
        server = server or SQL_SERVER
        database = database or SQL_DATABASE
        
        # Add comprehensive logging at the start
        logger.info(f"Starting save_predictions_to_db with username: {username}")
        logger.info(f"Connecting to {server}/{database}")
        logger.info(f"Number of dates in predictions: {len(predictions_dict)}")
        prediction_count = sum(len(work_types) for work_types in predictions_dict.values())
        logger.info(f"Total predictions to save: {prediction_count}")
        
        # Check if predictions_dict is empty
        if not predictions_dict:
            logger.error("No predictions to save. predictions_dict is empty.")
            return False
            
        # Get database connection
        conn = SQLDataConnector.connect_to_sql(
            server=server,
            database=database,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if conn is None:
            logger.error(f"Failed to connect to database {database} on server {server}")
            return False
        
        logger.info("Database connection successful")
        cursor = conn.cursor()
        
        # Count for success tracking
        success_count = 0
        error_count = 0
        
        # Process each prediction
        for date, work_types in predictions_dict.items():
            logger.info(f"Processing predictions for date: {date.strftime('%Y-%m-%d')}")
            
            for work_type, man_value in work_types.items():
                try:
                    # Convert work_type to integer for database storage
                    logger.info(f"Processing work type: {work_type}, Man Power: {man_value}")
                    
                    try:
                        punch_code_int = int(work_type)
                    except ValueError:
                        logger.warning(f"Skipping work type {work_type} - cannot convert to integer")
                        error_count += 1
                        continue
                    
                    # Ensure man_value is a valid float
                    if pd.isna(man_value) or man_value is None:
                        man_value = 0.0
                    else:
                        man_value = float(man_value)
                    
                    # Get hours value if available
                    hours_value = 0.0
                    if date in hours_dict and work_type in hours_dict[date]:
                        hours_value = hours_dict[date][work_type]
                        if pd.isna(hours_value) or hours_value is None:
                            hours_value = 0.0
                        else:
                            hours_value = float(hours_value)
                        logger.info(f"Hours value for {work_type}: {hours_value}")
                    
                    # Formatting date as string for SQL
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Log the parameters being sent to the stored procedure
                    log_msg = (f"Executing stored procedure with params: Date={date_str}, "
                             f"PunchCode={punch_code_int}, NoOfMan={man_value}, "
                             f"Hours={hours_value}, Username={username}")
                    logger.info(log_msg)
                    
                    # Execute stored procedure for this prediction
                    cursor.execute(
                        "EXEC usp_UpsertPrediction @Date=?, @PunchCode=?, @NoOfMan=?, @Hours=?, @Username=?",
                        date_str, 
                        punch_code_int, 
                        float(man_value),  # Ensure it's a float
                        float(hours_value),  # Ensure it's a float
                        username
                    )
                    
                    # Commit after each prediction to avoid losing all on error
                    conn.commit()
                    
                    success_count += 1
                    logger.info(f"Successfully executed stored procedure for {date}, {punch_code_int}")
                    
                except Exception as item_error:
                    # Log the specific error for this prediction
                    logger.error(f"Error processing prediction for date={date}, work_type={work_type}: {str(item_error)}")
                    logger.error(traceback.format_exc())
                    error_count += 1
                    
                    # Try to continue with next prediction
                    try:
                        conn.rollback()  # Rollback this specific prediction
                    except:
                        pass
        
        logger.info(f"Successfully saved {success_count} predictions to database")
        if error_count > 0:
            logger.warning(f"Failed to save {error_count} predictions due to errors")
        
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error saving predictions to database: {str(e)}")
        logger.error(traceback.format_exc())
        if 'conn' in locals() and conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if 'cursor' in locals() and cursor:
            try:
                cursor.close()
            except:
                pass
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass

def get_saved_predictions(start_date, end_date, work_types=None, server=None, database=None):
    """
    Get saved predictions from the database using stored procedure
    
    Parameters:
    -----------
    start_date : datetime.date
        Start date for predictions
    end_date : datetime.date
        End date for predictions
    work_types : list, optional
        List of work types to filter by, or None for all
    server : str, optional
        SQL Server name (defaults to configured SQL_SERVER)
    database : str, optional
        Database name (defaults to configured SQL_DATABASE)
        
    Returns:
    --------
    tuple
        (predictions_dict, hours_dict)
    """
    try:
        # Get configured values if not provided
        server = server or SQL_SERVER
        database = database or SQL_DATABASE
        
        # Get database connection
        conn = SQLDataConnector.connect_to_sql(
            server=server,
            database=database,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if conn is None:
            return {}, {}
        
        cursor = conn.cursor()
        
        # Prepare punch codes parameter if provided
        punch_codes_param = None
        if work_types:
            try:
                int_work_types = [int(wt) for wt in work_types]
                punch_codes_param = ','.join(map(str, int_work_types))
            except ValueError as e:
                logger.warning(f"Error converting work types to integers: {str(e)}")
                # Filter for valid integer work types only
                int_work_types = []
                for wt in work_types:
                    try:
                        int_work_types.append(int(wt))
                    except ValueError:
                        continue
                if int_work_types:
                    punch_codes_param = ','.join(map(str, int_work_types))
        
        # Execute stored procedure
        if punch_codes_param:
            cursor.execute(
                "EXEC usp_GetSavedPredictions @StartDate=?, @EndDate=?, @PunchCodes=?",
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                punch_codes_param
            )
        else:
            cursor.execute(
                "EXEC usp_GetSavedPredictions @StartDate=?, @EndDate=?",
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        # Initialize results
        predictions_dict = {}
        hours_dict = {}
        
        # Process results
        for row in cursor.fetchall():
            date_obj = row[0]  # Date from the database
            punch_code_int = row[1]  # PunchCode as integer
            no_of_man = row[2]
            hours = row[3]
            
            # Convert date from SQL Server format if needed
            if isinstance(date_obj, str):
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
            
            # Convert PunchCode to string for application use
            punch_code = str(punch_code_int)
            
            # Initialize dictionaries for this date if needed
            if date_obj not in predictions_dict:
                predictions_dict[date_obj] = {}
                hours_dict[date_obj] = {}
            
            # Store values
            predictions_dict[date_obj][punch_code] = no_of_man
            hours_dict[date_obj][punch_code] = hours
        
        result_count = len(predictions_dict)
        logger.info(f"Retrieved {result_count} prediction dates from database")
        return predictions_dict, hours_dict
    
    except Exception as e:
        logger.error(f"Error retrieving predictions from database: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def load_demand_with_kpi_data(next_working_day, server=SQL_SERVER, database=SQL_DATABASE_LIVE, 
                             trusted_connection=SQL_TRUSTED_CONNECTION,
                             username=None, password=None, chunk_size=CHUNK_SIZE):
    """
    Load demand forecast data with KPI values for next day prediction
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing demand data with KPI values
    """
    try:
        logger.info(f"Loading demand data with KPI from {server}/{database}")
        
        # Your provided SQL query
        query = f"""
       DECLARE @NextWorkingDay DATE = '{next_working_day}';
        -- Main query with KPIValue joined in
        SELECT 
            IIF(R08T1.oppdate <= @NextWorkingDay, @NextWorkingDay, R08T1.oppdate) AS PlanDate,
            COUNT(*) AS nrows,
            SUM(reqquant - delquant) AS Quantity,
            ISNULL(kpi.KPIValue, 0) AS KPIValue,
            pc.Punchcode
        FROM fsystemp.dbo.O08T1
        JOIN fsystemp.dbo.R08T1 
            ON O08T1.shortr08 = R08T1.shortr08
        -- PunchCode mapping
        OUTER APPLY (
            SELECT 
                CASE
                    WHEN routeno = 'MÄSSA' THEN '207'
                    WHEN routeno LIKE 'N[12]Z%' THEN '209'
                    WHEN routeno LIKE '[12]Z%' THEN '209'
                    WHEN routeno IN ('SORT1', 'SORTP1') THEN '209'
                    WHEN routeno IN ('BOOZT', 'ÅHLENS', 'AMZN', 'ENS1', 'ENS2', 'EMV', 'EXPRES', 'KLUBB', 
                                     'ÖP', 'ÖPFAPO', 'ÖPLOCK', 'ÖPSPEC', 'ÖPUTRI', 'PRINTW', 'RLEV') THEN '211'
                    WHEN routeno IN ('LÄROME', 'SORDER', 'ORKLA', 'REAAKB', 'REAUGG') THEN '214'
                    WHEN routeno IN ('ADLIB', 'BIB', 'BOKUS', 'DIVNÄT', 'BUYERS') THEN '215'
                    WHEN divcode IN ('LIB', 'NYP', 'STU') THEN '213'
                    WHEN routeno NOT IN ('LÄROME', 'SORDER', 'FSMAK') THEN '211'
                    ELSE '000'
                END AS Punchcode
        ) pc
        -- Join KPIData to fetch KPIValue for matching Punchcode + Date
        LEFT JOIN ABC.dbo.KPIData kpi
            ON kpi.PunchCode = pc.Punchcode AND kpi.KPIDate =@NextWorkingDay
        WHERE linestat IN (2, 4, 22, 30)
          AND R08T1.oppdate <= @NextWorkingDay
        GROUP BY 
            IIF(R08T1.oppdate <=@NextWorkingDay, @NextWorkingDay, R08T1.oppdate),
            pc.Punchcode,
            kpi.KPIValue
        -- Include KPI-only codes (e.g., 202, 203...) via UNION
        UNION
        SELECT 
            KPIDate AS PlanDate,
            0 AS nrows,
            0 AS Quantity,
            KPIValue,
            CAST(PunchCode AS VARCHAR) AS Punchcode
        FROM ABC.dbo.KPIData
        WHERE PunchCode IN (202, 203, 206, 210, 217)
          AND KPIDate = @NextWorkingDay
        -- Final sorting
        ORDER BY 
            PlanDate, 
            Punchcode;
        """
        
        # Execute query using existing connection method
        df = extract_sql_data(
            server=server,
            database=database,
            query=query,
            username=username,
            password=password,
            trusted_connection=trusted_connection,
            chunk_size=chunk_size
        )
        
        if df is not None and not df.empty:
            df['PlanDate'] = pd.to_datetime(df['PlanDate'])
            df['Punchcode'] = df['Punchcode'].astype(str)
            logger.info(f"Successfully loaded demand data with KPI. Shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading demand data with KPI: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    

def load_utilization_vs_prediction(start_date, end_date):
    query = f"""
    SELECT 
        u.Date AS Date,
        u.PunchCode AS PunchCode,
        u.Hours AS ActualHours,
        u.NoOfMan AS ActualNoOfMan,
        p.Hours AS PredictedHours,
        p.NoOfMan/8 AS PredictedNoOfMan
    FROM WorkUtilizationData u
    INNER JOIN PredictionData p
        ON u.Date = p.Date AND u.PunchCode = p.PunchCode
    WHERE u.Date BETWEEN '{start_date}' AND '{end_date}';
    """
    try:
        conn = SQLDataConnector.connect_to_sql(
                server=SQL_SERVER,
                database=SQL_DATABASE,
                username= SQL_USERNAME,
                password=SQL_PASSWORD,
                trusted_connection=SQL_TRUSTED_CONNECTION
            )
        
        if conn is None:
            return None
        
        # Extract data to DataFrame
        df = SQLDataConnector.extract_to_df(query, conn, chunk_size=CHUNK_SIZE)
        
        # Close connection
        conn.close()
        # Clean numeric strings (replace commas with dots and convert)
        for col in ['ActualHours', 'ActualNoOfMan', 'PredictedHours', 'PredictedNoOfMan']:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        return df

    except Exception as e:
        logger.error(f"Error in extract_sql_data: {str(e)}")
        logger.error(traceback.format_exc())
        return None


    # Clean numeric strings (replace commas with dots and convert)
    for col in ['ActualHours', 'ActualNoOfMan', 'PredictedHours', 'PredictedNoOfMan']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df



        

        
