"""
KPI data management utilities for the Work Utilization Prediction app.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
import pyodbc

from utils.sql_data_connector import SQLDataConnector
from config import SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure logger
logger = logging.getLogger(__name__)

def load_punch_codes():
    """
    Load punch codes from the PunchCodes reference table.
    """
    try:
        # Connect to SQL Server using the utility class
        conn = SQLDataConnector.connect_to_sql(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if not conn:
            logger.error("Failed to connect to database")
            return []
        
        cursor = conn.cursor()
        
        # Query active punch codes
        cursor.execute("""
            SELECT PunchCodeID, PunchCodeValue, Description 
            FROM PunchCodes 
            WHERE IsActive = 1 
            ORDER BY PunchCodeValue
        """)
        
        results = cursor.fetchall()
        
        # Format results as a list of dictionaries
        punch_codes = [
            {"id": row[0], "value": row[1], "description": row[2] if row[2] else ""}
            for row in results
        ]
        
        # If no punch codes found, use default list
        if not punch_codes:
            punch_codes = [
                {"id": None, "value": 213, "description": ""},
                {"id": None, "value": 217, "description": ""},
                {"id": None, "value": 214, "description": ""},
                {"id": None, "value": 206, "description": ""},
                {"id": None, "value": 211, "description": ""},
                {"id": None, "value": 210, "description": ""},
                {"id": None, "value": 215, "description": ""},
                {"id": None, "value": 209, "description": ""},
                {"id": None, "value": 202, "description": ""},
                {"id": None, "value": 203, "description": ""}
            ]
        
        logger.info(f"Loaded {len(punch_codes)} punch codes from database")
        return punch_codes
        
    except Exception as e:
        logger.error(f"Error loading punch codes: {str(e)}")
        logger.error(traceback.format_exc())
        # Return default list if error occurs
        return [
            {"id": None, "value": 213, "description": ""},
            {"id": None, "value": 217, "description": ""},
            {"id": None, "value": 214, "description": ""},
            {"id": None, "value": 206, "description": ""},
            {"id": None, "value": 211, "description": ""},
            {"id": None, "value": 210, "description": ""},
            {"id": None, "value": 215, "description": ""},
            {"id": None, "value": 209, "description": ""},
            {"id": None, "value": 202, "description": ""},
            {"id": None, "value": 203, "description": ""}
        ]
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def initialize_kpi_dataframe(from_date, to_date, date_range_type, punch_codes=None):
    """
    Initialize an empty KPI dataframe with punch codes as columns and dates as rows.
    """
    try:
        # If punch codes not provided, load them
        if punch_codes is None:
            punch_codes = load_punch_codes()
        
        # Extract just the punch code values
        punch_code_values = [str(pc["value"]) for pc in punch_codes]
        
        # Determine rows based on date range type
        if date_range_type == "Daily":
            # Daily dates
            dates = pd.date_range(from_date, to_date)
            date_labels = [d.strftime("%Y-%m-%d") for d in dates]
            
        elif date_range_type == "Weekly":
            # Weekly date ranges
            start = pd.Timestamp(from_date)
            end = pd.Timestamp(to_date)
            date_labels = []
            
            # Get the first Sunday before or on the start date (WEEKDAY=0 in SQL Server)
            # This matches the SQL Server DATEPART(WEEKDAY) function which starts weeks on Sunday
            current = start - pd.DateOffset(days=start.dayofweek + 1)  # +1 to go back to Sunday (Python week starts on Monday)
            if current > start:
                current -= pd.DateOffset(days=7)  # Go back another week if we went past the start date
            
            # Generate week ranges that match SQL Server's format
            while current <= end:
                week_end = current + pd.DateOffset(days=6)  # Sunday to Saturday
                date_labels.append(f"{current.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
                current += pd.DateOffset(days=7)
                
        else:  # Monthly
            # Monthly dates
            months = pd.date_range(
                start=pd.Timestamp(from_date).replace(day=1),
                end=pd.Timestamp(to_date).replace(day=28),
                freq='MS'
            )
            date_labels = [d.strftime("%Y-%m") for d in months]
        
        # Create DataFrame with the right structure - initialize with zeros as floats
        rows = []
        for date_label in date_labels:
            row = {"Date": date_label}
            # Add empty cells for each punch code - using 0.0 (float) instead of 0 (int)
            for code in punch_code_values:
                row[code] = 0.0
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Ensure all numeric columns are float type
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(float)
                
        # Debug log of the dates in the dataframe
        date_list = df['Date'].tolist()
        logger.info(f"Initialized dataframe with {len(date_list)} date entries: {date_list}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error initializing KPI dataframe: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a minimal dataframe if there's an error
        return pd.DataFrame({"Date": [from_date.strftime("%Y-%m-%d")]})

def load_kpi_data(start_date, end_date, period_type='DAILY'):
    """
    Load KPI data from the database for the specified date range and period type.
    Returns data with punch codes as columns and dates as rows.
    """
    try:
        # Get punch codes first
        punch_codes = load_punch_codes()
        
        # Create empty dataframe with proper structure
        result_df = initialize_kpi_dataframe(start_date, end_date, 
                                           "Daily" if period_type == 'DAILY' else 
                                           "Weekly" if period_type == 'WEEKLY' else "Monthly",
                                           punch_codes)
        
        # Connect to SQL Server
        conn = SQLDataConnector.connect_to_sql(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if not conn:
            logger.error("Failed to connect to database")
            return result_df
        
        cursor = conn.cursor()
        
        # Call stored procedure to get data
        cursor.execute(
            "EXEC usp_GetKPIDataByDateRange @StartDate=?, @EndDate=?, @PeriodType=?",
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), period_type
        )
        
        # Process results into dataframe
        rows = cursor.fetchall()
        logger.info(f"Retrieved {len(rows)} data points from database")
        
        # Process differently based on period type
        if period_type == 'DAILY':
            # Update dataframe with actual values - stored procedure returns KPIDate, PunchCodeValue, KPIValue
            for row in rows:
                kpi_date = row[0]
                punch_code = str(row[1])

                kpi_value = 0.0  # Default value
                if row[2] is not None:
                    kpi_value = float(row[2])
                
                # Format date to match dataframe format
                date_str = pd.Timestamp(kpi_date).strftime('%Y-%m-%d')
                
                # Find date in dataframe and update value
                date_rows = result_df[result_df['Date'] == date_str]
                if not date_rows.empty and punch_code in result_df.columns:
                    idx = date_rows.index[0]
                    result_df.at[idx, punch_code] = kpi_value


        elif period_type == 'WEEKLY':
            # Update weekly data - stored procedure returns WeekRange, PunchCodeValue, AverageKPIValue
            for row in rows:
                week_range = row[0]
                punch_code = str(row[1])
                
                # Handle NULL values in AverageKPIValue
                avg_value = 0.0  # Default value
                if row[2] is not None:
                    avg_value = float(row[2])
                
                # Debug logging to see what's coming from the database
                logger.info(f"Weekly data: Week={week_range}, PunchCode={punch_code}, Value={avg_value}")
                
                # Find week in dataframe and update value
                week_rows = result_df[result_df['Date'] == week_range]
                if not week_rows.empty and punch_code in result_df.columns:
                    idx = week_rows.index[0]
                    result_df.at[idx, punch_code] = avg_value
                else:
                    logger.warning(f"Could not find matching row for week {week_range} and punch code {punch_code}")

                    
        # elif period_type == 'WEEKLY':
        #     # Update weekly data - stored procedure returns WeekRange, PunchCodeValue, AverageKPIValue
        #     for row in rows:
        #         week_range = row[0]
        #         punch_code = str(row[1])
                
        #         # Handle NULL values in AverageKPIValue
        #         avg_value = 0.0  # Default value
        #         if row[2] is not None:
        #             avg_value = float(row[2])
                
        #         # Find week in dataframe and update value
        #         week_rows = result_df[result_df['Date'] == week_range]
        #         if not week_rows.empty and punch_code in result_df.columns:
        #             idx = week_rows.index[0]
        #             result_df.at[idx, punch_code] = avg_value
                    
        else:  # MONTHLY
            # Update monthly data - stored procedure returns MonthYearStr, PunchCodeValue, AverageKPIValue
            for row in rows:
                month_str = row[0]
                punch_code = str(row[1])
                
                # Handle NULL values in AverageKPIValue
                avg_value = 0.0  # Default value
                if row[2] is not None:
                    avg_value = float(row[2])
                
                # Find month in dataframe and update value
                month_rows = result_df[result_df['Date'] == month_str]
                if not month_rows.empty and punch_code in result_df.columns:
                    idx = month_rows.index[0]
                    result_df.at[idx, punch_code] = avg_value

        for col in result_df.columns:
            if col != 'Date':
                result_df[col] = result_df[col].fillna(0.0)
                
        return result_df
        
    except Exception as e:
        logger.error(f"Error loading KPI data: {str(e)}")
        logger.error(traceback.format_exc())
        # If error, return empty dataframe with correct structure
        return initialize_kpi_dataframe(start_date, end_date, 
                                       "Daily" if period_type == 'DAILY' else 
                                       "Weekly" if period_type == 'WEEKLY' else "Monthly")
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

def save_kpi_data(kpi_df, start_date, end_date, username, period_type='DAILY'):
    """
    Save KPI data to the database from a dataframe with dates as rows and punch codes as columns.
    """
    try:
        # Connect to SQL Server
        conn = SQLDataConnector.connect_to_sql(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if not conn:
            logger.error("Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        
        # Get list of punch codes (column names except 'Date')
        punch_codes = [col for col in kpi_df.columns if col != 'Date']
        
        # Counter for successful saves
        save_count = 0
        print(kpi_df.columns)
        # Loop through each row (date)
        for _, row in kpi_df.iterrows():
            date_label = row['Date']
            
            # Process based on period type
            if period_type == 'DAILY':
                # Single date
                kpi_date = pd.Timestamp(date_label)
                
                # For each punch code (column)
                for punch_code in punch_codes:
                    kpi_value = row[punch_code]
                    
                    # Skip empty or zero values
                    if pd.notnull(kpi_value) and kpi_value != "" and float(kpi_value) != 0:
                        try:
                            punch_code_int = int(punch_code)
                            
                            # Get PunchCodeID
                            cursor.execute("SELECT PunchCodeID FROM PunchCodes WHERE PunchCodeValue = ?", punch_code_int)
                            result = cursor.fetchone()
                            
                            if not result:
                                # Create punch code if it doesn't exist
                                cursor.execute(
                                    "INSERT INTO PunchCodes (PunchCodeValue, CreatedBy) VALUES (?, ?)",
                                    punch_code_int, username
                                )
                                # Get the newly created PunchCodeID
                                cursor.execute("SELECT PunchCodeID FROM PunchCodes WHERE PunchCodeValue = ?", punch_code_int)
                                result = cursor.fetchone()
                            
                            if result:
                                punch_code_id = result[0]
                                # Save the daily value
                                cursor.execute(
                                    "EXEC usp_UpsertKPIData @PunchCodeID=?, @KPIDate=?, @KPIValue=?, @Username=?",
                                    punch_code_id, kpi_date.strftime('%Y-%m-%d'), float(kpi_value), username
                                )
                                save_count += 1
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not process punch code {punch_code}: {e}")
            
            elif period_type == 'WEEKLY':
                # Parse week range "YYYY-MM-DD to YYYY-MM-DD"
                dates = date_label.split(' to ')
                if len(dates) == 2:
                    week_start = pd.Timestamp(dates[0])
                    week_end = pd.Timestamp(dates[1])
                    
                    # For each punch code
                    for punch_code in punch_codes:
                        kpi_value = row[punch_code]
                        
                        # Skip empty or zero values
                        if pd.notnull(kpi_value) and kpi_value != "" and float(kpi_value) != 0:
                            try:
                                punch_code_int = int(punch_code)
                                
                                # Get PunchCodeID
                                cursor.execute("SELECT PunchCodeID FROM PunchCodes WHERE PunchCodeValue = ?", punch_code_int)
                                result = cursor.fetchone()
                                
                                if not result:
                                    # Create punch code if it doesn't exist
                                    cursor.execute(
                                        "INSERT INTO PunchCodes (PunchCodeValue, CreatedBy) VALUES (?, ?)",
                                        punch_code_int, username
                                    )
                                    cursor.execute("SELECT PunchCodeID FROM PunchCodes WHERE PunchCodeValue = ?", punch_code_int)
                                    result = cursor.fetchone()
                                
                                if result:
                                    punch_code_id = result[0]
                                    
                                    # Save the same value for each day in the week
                                    current_date = week_start
                                    while current_date <= week_end:
                                        cursor.execute(
                                            "EXEC usp_UpsertKPIData @PunchCodeID=?, @KPIDate=?, @KPIValue=?, @Username=?",
                                            punch_code_id, current_date.strftime('%Y-%m-%d'), float(kpi_value), username
                                        )
                                        save_count += 1
                                        current_date += pd.DateOffset(days=1)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Could not process punch code {punch_code}: {e}")
            
            elif period_type == 'MONTHLY':
                # Parse month from "YYYY-MM" format
                try:
                    year, month = map(int, date_label.split('-'))
                    month_start = pd.Timestamp(year, month, 1)
                    month_end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)
                    
                    # For each punch code
                    for punch_code in punch_codes:
                        kpi_value = row[punch_code]
                        
                        # Skip empty or zero values
                        if pd.notnull(kpi_value) and kpi_value != "" and float(kpi_value) != 0:
                            try:
                                punch_code_int = int(punch_code)
                                
                                # Get PunchCodeID
                                cursor.execute("SELECT PunchCodeID FROM PunchCodes WHERE PunchCodeValue = ?", punch_code_int)
                                result = cursor.fetchone()
                                
                                if not result:
                                    # Create punch code if it doesn't exist
                                    cursor.execute(
                                        "INSERT INTO PunchCodes (PunchCodeValue, CreatedBy) VALUES (?, ?)",
                                        punch_code_int, username
                                    )
                                    cursor.execute("SELECT PunchCodeID FROM PunchCodes WHERE PunchCodeValue = ?", punch_code_int)
                                    result = cursor.fetchone()
                                
                                if result:
                                    punch_code_id = result[0]
                                    
                                    # Save the same value for each day in the month
                                    current_date = month_start
                                    while current_date <= month_end:
                                        cursor.execute(
                                            "EXEC usp_UpsertKPIData @PunchCodeID=?, @KPIDate=?, @KPIValue=?, @Username=?",
                                            punch_code_id, current_date.strftime('%Y-%m-%d'), float(kpi_value), username
                                        )
                                        save_count += 1
                                        current_date += pd.DateOffset(days=1)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Could not process punch code {punch_code}: {e}")
                except ValueError as e:
                    logger.error(f"Could not parse month format '{date_label}': {e}")
                    
        conn.commit()
        logger.info(f"Successfully saved {save_count} KPI data points")
        return True
        
    except Exception as e:
        logger.error(f"Error saving KPI data: {str(e)}, {logger.info(kpi_df.columns)}")
        logger.error(traceback.format_exc())
        if 'conn' in locals() and conn:
            conn.rollback()
        return False
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()


def save_financial_year(start_date, end_date, username):
    """
    Save/transfer financial year KPI data from KPIData table to KPIFinancialYears table
    
    Parameters:
    -----------
    start_date : datetime.date
        Start date of the financial year (May 1)
    end_date : datetime.date
        End date of the financial year (April 30)
    username : str
        Username for audit purposes
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Connect to SQL Server
        conn = SQLDataConnector.connect_to_sql(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if not conn:
            logger.error("Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        
        # Create financial year description (e.g. "FY 2025-2026")
        fy_description = f"{start_date.year}-{end_date.year}"
        
        # Convert dates to string format and boolean to int to avoid parameter binding issues
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        is_active_int = 1  # Convert boolean to int for BIT parameter
        # -print(end_date_str, '-', start_date_str)
        # Call stored procedure to transfer data from KPIData to KPIFinancialYears
        try:
            cursor.execute(
                "EXEC usp_UpsertKPIFinancialYear @StartDate=?, @EndDate=?, @Description=?, @Username=?, @IsActive=?",
                start_date_str, end_date_str, fy_description, username, is_active_int
            )
            logger.info('Success')
        except Exception as e:
            logger.error(f"XXXXXXXXXX {str(e)}")

        
        # Get the result
        result = cursor.fetchone()
        if result:
            fin_year = result[0]
            records_transferred = result[1]
            result_start_date = result[2]
            result_end_date = result[3]
            status = result[4]
            message = result[5]
            
            if status == 'SUCCESS':
                logger.info(f"Successfully transferred {records_transferred} records for financial year {fin_year}")
                logger.info(f"Period: {result_start_date} to {result_end_date}")
                logger.info(f"Message: {message}")
                return True
            else:
                logger.error(f"Error processing financial year: {message}")
                return False
        else:
            logger.error("No result returned from stored procedure")
            return False
    
    except Exception as e:
        logger.error(f"Error saving financial year: {str(e)}")
        logger.error(traceback.format_exc())

        return False
    
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()