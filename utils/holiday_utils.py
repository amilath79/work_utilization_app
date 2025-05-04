"""
Holiday utilities for checking Swedish holidays.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)

def get_swedish_holidays(start_year, end_year=None):
    """
    Get a dictionary of Swedish holidays for the specified year range
    
    Parameters:
    -----------
    start_year : int
        Start year
    end_year : int, optional
        End year, if different from start_year
    
    Returns:
    --------
    dict
        Dictionary of holidays with dates as keys and holiday names as values
    """
    if end_year is None:
        end_year = start_year
        
    try:
        # Import holidays library
        try:
            from holidays import Sweden
        except ImportError:
            logger.error("holidays package not installed. Install with: pip install holidays")
            return {}
            
        # Get Swedish holidays for the specified years
        se_holidays = Sweden(years=range(start_year, end_year + 1))
        return se_holidays
    except Exception as e:
        logger.error(f"Error getting Swedish holidays: {str(e)}")
        return {}

def is_swedish_holiday(date):
    """
    Check if a date is a Swedish holiday
    
    Parameters:
    -----------
    date : datetime.date or datetime.datetime
        Date to check
    
    Returns:
    --------
    tuple
        (is_holiday, holiday_name)
    """
    try:
        year = date.year
        se_holidays = get_swedish_holidays(year)
        
        # Convert to datetime.date if it's a datetime
        if isinstance(date, datetime):
            date = date.date()
            
        if date in se_holidays:
            return True, se_holidays[date]
        else:
            return False, None
    except Exception as e:
        logger.error(f"Error checking if date {date} is a Swedish holiday: {str(e)}")
        return False, None

def is_non_working_day(date):
    """
    Check if the date is a non-working day (Saturday or a Swedish holiday)
    For this company: Sunday is a working day, Saturday is not
    
    Parameters:
    -----------
    date : datetime.date or datetime.datetime
        Date to check
    
    Returns:
    --------
    tuple
        (is_non_working_day, reason)
    """
    try:
        # Check if it's a holiday
        is_holiday, holiday_name = is_swedish_holiday(date)
        if is_holiday:
            return True, f"Swedish Holiday: {holiday_name}"
        
        # Check if it's Saturday (6 = Saturday in Python's weekday())
        if isinstance(date, datetime):
            weekday = date.weekday()
        else:
            weekday = date.weekday()
        
        if weekday == 5:  # 5 = Saturday
            return True, "Saturday (Weekend)"
        
        # It's a working day
        return False, None
        
    except Exception as e:
        logger.error(f"Error checking if date {date} is a non-working day: {str(e)}")
        return False, None

def add_holiday_features(df, date_col='Date'):
    """
    Add holiday-related features to a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str, optional
        Name of the date column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with holiday features
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Ensure the date column is datetime
        if pd.api.types.is_string_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Get unique years in the data
        years = data[date_col].dt.year.unique()
        
        # Get holidays for all years in the data
        all_holidays = get_swedish_holidays(min(years), max(years))
        
        # Create holiday features
        data['IsHoliday'] = data[date_col].apply(lambda x: 1 if x.date() in all_holidays else 0)
        data['HolidayName'] = data[date_col].apply(lambda x: all_holidays.get(x.date(), ''))
        
        # Add day before/after holiday flags
        holiday_dates = [d for d in all_holidays.keys()]
        
        # Day before holiday
        data['IsDayBeforeHoliday'] = data[date_col].apply(
            lambda x: 1 if (x.date() + timedelta(days=1)) in holiday_dates else 0)
        
        # Day after holiday
        data['IsDayAfterHoliday'] = data[date_col].apply(
            lambda x: 1 if (x.date() - timedelta(days=1)) in holiday_dates else 0)
        
        # Add IsWeekend feature (only Saturdays, not Sundays)
        data['IsSaturday'] = data[date_col].dt.dayofweek.apply(lambda x: 1 if x == 5 else 0)  # 5 = Saturday
        
        # Add non-working day feature (combines holidays and Saturdays)
        data['IsNonWorkingDay'] = data.apply(
            lambda row: 1 if row['IsHoliday'] == 1 or row['IsSaturday'] == 1 else 0, 
            axis=1
        )
        
        logger.info(f"Added holiday features. Found {len(holiday_dates)} holidays.")
        return data
    
    except Exception as e:
        logger.error(f"Error adding holiday features: {str(e)}")
        return df  # Return original dataframe if there's an error