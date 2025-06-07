"""
Holiday utilities for checking Swedish holidays.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging

# Configure logger
logger = logging.getLogger(__name__)

def get_swedish_holidays(year):
    """
    Get a dictionary of Swedish holidays for the specified year,
    excluding Sundays (as they are working days for this company)
    
    Parameters:
    -----------
    year : int
        Year to get holidays for
    
    Returns:
    --------
    dict
        Dictionary of holidays with dates as keys and holiday names as values
    """
    try:
        # Create a dictionary of known Swedish holidays for 2025
        if year == 2025:
            holidays = {
                date(2025, 1, 1): "New Year's Day",
                date(2025, 1, 6): "Epiphany",
                date(2025, 4, 18): "Good Friday",
                date(2025, 4, 21): "Easter Monday",
                date(2025, 5, 1): "Labor Day",
                date(2025, 5, 29): "Ascension Day",
                date(2025, 6, 6): "National Day of Sweden",
                date(2025, 6, 20): "Midsummer Eve",
                date(2025, 6, 21): "Midsummer's Day",
                date(2025, 11, 1): "All Saints' Day",
                date(2025, 12, 25): "Christmas Day",
                date(2025, 12, 26): "Boxing Day"
            }
            
            # Filter out any holidays that fall on a Sunday
            return {k: v for k, v in holidays.items() if k.weekday() != 6}
        
        # For other years, try using the holidays package or fallback to basic holidays
        try:
            # Try to import the holidays package
            from holidays import Sweden
            se_holidays = Sweden(years=year)
            
            # Filter out Sundays
            filtered_holidays = {}
            for date_obj, holiday_name in se_holidays.items():
                if date_obj.weekday() != 6:  # Not a Sunday
                    filtered_holidays[date_obj] = holiday_name
            
            # Ensure New Year's Day is included if it's not a Sunday
            new_years = date(year, 1, 1)
            if new_years.weekday() != 6 and new_years not in filtered_holidays:
                filtered_holidays[new_years] = "New Year's Day"
                
            return filtered_holidays
            
        except ImportError:
            # Fallback for other years if holidays package isn't available
            logger.warning(f"holidays package not installed. Creating basic holidays for {year}.")
            
            # Create basic holidays that are fixed each year
            basic_holidays = {
                date(year, 1, 1): "New Year's Day",
                date(year, 1, 6): "Epiphany",
                date(year, 5, 1): "Labor Day",
                date(year, 6, 6): "National Day of Sweden",
                date(year, 12, 25): "Christmas Day",
                date(year, 12, 26): "Boxing Day"
            }
            
            # Filter out any holidays that fall on a Sunday
            return {k: v for k, v in basic_holidays.items() if k.weekday() != 6}
            
    except Exception as e:
        logger.error(f"Error getting Swedish holidays for {year}: {str(e)}")
        # Ensure at least New Year's Day is included as a fallback
        new_years = date(year, 1, 1)
        if new_years.weekday() != 6:  # If not a Sunday
            return {new_years: "New Year's Day"}
        return {}

def is_swedish_holiday(date_to_check):
    """
    Check if a date is a Swedish holiday (excluding Sundays)
    
    Parameters:
    -----------
    date_to_check : datetime.date or datetime.datetime
        Date to check
    
    Returns:
    --------
    tuple
        (is_holiday, holiday_name)
    """
    try:
        # Convert to date object if it's a datetime
        if isinstance(date_to_check, datetime):
            date_obj = date_to_check.date()
        else:
            date_obj = date_to_check
        
        # First, check if it's a Sunday (6 = Sunday) - Sundays are working days for this company
        if date_obj.weekday() == 6:  # Sunday
            return False, None
        
        # Get the holidays for this year
        year_holidays = get_swedish_holidays(date_obj.year)
        
        # Check if the date is in our holidays dictionary
        if date_obj in year_holidays:
            return True, year_holidays[date_obj]
        
        # Special case for New Year's Day (in case it wasn't in the dictionary)
        if date_obj.month == 1 and date_obj.day == 1 and date_obj.weekday() != 6:
            return True, "New Year's Day"
            
        return False, None
        
    except Exception as e:
        logger.error(f"Error checking if date {date_to_check} is a Swedish holiday: {str(e)}")
        # Special case for New Year's Day as a fallback
        try:
            if isinstance(date_to_check, datetime):
                if date_to_check.month == 1 and date_to_check.day == 1 and date_to_check.weekday() != 6:
                    return True, "New Year's Day"
            elif date_to_check.month == 1 and date_to_check.day == 1 and date_to_check.weekday() != 6:
                return True, "New Year's Day"
        except:
            pass
        return False, None

# def is_non_working_day(date_to_check):
#     """
#     Check if the date is a non-working day (Saturday or a Swedish holiday)
#     For this company: Sunday is a working day, Saturday is not
    
#     Parameters:
#     -----------
#     date_to_check : datetime.date or datetime.datetime
#         Date to check
    
#     Returns:
#     --------
#     tuple
#         (is_non_working_day, reason)
#     """
#     try:
#         # Check if it's a holiday
#         is_holiday, holiday_name = is_swedish_holiday(date_to_check)
#         if is_holiday:
#             return True, f"Swedish Holiday: {holiday_name}"
        
#         # Get the day of week
#         if isinstance(date_to_check, datetime):
#             day_of_week = date_to_check.weekday()
#         else:
#             day_of_week = date_to_check.weekday()
        
#         # Check if it's Saturday (5 = Saturday in Python's weekday())
#         if day_of_week == 5:  # 5 = Saturday
#             return True, "Saturday (Weekend)"
        
#         # Check if it's Sunday (6 = Sunday in Python's weekday())
#         if day_of_week == 6:  # 6 = Sunday
#             return False, "Sunday (Working Day)"
        
#         # It's a working day
#         return False, None
        
#     except Exception as e:
#         logger.error(f"Error checking if date {date_to_check} is a non-working day: {str(e)}")
#         # In case of an error, try to at least check if it's New Year's Day
#         try:
#             if isinstance(date_to_check, datetime):
#                 if date_to_check.month == 1 and date_to_check.day == 1:
#                     return True, "Swedish Holiday: New Year's Day"
#         except:
#             pass
#         return False, None

def is_non_working_day(date_to_check):
    """
    Check if the date is a non-working day (Saturday or a Swedish holiday)
    For this company: Sunday is a working day, Saturday is not
    
    Parameters:
    -----------
    date_to_check : datetime.date or datetime.datetime
        Date to check
    
    Returns:
    --------
    tuple
        (is_non_working_day, reason)
    """
    try:
        # Convert to date object if it's a datetime
        if isinstance(date_to_check, datetime):
            date_obj = date_to_check.date()
        else:
            date_obj = date_to_check
            
       
        # Check if it's a holiday
        is_holiday, holiday_name = is_swedish_holiday(date_to_check)
        if is_holiday:
            logger.info(f"Date {date_obj.strftime('%Y-%m-%d')} is a holiday: {holiday_name}")
            return True, f"Swedish Holiday: {holiday_name}"
        
        # Check if it's Saturday (5 = Saturday in Python's weekday())
        if date_obj.weekday() == 5:  # 5 = Saturday
            logger.info(f"Date {date_obj.strftime('%Y-%m-%d')} is a Saturday")
            return True, "Saturday (Weekend)"
        
        # It's a working day
        # logger.info(f"Date {date_obj.strftime('%Y-%m-%d')} is a normal working day")
        return False, None
        
    except Exception as e:
        logger.error(f"Error checking if date {date_to_check} is a non-working day: {str(e)}")
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
        all_holidays = {}
        for year in years:
            year_holidays = get_swedish_holidays(year)
            all_holidays.update(year_holidays)
        
        # Create holiday features
        data['IsHoliday'] = data[date_col].apply(
            lambda x: 1 if is_swedish_holiday(x)[0] else 0
        )
        
        data['HolidayName'] = data[date_col].apply(
            lambda x: is_swedish_holiday(x)[1] if is_swedish_holiday(x)[0] else ''
        )
        
        # Add day before/after holiday flags
        holiday_dates = list(all_holidays.keys())
        
        # Day before holiday
        data['IsDayBeforeHoliday'] = data[date_col].apply(
            lambda x: 1 if (x.date() + timedelta(days=1)) in holiday_dates else 0
        )
        
        # Day after holiday
        data['IsDayAfterHoliday'] = data[date_col].apply(
            lambda x: 1 if (x.date() - timedelta(days=1)) in holiday_dates else 0
        )
        
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
    

def is_working_day_for_punch_code(date_to_check, punch_code):
    """
    Check if a specific punch code should work on the given date
    
    Parameters:
    -----------
    date_to_check : datetime.date or datetime.datetime
        Date to check
    punch_code : str
        Punch code to check working rules for
    
    Returns:
    --------
    tuple
        (is_working_day, reason_if_not_working)
    """
    try:
        from config import PUNCH_CODE_WORKING_RULES, DEFAULT_PUNCH_CODE_WORKING_DAYS
        
        # Convert to date object if it's a datetime
        if isinstance(date_to_check, datetime):
            date_obj = date_to_check.date()
        else:
            date_obj = date_to_check
        
        # Check if it's a Swedish holiday first
        is_holiday, holiday_name = is_swedish_holiday(date_to_check)
        if is_holiday:
            return False, f"Swedish Holiday: {holiday_name}"
        
        # Get the day of week (0=Monday, 6=Sunday)
        day_of_week = date_obj.weekday()
        
        # Get working days for this punch code
        punch_code_str = str(punch_code)
        working_days = PUNCH_CODE_WORKING_RULES.get(punch_code_str, DEFAULT_PUNCH_CODE_WORKING_DAYS)
        
        # Check if this punch code works on this day of week
        if day_of_week not in working_days:
            if day_of_week == 5:  # Saturday
                return False, "Saturday (Non-working for this punch code)"
            elif day_of_week == 6:  # Sunday
                return False, "Sunday (Non-working for this punch code)"
            else:
                return False, f"Non-working day for punch code {punch_code}"
        
        # It's a working day for this punch code
        return True, None
        
    except Exception as e:
        logger.error(f"Error checking working day for punch code {punch_code} on {date_to_check}: {str(e)}")
        # Default to non-working day on error
        return False, "Error checking working day rules"

def is_non_working_day_for_punch_code(date_to_check, punch_code):
    """
    Check if the date is a non-working day for a specific punch code
    (Inverse of is_working_day_for_punch_code for backward compatibility)
    
    Parameters:
    -----------
    date_to_check : datetime.date or datetime.datetime
        Date to check
    punch_code : str
        Punch code to check
    
    Returns:
    --------
    tuple
        (is_non_working_day, reason)
    """
    is_working, reason = is_working_day_for_punch_code(date_to_check, punch_code)
    return not is_working, reason