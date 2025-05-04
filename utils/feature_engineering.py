"""
Feature engineering utilities for work utilization prediction.
"""
import pandas as pd
import numpy as np
import logging
import streamlit as st
from datetime import datetime, timedelta
import traceback
from config import LAG_DAYS, ROLLING_WINDOWS, CACHE_TTL

# Configure logger
logger = logging.getLogger(__name__)

@st.cache_data(ttl=CACHE_TTL)
def engineer_features(df):
    """
    Create relevant features for the prediction model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    try:
        logger.info("Engineering features")
        
        # Create a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Extract date features - these columns might already exist in the dataset
        # We'll create our own versions to ensure consistency
        data['Year_feat'] = data['Date'].dt.year
        data['Month_feat'] = data['Date'].dt.month
        data['DayOfMonth'] = data['Date'].dt.day
        data['DayOfWeek_feat'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        data['Quarter'] = data['Date'].dt.quarter
        
        # For this company: only Saturday (5) is a weekend, Sunday (6) is a working day
        data['IsWeekend_feat'] = data['DayOfWeek_feat'].apply(lambda x: 1 if x == 5 else 0)
        
        # Add week of year
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week
        
        # Add day of year
        data['DayOfYear'] = data['Date'].dt.dayofyear
        
        # Calculate days since the start of the dataset
        min_date = data['Date'].min()
        data['DaysSinceStart'] = (data['Date'] - min_date).dt.days
        
        # Convert WorkType to string to handle mixed numeric and string types
        data['WorkType'] = data['WorkType'].astype(str)
        
        logger.info(f"Feature engineering completed. Added {len(data.columns) - len(df.columns)} new features.")
        return data
    
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to engineer features: {str(e)}")

@st.cache_data(ttl=CACHE_TTL)
def create_lag_features(data, group_col='WorkType', target_col='NoOfMan', lag_days=None, rolling_windows=None):
    """
    Create lag features for each WorkType's NoOfMan value
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame with engineered features
    group_col : str
        Column to group by (default: 'WorkType')
    target_col : str
        Target column to create lag features for (default: 'NoOfMan')
    lag_days : list
        List of lag days to create (default: from config)
    rolling_windows : list
        List of rolling window sizes to create (default: from config)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    try:
        logger.info(f"Creating lag features for {target_col} grouped by {group_col}")
        
        # Set default values from config if not provided
        if lag_days is None:
            lag_days = LAG_DAYS
        
        if rolling_windows is None:
            rolling_windows = ROLLING_WINDOWS
        
        # Make a copy of the input dataframe
        data_copy = data.copy()
        
        # Group by WorkType and Date to get daily aggregates
        daily_data = data_copy.groupby([group_col, 'Date'])[target_col].sum().reset_index()
        
        # Add the engineered features we need
        daily_data['DayOfWeek_feat'] = daily_data['Date'].dt.dayofweek
        daily_data['Month_feat'] = daily_data['Date'].dt.month
        # For this company: only Saturday (5) is a weekend, Sunday (6) is a working day
        daily_data['IsWeekend_feat'] = daily_data['DayOfWeek_feat'].apply(lambda x: 1 if x == 5 else 0)
        daily_data['Year_feat'] = daily_data['Date'].dt.year
        daily_data['Quarter'] = daily_data['Date'].dt.quarter
        daily_data['DayOfMonth'] = daily_data['Date'].dt.day
        daily_data['WeekOfYear'] = daily_data['Date'].dt.isocalendar().week
        
        # Sort by WorkType and Date
        daily_data = daily_data.sort_values([group_col, 'Date'])
        
        # Create lag features for each work type
        for lag in lag_days:
            daily_data[f'{target_col}_lag_{lag}'] = daily_data.groupby(group_col)[target_col].shift(lag)
        
        # Create rolling average features for different window sizes
        for window in rolling_windows:
            daily_data[f'{target_col}_rolling_mean_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            
            # Add more advanced rolling features
            daily_data[f'{target_col}_rolling_max_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
            
            daily_data[f'{target_col}_rolling_min_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            
            daily_data[f'{target_col}_rolling_std_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
        
        # Create same day of week lag (e.g., last week's Monday for this Monday)
        daily_data[f'{target_col}_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])[target_col].shift(1)
        
        # Create same day of month lag (e.g., last month's 15th for this month's 15th)
        daily_data[f'{target_col}_same_dom_lag'] = daily_data.groupby([group_col, 'DayOfMonth'])[target_col].shift(1)
        
        # Add trend indicators
        # 7-day trend (increase/decrease compared to last week)
        if 7 in lag_days:
            daily_data[f'{target_col}_7day_trend'] = daily_data[target_col] - daily_data[f'{target_col}_lag_7']
        
        # 1-day trend
        if 1 in lag_days:
            daily_data[f'{target_col}_1day_trend'] = daily_data[target_col] - daily_data[f'{target_col}_lag_1']
        
        # Drop rows with NaN values (initial rows where lag isn't available)
        rows_before = len(daily_data)
        daily_data = daily_data.dropna()
        rows_after = len(daily_data)
        
        logger.info(f"Lag features created. Dropped {rows_before - rows_after} rows with NaN values.")
        logger.info(f"Final shape: {daily_data.shape}")
        
        return daily_data
    
    except Exception as e:
        logger.error(f"Error creating lag features: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to create lag features: {str(e)}")

def get_feature_lists(include_advanced_features=True):
    """
    Get the list of feature columns to use for modeling
    
    Parameters:
    -----------
    include_advanced_features : bool
        Whether to include advanced features
    
    Returns:
    --------
    tuple
        (numeric_features, categorical_features)
    """
    # Basic features that should always be included
    numeric_features = [
        'NoOfMan_lag_1', 
        'NoOfMan_lag_2', 
        'NoOfMan_lag_3', 
        'NoOfMan_lag_7', 
        'NoOfMan_rolling_mean_7', 
        'IsWeekend_feat'
    ]
    
    categorical_features = [
        'DayOfWeek_feat', 
        'Month_feat'
    ]
    
    # Add advanced features if requested
    if include_advanced_features:
        advanced_numeric = [
            'NoOfMan_rolling_max_7',
            'NoOfMan_rolling_min_7',
            'NoOfMan_rolling_std_7',
            'NoOfMan_same_dow_lag',
            'NoOfMan_same_dom_lag'
        ]
        
        # Only add trend features if they would exist based on configured lag days
        if 7 in LAG_DAYS and 1 in LAG_DAYS:
            advanced_numeric.extend([
                'NoOfMan_7day_trend',
                'NoOfMan_1day_trend'
            ])
            
        advanced_categorical = [
            'Quarter',
            'Year_feat'
        ]
        
        numeric_features.extend(advanced_numeric)
        categorical_features.extend(advanced_categorical)
    
    return numeric_features, categorical_features