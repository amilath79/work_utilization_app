"""
Feature engineering utilities for work utilization prediction with productivity metrics.
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
        
        # Ensure Date column is datetime
        if 'Date' in data.columns:
            # Check the type of Date column
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                logger.info("Converting Date column to datetime")
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                
                # Check if conversion created NaT values and log a warning
                nat_count = data['Date'].isna().sum()
                if nat_count > 0:
                    logger.warning(f"Date conversion created {nat_count} NaT values")
                    # Drop NaT values to avoid issues in feature engineering
                    data = data.dropna(subset=['Date'])
        else:
            logger.error("Date column not found in dataframe")
            raise ValueError("Date column is required for feature engineering")
        
        # Extract date features from the Date column only
        data['Year_feat'] = data['Date'].dt.year
        data['Month_feat'] = data['Date'].dt.month
        data['DayOfMonth'] = data['Date'].dt.day
        data['DayOfWeek_feat'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        data['Quarter'] = data['Date'].dt.quarter
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week
        data['DayOfYear'] = data['Date'].dt.dayofyear
        
        # For this company: only Saturday (5) is a weekend, Sunday (6) is a working day
        data['IsWeekend_feat'] = data['DayOfWeek_feat'].apply(lambda x: 1 if x == 5 else 0)
        
        # Calculate days since the start of the dataset
        min_date = data['Date'].min()
        data['DaysSinceStart'] = (data['Date'] - min_date).dt.days
        
        # Convert WorkType to string to handle mixed numeric and string types
        data['WorkType'] = data['WorkType'].astype(str)
        
        # Process productivity metrics if they exist in the dataset
        if 'SystemHours' in data.columns and 'Quantity' in data.columns:
            # Ensure numeric values and handle potential zeros
            for col in ['SystemHours', 'Hours', 'Quantity', 'ResourceKPI', 'SystemKPI']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            
            # Feature 1: Hours to SystemHours Relationship
            # This captures the relationship without assuming either is more accurate
            data['Hours_SystemHours_Ratio'] = np.where(
                data['SystemHours'] > 0, 
                data['Hours'] / data['SystemHours'], 
                1
            )
            
            # Feature 2: Productivity per Worker 
            # How much quantity handled per worker
            data['Quantity_per_Worker'] = np.where(
                data['NoOfMan'] > 0,
                data['Quantity'] / data['NoOfMan'],
                0
            )
            
            # Feature 3: Balanced efficiency using both ResourceKPI and SystemKPI
            # Don't normalize assuming either is better, just combine them
            data['Combined_KPI'] = (data['ResourceKPI'] + data['SystemKPI']) / 2
            
            # Feature 4: Work complexity indicator
            # Higher ratio might indicate more complex work that takes more time
            data['Hours_per_Quantity'] = np.where(
                data['Quantity'] > 0,
                data['Hours'] / data['Quantity'],
                0
            )
            
            # Feature 5: System efficiency indicator
            # Higher ratio might indicate system processing takes more time
            data['SystemHours_per_Quantity'] = np.where(
                data['Quantity'] > 0,
                data['SystemHours'] / data['Quantity'],
                0
            )
            
            # Feature 6: Relative workload compared to typical for this work type
            # How this day's quantity compares to average for this work type
            # data['Relative_Quantity'] = data['Quantity'] / data.groupby('WorkType')['Quantity'].transform('mean').replace(0, 1)
            # print(data[['Date', 'Relative_Quantity']])
            # Fill any remaining NaN values
            data = data.fillna(0)
        
        logger.info(f"Feature engineering completed. Added {len(data.columns) - len(df.columns)} new features.")
        return data
    
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to engineer features: {str(e)}")

@st.cache_data(ttl=CACHE_TTL)
def create_lag_features(data, group_col='WorkType', target_col='NoOfMan', lag_days=None, rolling_windows=None):
    """
    Create lag features for each WorkType's NoOfMan value and productivity metrics
    
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
        
        # Define productivity metrics to aggregate if they exist
        productivity_metrics = [
            'Hours', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI', 
            'Combined_KPI', 'Quantity_per_Worker', 'Hours_SystemHours_Ratio',
            'Hours_per_Quantity', 'SystemHours_per_Quantity', 'Relative_Quantity'
        ]
        
        # Group by WorkType and Date to get daily aggregates
        existing_metrics = [col for col in productivity_metrics if col in data_copy.columns]

        if existing_metrics:
            # Create aggregation dictionary with target column and all available metrics
            agg_dict = {target_col: 'sum'}
            for metric in existing_metrics:
                agg_dict[metric] = 'mean' if metric.endswith('Ratio') else 'sum'
            
            daily_data = data_copy.groupby([group_col, 'Date']).agg(agg_dict).reset_index()
        else:
            daily_data = data_copy.groupby([group_col, 'Date'])[target_col].sum().reset_index()


        # Add date-derived features
        daily_data['DayOfWeek_feat'] = daily_data['Date'].dt.dayofweek
        daily_data['Month_feat'] = daily_data['Date'].dt.month
        daily_data['IsWeekend_feat'] = daily_data['DayOfWeek_feat'].apply(lambda x: 1 if x == 5 else 0)
        daily_data['Year_feat'] = daily_data['Date'].dt.year
        daily_data['Quarter'] = daily_data['Date'].dt.quarter
        daily_data['DayOfMonth'] = daily_data['Date'].dt.day
        daily_data['WeekOfYear'] = daily_data['Date'].dt.isocalendar().week
        
        # Sort by WorkType and Date
        daily_data = daily_data.sort_values([group_col, 'Date'])
        
        # Create lag features for target column (NoOfMan)
        for lag in lag_days:
            daily_data[f'{target_col}_lag_{lag}'] = daily_data.groupby(group_col)[target_col].shift(lag)
        
        # Create lag features for productivity metrics
        for metric in existing_metrics:
            # Create 1-day and 7-day lags for main productivity metrics
            if metric in ['Quantity', 'ResourceKPI', 'SystemKPI', 'Combined_KPI', 'Quantity_per_Worker']:
                for lag in [1, 7]:
                    if lag in lag_days:
                        daily_data[f'{metric}_lag_{lag}'] = daily_data.groupby(group_col)[metric].shift(lag)
            
            # Only 1-day lag for ratio features (to avoid too many features)
            elif metric.endswith('Ratio'):
                if 1 in lag_days:
                    daily_data[f'{metric}_lag_1'] = daily_data.groupby(group_col)[metric].shift(1)


        # Create rolling features for target column
        for window in rolling_windows:
            # Standard rolling features for NoOfMan
            window_funcs = {
                'mean': lambda x: x.rolling(window=window, min_periods=1).mean(),
                'max': lambda x: x.rolling(window=window, min_periods=1).max(),
                'min': lambda x: x.rolling(window=window, min_periods=1).min(),
                'std': lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            }
            
            for func_name, func in window_funcs.items():
                daily_data[f'{target_col}_rolling_{func_name}_{window}'] = daily_data.groupby(group_col)[target_col].transform(func)
            
            # Also create mean rolling features for key productivity metrics
            for metric in ['Quantity', 'Combined_KPI'] if 'Quantity' in existing_metrics else []:
                daily_data[f'{metric}_rolling_mean_{window}'] = daily_data.groupby(group_col)[metric].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Create same day of week and month lag features
        daily_data[f'{target_col}_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])[target_col].shift(1)
        daily_data[f'{target_col}_same_dom_lag'] = daily_data.groupby([group_col, 'DayOfMonth'])[target_col].shift(1)
        
        # Create quantity-based same-day-of-week lag if available
        if 'Quantity' in existing_metrics:
            daily_data['Quantity_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])['Quantity'].shift(1)
        
        # Add trend indicators
        if 7 in lag_days and 1 in lag_days:
            # Workforce trends
            daily_data[f'{target_col}_7day_trend'] = daily_data[target_col] - daily_data[f'{target_col}_lag_7']
            daily_data[f'{target_col}_1day_trend'] = daily_data[target_col] - daily_data[f'{target_col}_lag_1']
            
            # Quantity trends if available
            if 'Quantity' in existing_metrics:
                daily_data['Quantity_7day_trend'] = daily_data['Quantity'] - daily_data['Quantity_lag_7']
                daily_data['Quantity_1day_trend'] = daily_data['Quantity'] - daily_data['Quantity_lag_1']
        
        # Create workforce prediction based on quantity if available
        if 'Quantity' in existing_metrics and 'Quantity_lag_1' in daily_data.columns:
            # Calculate average NoOfMan per Quantity for each work type
            avg_workers_per_unit = daily_data.groupby(group_col).apply(
                lambda x: (x[target_col] / x['Quantity']).replace([np.inf, -np.inf], np.nan).mean()
            ).fillna(0.1)  # Default to 0.1 workers per unit if we can't calculate
            
            # Map this average back to each row
            daily_data['avg_workers_per_unit'] = daily_data[group_col].map(avg_workers_per_unit)
            
            # Predict workers based on previous day's quantity
            daily_data['Workers_Predicted_from_Quantity'] = daily_data['Quantity_lag_1'] * daily_data['avg_workers_per_unit']

        # daily_data.to_excel('daily_data.xlsx')
        # Drop rows with NaN values (initial rows where lag isn't available)
        rows_before = len(daily_data)
        # daily_data = daily_data.dropna()
        rows_after = len(daily_data)

        logger.info(f"Lag features created. Dropped {rows_before - rows_after} rows with NaN values.")
        logger.info(f"Final shape: {daily_data.shape}")

        return daily_data
    
    except Exception as e:
        logger.error(f"Error creating lag features: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to create lag features: {str(e)}")

def get_feature_lists(include_advanced_features=True, include_productivity_metrics=True):
    """
    Get the list of feature columns to use for modeling
    
    Parameters:
    -----------
    include_advanced_features : bool
        Whether to include advanced features
    include_productivity_metrics : bool
        Whether to include productivity metrics features
    
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
    
    # Add productivity metrics if requested
    if include_productivity_metrics:
        productivity_features = [
            # Quantity-based features
            'Quantity_lag_1',
            'Quantity_lag_7',
            'Quantity_rolling_mean_7',
            'Quantity_per_Worker',
            'Relative_Quantity',
            'Quantity_same_dow_lag',
            
            # KPI-based features
            'ResourceKPI_lag_1',
            'SystemKPI_lag_1',
            'Combined_KPI_lag_1',
            
            # Ratio features
            'Hours_SystemHours_Ratio_lag_1',
            'Hours_per_Quantity',
            'SystemHours_per_Quantity',
            
            # Quantity-based workforce prediction
            'Workers_Predicted_from_Quantity'
        ]
        
        # Only add if they would exist based on configured lag days
        if 7 in LAG_DAYS and 1 in LAG_DAYS:
            productivity_features.extend([
                'Quantity_7day_trend',
                'Quantity_1day_trend'
            ])
        
        numeric_features.extend(productivity_features)
    
    return numeric_features, categorical_features