"""
Feature engineering utilities - Config-driven to prevent overfitting
"""
import pandas as pd
import numpy as np
import logging
import streamlit as st
from datetime import datetime, timedelta
import traceback
from config import (
    LAG_DAYS, ROLLING_WINDOWS, CACHE_TTL, 
    FEATURE_GROUPS, PRODUCTIVITY_FEATURES, 
    ESSENTIAL_LAGS, ESSENTIAL_WINDOWS, DATE_FEATURES,
    LAG_FEATURES_COLUMNS, ROLLING_FEATURES_COLUMNS, CYCLICAL_FEATURES
)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Then use conditional decorator
def _cache_if_available(func):
    """Apply Streamlit cache only if available"""
    if STREAMLIT_AVAILABLE:
        return st.cache_data(ttl=3600)(func)
    return func

logger = logging.getLogger(__name__)

@_cache_if_available
def engineer_features(df):
    """Config-driven feature engineering to prevent overfitting"""
    try:
        data = df.copy()
        
        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date'])
        
        # Essential date features (always included)
        if FEATURE_GROUPS['DATE_FEATURES']:
            data['DayOfWeek_feat'] = data['Date'].dt.dayofweek
            data['Month_feat'] = data['Date'].dt.month
            data['IsWeekend_feat'] = data['DayOfWeek_feat'].apply(lambda x: 1 if x == 5 else 0)
            data['Quarter'] = data['Date'].dt.quarter
            data['DayOfMonth'] = data['Date'].dt.day
        
        # Convert WorkType to string
        data['WorkType'] = data['WorkType'].astype(str)
        
        # Productivity features (only if enabled and data available)
        if (FEATURE_GROUPS['PRODUCTIVITY_FEATURES'] and 
            'SystemHours' in data.columns and 'Quantity' in data.columns):
            
            # Ensure numeric values
            for col in ['SystemHours', 'Hours', 'Quantity', 'ResourceKPI', 'SystemKPI']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            
            # Create only essential productivity features from config
            for feature in PRODUCTIVITY_FEATURES:
                if feature == 'Workers_per_Hour':
                    data[feature] = np.where(data['Hours'] > 0, data['NoOfMan'] / data['Hours'], 0)
                elif feature == 'Quantity_per_Hour':
                    data[feature] = np.where(data['Hours'] > 0, data['Quantity'] / data['Hours'], 0)
                elif feature == 'Workload_Density':
                    data[feature] = np.where(data['NoOfMan'] > 0, data['Quantity'] / data['NoOfMan'], 0)
                elif feature == 'KPI_Performance':
                    data[feature] = np.where(data['SystemKPI'] > 0, data['ResourceKPI'] / data['SystemKPI'], 1)
        
        # Cyclical features (optional - can cause overfitting)
        if FEATURE_GROUPS['CYCLICAL_FEATURES']:
            data['Month_Sin'] = np.sin(2 * np.pi * data['Month_feat'] / 12)
            data['Week_Sin'] = np.sin(2 * np.pi * data['DayOfWeek_feat'] / 7)
        
        data = data.fillna(0)
        return data
    
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        raise Exception(f"Failed to engineer features: {str(e)}")

@_cache_if_available
def create_lag_features(data, group_col='WorkType', target_col='NoOfMan', lag_days=None, rolling_windows=None):
    """Config-driven lag feature creation"""
    try:
        # Use essential values from config to prevent overfitting
        if lag_days is None:
            lag_days = ESSENTIAL_LAGS if FEATURE_GROUPS['LAG_FEATURES'] else []
        if rolling_windows is None:
            rolling_windows = ESSENTIAL_WINDOWS if FEATURE_GROUPS['ROLLING_FEATURES'] else []

        data_copy = data.copy()
        
        # Simple aggregation - avoid complexity
        daily_data = data_copy.groupby([group_col, 'Date']).agg({
            target_col: 'sum',
            'Quantity': 'sum' if 'Quantity' in data_copy.columns else lambda x: 0,
            'Hours': 'sum' if 'Hours' in data_copy.columns else lambda x: 0
        }).reset_index()

        # Add essential date features
        daily_data['DayOfWeek_feat'] = daily_data['Date'].dt.dayofweek
        daily_data['Month_feat'] = daily_data['Date'].dt.month
        daily_data['IsWeekend_feat'] = daily_data['DayOfWeek_feat'].apply(lambda x: 1 if x == 5 else 0)
        daily_data['DayOfMonth'] = daily_data['Date'].dt.day
        daily_data['Quarter'] = daily_data['Date'].dt.quarter
        
        daily_data = daily_data.sort_values([group_col, 'Date'])
        
        # Create lag features (only essential ones from config)
        if FEATURE_GROUPS['LAG_FEATURES']:
            for lag in lag_days:
                daily_data[f'{target_col}_lag_{lag}'] = daily_data.groupby(group_col)[target_col].shift(lag)
                
                # Add quantity lag only for most important lag
                if lag == 1 and 'Quantity' in daily_data.columns:
                    daily_data[f'Quantity_lag_{lag}'] = daily_data.groupby(group_col)['Quantity'].shift(lag)

        # Create rolling features (only essential ones from config)
        if FEATURE_GROUPS['ROLLING_FEATURES']:
            for window in rolling_windows:
                daily_data[f'{target_col}_rolling_mean_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())

        # Pattern features (optional - can cause overfitting)
        if FEATURE_GROUPS['PATTERN_FEATURES']:
            daily_data[f'{target_col}_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])[target_col].shift(1)

        # Trend features (optional - can cause overfitting)
        if FEATURE_GROUPS['TREND_FEATURES'] and 1 in lag_days and 7 in lag_days:
            daily_data[f'{target_col}_trend_7d'] = daily_data[target_col] - daily_data[f'{target_col}_lag_7']

        logger.info(f"Created features using config: LAG_FEATURES={FEATURE_GROUPS['LAG_FEATURES']}, "
                   f"PRODUCTIVITY_FEATURES={FEATURE_GROUPS['PRODUCTIVITY_FEATURES']}")
        logger.info(f"Final shape: {daily_data.shape}")

        return daily_data
    
    except Exception as e:
        logger.error(f"Error creating lag features: {str(e)}")
        raise Exception(f"Failed to create lag features: {str(e)}")
    



def add_rolling_features_by_group(df):
    """
    Adds rolling mean and std features for selected columns grouped by WorkType and ordered by Date.
    """
    df = df.sort_values(['WorkType', 'Date']).copy()
    
    for col in ROLLING_FEATURES_COLUMNS:
        for window in ROLLING_WINDOWS:
            roll_mean_col = f"{col}_roll_{window}_mean"
            roll_std_col = f"{col}_roll_{window}_std"
            
            df[roll_mean_col] = df.groupby('WorkType')[col].shift(1).rolling(window).mean()
            df[roll_std_col] = df.groupby('WorkType')[col].shift(1).rolling(window).std()
    
    return df


def add_lag_features_by_group(df):
    """
    Adds lag features for selected columns grouped by WorkType and ordered by Date.
    """
    df = df.sort_values(['WorkType', 'Date']).copy()
    
    for col in LAG_FEATURES_COLUMNS:
        for lag in ESSENTIAL_LAGS:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df.groupby('WorkType')[col].shift(lag)
    
    return df


def add_rolling_features_by_group(df):
    """
    Adds rolling mean and std features for selected columns grouped by WorkType and ordered by Date.
    """
    df = df.sort_values(['WorkType', 'Date']).copy()
    
    for col in ROLLING_FEATURES_COLUMNS:
        for window in ROLLING_WINDOWS:
            roll_mean_col = f"{col}_roll_{window}_mean"
            roll_std_col = f"{col}_roll_{window}_std"
            
            df[roll_mean_col] = df.groupby('WorkType')[col].shift(1).rolling(window).mean()
            df[roll_std_col] = df.groupby('WorkType')[col].shift(1).rolling(window).std()
    
    return df

def add_cyclical_features(df):
    """
    Adds cyclical encodings (sin, cos) for features defined in config.CYCLICAL_FEATURES.
    Example: 'DayOfWeek' with period 7 will produce 'DayOfWeek_sin' and 'DayOfWeek_cos'.
    """
    for feature, period in CYCLICAL_FEATURES.items():
        sin_col = f"{feature}_sin"
        cos_col = f"{feature}_cos"

        df[sin_col] = np.sin(2 * np.pi * df[feature] / period)
        df[cos_col] = np.cos(2 * np.pi * df[feature] / period)
    
    return df

def add_trend_features(df):
    # Example: cumulative sum of quantity to reflect trend
    df = df.sort_values('Date')
    df['Cumulative_Quantity'] = df['Quantity'].cumsum()
    return df


def add_pattern_features(df):
    # Example: simple rolling mean to identify patterns
    df = df.sort_values('Date')
    df['Quantity_3d_avg'] = df['Quantity'].rolling(window=3, min_periods=1).mean()
    return df

