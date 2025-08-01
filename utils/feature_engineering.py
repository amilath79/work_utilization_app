import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import config
from utils.feature_selection import FeatureSelector

from config import (
    FEATURE_GROUPS, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS,
    LAG_FEATURES_COLUMNS, ROLLING_FEATURES_COLUMNS, 
    CYCLICAL_FEATURES, DATE_FEATURES, PRODUCTIVITY_FEATURES,
    TREND_WINDOWS, TREND_FEATURES_COLUMNS, TREND_CALCULATIONS  # Add these
)

class EnhancedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Config-driven transformer for enhanced feature engineering.
    Applies lag, rolling, date, cyclical, pattern, trend, and interaction features.
    """

    def __init__(self):
        self.lag_days = ESSENTIAL_LAGS if FEATURE_GROUPS.get('LAG_FEATURES', False) else []
        self.rolling_windows = ESSENTIAL_WINDOWS if FEATURE_GROUPS.get('ROLLING_FEATURES', False) else []
        self.lag_columns = LAG_FEATURES_COLUMNS if hasattr(config, 'LAG_FEATURES_COLUMNS') else ['Hours']
        self.rolling_columns = ROLLING_FEATURES_COLUMNS if hasattr(config, 'ROLLING_FEATURES_COLUMNS') else ['Hours']
        self.cyclical_features = CYCLICAL_FEATURES if FEATURE_GROUPS.get('CYCLICAL_FEATURES', False) else {}

        self.trend_windows = TREND_WINDOWS if hasattr(config, 'TREND_WINDOWS') else [7, 30, 90]
        self.trend_columns = TREND_FEATURES_COLUMNS if hasattr(config, 'TREND_FEATURES_COLUMNS') else ['Hours']
        self.trend_calculations = TREND_CALCULATIONS if hasattr(config, 'TREND_CALCULATIONS') else {}
        self.fitted_features_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.fitted_features_ = self._get_expected_features(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        X = self._add_date_features(X)
        X = self._add_lag_features(X)
        X = self._add_enhanced_rolling_features(X)  # ENHANCED
        X = self._add_peak_detection_features(X)    # NEW
        X = self._add_cyclical_features(X)
        X = self._add_system_features(X)
        X = self._add_trend_features(X)
        X = self._add_pattern_features(X)
        X = self._add_interaction_features(X)

        # Ensure all expected features exist
        for feat in self.fitted_features_:
            if feat not in X.columns:
                X[feat] = 0.0

        # Convert WorkType to int if present
        if 'WorkType' in X.columns:
            X['WorkType'] = pd.to_numeric(X['WorkType'], errors='coerce').fillna(0).astype(int)

        # Final column ordering
        essential = [c for c in ['WorkType', 'Quantity', 'Hours'] if c in X.columns]
        cols = essential + [f for f in self.fitted_features_ if f not in essential]
        return X[cols].fillna(0)

    def _get_expected_features(self, X):
        features = []
        # Date features
        if FEATURE_GROUPS.get('DATE_FEATURES', False):
                # Date features - NOW USING CONFIG
            features.extend(DATE_FEATURES.get('categorical', []))
            features.extend(DATE_FEATURES.get('numeric', []))
        # Lag features
        if FEATURE_GROUPS.get('LAG_FEATURES', False):
            for col in self.lag_columns:
                for lag in self.lag_days:
                    features.append(f'{col}_lag_{lag}')
        # Rolling features
        if FEATURE_GROUPS.get('ROLLING_FEATURES', False):
            for col in self.rolling_columns:
                for window in self.rolling_windows:
                    features.append(f'{col}_rolling_mean_{window}')
                    features.append(f'{col}_rolling_std_{window}')
        # Cyclical features
        if FEATURE_GROUPS.get('CYCLICAL_FEATURES', False):
            for feature, period in self.cyclical_features.items():
                features.extend([f'{feature}_sin', f'{feature}_cos'])
        # Productivity features
        if FEATURE_GROUPS.get('PRODUCTIVITY_FEATURES', False):
            features.extend(['SystemHours'])
            if hasattr(config, 'PRODUCTIVITY_FEATURES') and isinstance(PRODUCTIVITY_FEATURES, list):
                features.extend(PRODUCTIVITY_FEATURES)
        # Trend features
        if FEATURE_GROUPS.get('TREND_FEATURES', False):
            features.append('Cumulative_Quantity')
            # Enhanced trend features
            for col in self.trend_columns:
                for window in self.trend_windows:
                    if self.trend_calculations.get('slope', True):
                        features.append(f'{col}_trend_slope_{window}')
                    if self.trend_calculations.get('strength', True):
                        features.append(f'{col}_trend_strength_{window}')
                    if self.trend_calculations.get('detrended', True):
                        features.append(f'{col}_detrended_{window}')
                
                # Trend changes
                if self.trend_calculations.get('change', True) and len(self.trend_windows) >= 2:
                    for i in range(len(self.trend_windows) - 1):
                        w1, w2 = self.trend_windows[i], self.trend_windows[i + 1]
                        features.append(f'{col}_trend_change_{w1}_{w2}')
                
                # Acceleration
                if self.trend_calculations.get('acceleration', True):
                    features.append(f'{col}_trend_acceleration')
        # Pattern features
        if FEATURE_GROUPS.get('PATTERN_FEATURES', False):
            features.append('Quantity_3d_avg')
        # Interaction features
        if FEATURE_GROUPS.get('INTERACTION_FEATURES', False):
            features.extend(['Quantity_SystemHours', 'DayOfWeek_Month', 'Year_Quarter'])
        return features


    def _add_peak_detection_features(self, X):
        """Add features specifically for handling high-value spikes"""
        if 'Hours' not in X.columns:
            return X
            
        # Calculate rolling quantiles for peak detection
        for window in [7, 14, 30]:
            X[f'Hours_rolling_q75_{window}'] = X['Hours'].rolling(window, min_periods=1).quantile(0.75)
            X[f'Hours_rolling_q90_{window}'] = X['Hours'].rolling(window, min_periods=1).quantile(0.90)
            X[f'Hours_above_q75_{window}'] = (X['Hours'] > X[f'Hours_rolling_q75_{window}']).astype(int)
            X[f'Hours_peak_ratio_{window}'] = X['Hours'] / (X[f'Hours_rolling_q75_{window}'] + 1e-6)
        
        # Volatility features for spike detection
        for window in [7, 14]:
            X[f'Hours_volatility_{window}'] = X['Hours'].rolling(window, min_periods=1).std() / (X['Hours'].rolling(window, min_periods=1).mean() + 1e-6)
            X[f'Hours_acceleration_{window}'] = X['Hours'].diff().rolling(window, min_periods=1).std()
        
        return X

    # AFTER - Config-driven date feature creation
    def _add_date_features(self, df):
        if FEATURE_GROUPS.get('DATE_FEATURES', False) and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Get required features from config
            categorical_features = DATE_FEATURES.get('categorical', [])
            numeric_features = DATE_FEATURES.get('numeric', [])
            all_date_features = categorical_features + numeric_features
            
            # Create features based on config
            if 'DayOfWeek' in all_date_features:
                df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
            if 'Month' in all_date_features:
                df['Month'] = df['Date'].dt.month
            if 'WeekNo' in all_date_features:
                df['WeekNo'] = df['Date'].dt.isocalendar().week
            if 'IsWeekend' in all_date_features:
                df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
            if 'Quarter' in all_date_features:
                df['Quarter'] = df['Date'].dt.quarter
            if 'Year' in all_date_features:
                df['Year'] = df['Date'].dt.year
            if 'Day' in all_date_features:
                df['Day'] = df['Date'].dt.day
            if 'IsMonthEnd' in all_date_features:
                df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
            if 'IsMonthStart' in all_date_features:
                df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
                
        return df

    def _add_lag_features(self, df):
        if FEATURE_GROUPS.get('LAG_FEATURES', False) and 'WorkType' in df.columns:
            df = df.sort_values(['WorkType', 'Date'] if 'Date' in df.columns else ['WorkType'])
            for col in self.lag_columns:
                if col in df.columns:
                    for lag in self.lag_days:
                        df[f'{col}_lag_{lag}'] = df.groupby('WorkType')[col].shift(lag)
        return df

    def _add_enhanced_rolling_features(self, X):
        """Enhanced rolling features beyond just mean"""
        if not FEATURE_GROUPS.get('ROLLING_FEATURES', False):
            return X
            
        for col in self.rolling_columns:
            if col not in X.columns:
                continue
                
            for window in self.rolling_windows:
                # Existing mean
                X[f'{col}_rolling_mean_{window}'] = X[col].rolling(window, min_periods=1).mean()
                
                # NEW: Additional statistics for better pattern capture
                X[f'{col}_rolling_std_{window}'] = X[col].rolling(window, min_periods=1).std()
                X[f'{col}_rolling_max_{window}'] = X[col].rolling(window, min_periods=1).max()
                X[f'{col}_rolling_min_{window}'] = X[col].rolling(window, min_periods=1).min()
                X[f'{col}_rolling_skew_{window}'] = X[col].rolling(window, min_periods=1).skew()
                
                # Relative position features
                X[f'{col}_vs_rolling_mean_{window}'] = X[col] / (X[f'{col}_rolling_mean_{window}'] + 1e-6)
                X[f'{col}_vs_rolling_max_{window}'] = X[col] / (X[f'{col}_rolling_max_{window}'] + 1e-6)
        
        return X

    def _add_rolling_features(self, df):
        if FEATURE_GROUPS.get('ROLLING_FEATURES', False) and 'WorkType' in df.columns:
            for col in self.rolling_columns:
                if col in df.columns:
                    for window in self.rolling_windows:
                        rolling = df.groupby('WorkType')[col].rolling(window, min_periods=1)
                        df[f'{col}_rolling_mean_{window}'] = rolling.mean().reset_index(0, drop=True)
                        df[f'{col}_rolling_std_{window}'] = rolling.std().reset_index(0, drop=True)
        return df

    def _add_cyclical_features(self, df):
        if FEATURE_GROUPS.get('CYCLICAL_FEATURES', False):
            for feature, period in self.cyclical_features.items():
                if feature in df.columns:
                    df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / period)
                    df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / period)
        return df

    def _add_system_features(self, df):
        if FEATURE_GROUPS.get('PRODUCTIVITY_FEATURES', False):
            if 'SystemHours' not in df.columns:
                df['SystemHours'] = 8.0
        return df

    def _add_trend_features(self, df):
        if FEATURE_GROUPS.get('TREND_FEATURES', False) and 'WorkType' in df.columns:
            df = df.sort_values(['WorkType', 'Date'] if 'Date' in df.columns else ['WorkType'])
            
            # Process each trend column
            for col in self.trend_columns:
                if col not in df.columns:
                    continue
                    
                for window in self.trend_windows:
                    # Calculate trend slope
                    if self.trend_calculations.get('slope', True):
                        df[f'{col}_trend_slope_{window}'] = df.groupby('WorkType')[col].transform(
                            lambda x: self._calculate_trend_slope(x, window)
                        )
                    
                    # Calculate trend strength (R²)
                    if self.trend_calculations.get('strength', True):
                        df[f'{col}_trend_strength_{window}'] = df.groupby('WorkType')[col].transform(
                            lambda x: self._calculate_trend_strength(x, window)
                        )
                    
                    # Detrended values
                    if self.trend_calculations.get('detrended', True):
                        df[f'{col}_detrended_{window}'] = df.groupby('WorkType')[col].transform(
                            lambda x: self._calculate_detrended_values(x, window)
                        )
                
                # Trend change detection (if we have at least 2 windows)
                if self.trend_calculations.get('change', True) and len(self.trend_windows) >= 2:
                    for i in range(len(self.trend_windows) - 1):
                        w1, w2 = self.trend_windows[i], self.trend_windows[i + 1]
                        if f'{col}_trend_slope_{w1}' in df.columns and f'{col}_trend_slope_{w2}' in df.columns:
                            df[f'{col}_trend_change_{w1}_{w2}'] = (
                                df[f'{col}_trend_slope_{w2}'] - df[f'{col}_trend_slope_{w1}']
                            )
                
                # Trend acceleration
                if self.trend_calculations.get('acceleration', True) and self.trend_windows:
                    min_window = min(self.trend_windows)
                    if f'{col}_trend_slope_{min_window}' in df.columns:
                        df[f'{col}_trend_acceleration'] = df.groupby('WorkType')[f'{col}_trend_slope_{min_window}'].transform(
                            lambda x: x.diff()
                        )
            
            # Keep existing Cumulative_Quantity for backward compatibility
            if 'Quantity' in df.columns:
                df['Cumulative_Quantity'] = df.groupby('WorkType')['Quantity'].cumsum()
                
        return df

    def _add_pattern_features(self, df):
        if FEATURE_GROUPS.get('PATTERN_FEATURES', False) and 'Quantity' in df.columns:
            df = df.sort_values('Date' if 'Date' in df.columns else df.index)
            df['Quantity_3d_avg'] = df['Quantity'].rolling(window=3, min_periods=1).mean()
        return df

    def _add_interaction_features(self, df):
        if FEATURE_GROUPS.get('INTERACTION_FEATURES', False):
            if 'Quantity' in df.columns and 'SystemHours' in df.columns:
                df['Quantity_SystemHours'] = df['Quantity'] * df['SystemHours']
            if 'DayOfWeek' in df.columns and 'Month' in df.columns:
                df['DayOfWeek_Month'] = df['DayOfWeek'] * df['Month']
            if 'Year' in df.columns and 'Quarter' in df.columns:
                df['Year_Quarter'] = df['Year'] * df['Quarter']
        return df

    def _calculate_trend_slope(self, series, window):
        """Calculate trend slope using linear regression over window"""
        result = pd.Series(index=series.index, dtype=float)
        series_clean = series.ffill().bfill()  # CHANGED from fillna(method='ffill')
        
        for i in range(len(series_clean)):
            if i < window - 1:
                result.iloc[i] = 0.0
            else:
                y = series_clean.iloc[i-window+1:i+1].values
                x = np.arange(window)
                if len(y) == window and not np.all(np.isnan(y)):
                    # Simple linear regression
                    x_mean = x.mean()
                    y_mean = y.mean()
                    numerator = ((x - x_mean) * (y - y_mean)).sum()
                    denominator = ((x - x_mean) ** 2).sum()
                    slope = numerator / denominator if denominator != 0 else 0
                    result.iloc[i] = slope
                else:
                    result.iloc[i] = 0.0
        return result

    def _calculate_trend_strength(self, series, window):
        """Calculate R² of linear fit over window"""
        result = pd.Series(index=series.index, dtype=float)
        series_clean = series.ffill().bfill()
        
        for i in range(len(series_clean)):
            if i < window - 1:
                result.iloc[i] = 0.0
            else:
                y = series_clean.iloc[i-window+1:i+1].values
                x = np.arange(window)
                if len(y) == window and not np.all(np.isnan(y)):
                    # Calculate R²
                    x_mean = x.mean()
                    y_mean = y.mean()
                    ss_tot = ((y - y_mean) ** 2).sum()
                    if ss_tot > 0:
                        numerator = ((x - x_mean) * (y - y_mean)).sum()
                        denominator = ((x - x_mean) ** 2).sum()
                        if denominator > 0:
                            slope = numerator / denominator
                            intercept = y_mean - slope * x_mean
                            y_pred = slope * x + intercept
                            ss_res = ((y - y_pred) ** 2).sum()
                            r_squared = 1 - (ss_res / ss_tot)
                            result.iloc[i] = max(0, r_squared)  # Ensure non-negative
                        else:
                            result.iloc[i] = 0.0
                    else:
                        result.iloc[i] = 0.0
                else:
                    result.iloc[i] = 0.0
        return result

    def _calculate_detrended_values(self, series, window):
        """Calculate detrended values by removing linear trend"""
        result = pd.Series(index=series.index, dtype=float)
        series_clean = series.ffill().bfill()  # CHANGED from fillna(method='ffill')
        
        for i in range(len(series_clean)):
            if i < window - 1:
                result.iloc[i] = series_clean.iloc[i]
            else:
                y = series_clean.iloc[i-window+1:i+1].values
                x = np.arange(window)
                if len(y) == window and not np.all(np.isnan(y)):
                    # Fit linear trend
                    x_mean = x.mean()
                    y_mean = y.mean()
                    numerator = ((x - x_mean) * (y - y_mean)).sum()
                    denominator = ((x - x_mean) ** 2).sum()
                    if denominator > 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                        # Remove trend from current value
                        trend_value = slope * (window - 1) + intercept
                        result.iloc[i] = series_clean.iloc[i] - trend_value
                    else:
                        result.iloc[i] = 0.0
                else:
                    result.iloc[i] = series_clean.iloc[i]
        return result