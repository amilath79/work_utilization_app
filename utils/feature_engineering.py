import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import config
from utils.feature_selection import FeatureSelector

from config import (
    FEATURE_GROUPS, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS,
    LAG_FEATURES_COLUMNS, ROLLING_FEATURES_COLUMNS, 
    CYCLICAL_FEATURES, DATE_FEATURES, PRODUCTIVITY_FEATURES
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
        self.fitted_features_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.fitted_features_ = self._get_expected_features(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        X = self._add_date_features(X)
        X = self._add_lag_features(X)
        X = self._add_rolling_features(X)
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
            features.extend(['DayOfWeek', 'Month', 'WeekNo', 'IsWeekend', 'Quarter', 'Year', 'Day'])
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
            features.extend(['SystemHours', 'SystemKPI'])
            if hasattr(config, 'PRODUCTIVITY_FEATURES') and isinstance(PRODUCTIVITY_FEATURES, list):
                features.extend(PRODUCTIVITY_FEATURES)
        # Trend features
        if FEATURE_GROUPS.get('TREND_FEATURES', False):
            features.append('Cumulative_Quantity')
        # Pattern features
        if FEATURE_GROUPS.get('PATTERN_FEATURES', False):
            features.append('Quantity_3d_avg')
        # Interaction features
        if FEATURE_GROUPS.get('INTERACTION_FEATURES', False):
            features.extend(['Quantity_SystemHours', 'DayOfWeek_Month', 'Year_Quarter'])
        return features

    def _add_date_features(self, df):
        if FEATURE_GROUPS.get('DATE_FEATURES', False) and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
            df['Month'] = df['Date'].dt.month
            df['WeekNo'] = df['Date'].dt.isocalendar().week
            df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
            df['Quarter'] = df['Date'].dt.quarter
            df['Year'] = df['Date'].dt.year
            df['Day'] = df['Date'].dt.day
        return df

    def _add_lag_features(self, df):
        if FEATURE_GROUPS.get('LAG_FEATURES', False) and 'WorkType' in df.columns:
            df = df.sort_values(['WorkType', 'Date'] if 'Date' in df.columns else ['WorkType'])
            for col in self.lag_columns:
                if col in df.columns:
                    for lag in self.lag_days:
                        df[f'{col}_lag_{lag}'] = df.groupby('WorkType')[col].shift(lag)
        return df

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
            if 'SystemKPI' not in df.columns:
                df['SystemKPI'] = 1.0
        return df

    def _add_trend_features(self, df):
        if FEATURE_GROUPS.get('TREND_FEATURES', False) and 'Quantity' in df.columns:
            df = df.sort_values('Date' if 'Date' in df.columns else df.index)
            df['Cumulative_Quantity'] = df['Quantity'].cumsum()
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