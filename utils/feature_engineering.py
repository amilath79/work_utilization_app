from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging
import config

# Import config-driven parameters (same as your create_enhanced_features)
from config import (
    FEATURE_GROUPS, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS,
    LAG_FEATURES_COLUMNS, ROLLING_FEATURES_COLUMNS, 
    CYCLICAL_FEATURES, DATE_FEATURES, PRODUCTIVITY_FEATURES
)


logger = logging.getLogger(__name__)

def create_lag_features(df, group_col='WorkType', target_col='Hours', lag_days=None, rolling_windows=None):
    """
    Create lag features while preserving essential columns for UI
    
    Parameters:
    -----------
    df : pd.DataFramex§
        Input dataframe with Date, WorkType, Hours columns
    group_col : str
        Column to group by (default: 'WorkType')  
    target_col : str
        Target column for lag features (default: 'Hours')  # CHANGED
    """
    try:
        logger.info("🔧 Creating lag features while preserving UI columns")
        
        # Set defaults
        if lag_days is None:
            lag_days = ESSENTIAL_LAGS
        if rolling_windows is None:
            rolling_windows = ESSENTIAL_WINDOWS
            
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Ensure Date is datetime
        if 'Date' in result_df.columns:
            result_df['Date'] = pd.to_datetime(result_df['Date'])
            
        # Sort by group and date for proper lag calculation
        if 'Date' in result_df.columns and group_col in result_df.columns:
            result_df = result_df.sort_values([group_col, 'Date'])
        
        # Add lag features for the target column
        if FEATURE_GROUPS.get('LAG_FEATURES', False) and target_col in result_df.columns:
            for lag in lag_days:
                result_df[f'{target_col}_lag_{lag}'] = result_df.groupby(group_col)[target_col].shift(lag)
                
        # Add rolling features for the target column  
        if FEATURE_GROUPS.get('ROLLING_FEATURES', False) and target_col in result_df.columns:
            for window in rolling_windows:
                rolling_group = result_df.groupby(group_col)[target_col].rolling(window, min_periods=1)
                result_df[f'{target_col}_rolling_mean_{window}'] = rolling_group.mean().reset_index(0, drop=True)
                result_df[f'{target_col}_rolling_std_{window}'] = rolling_group.std().reset_index(0, drop=True)
        
        # Add lag features for other configured columns
        if FEATURE_GROUPS.get('LAG_FEATURES', False):
            lag_columns = LAG_FEATURES_COLUMNS if hasattr(config, 'LAG_FEATURES_COLUMNS') else []
            for col in lag_columns:
                if col in result_df.columns and col != target_col:
                    for lag in lag_days:
                        result_df[f'{col}_lag_{lag}'] = result_df.groupby(group_col)[col].shift(lag)
        
        # Add rolling features for other configured columns
        if FEATURE_GROUPS.get('ROLLING_FEATURES', False):
            rolling_columns = ROLLING_FEATURES_COLUMNS if hasattr(config, 'ROLLING_FEATURES_COLUMNS') else []
            for col in rolling_columns:
                if col in result_df.columns and col != target_col:
                    for window in rolling_windows:
                        rolling_group = result_df.groupby(group_col)[col].rolling(window, min_periods=1)
                        result_df[f'{col}_rolling_mean_{window}'] = rolling_group.mean().reset_index(0, drop=True)
                        result_df[f'{col}_rolling_std_{window}'] = rolling_group.std().reset_index(0, drop=True)
        
        # Add basic date features if enabled
        if FEATURE_GROUPS.get('DATE_FEATURES', False) and 'Date' in result_df.columns:
            result_df['DayOfWeek_feat'] = result_df['Date'].dt.dayofweek
            result_df['Month_feat'] = result_df['Date'].dt.month  
            result_df['IsWeekend_feat'] = (result_df['Date'].dt.dayofweek >= 5).astype(int)
            result_df['DayOfMonth'] = result_df['Date'].dt.day
            result_df['Quarter'] = result_df['Date'].dt.quarter
            result_df['Year'] = result_df['Date'].dt.year
            result_df['Day'] = result_df['Date'].dt.day
            
        # Fill NaN values created by lag/rolling operations
        # Smart NaN handling based on feature type
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if result_df[col].isna().any():
                if 'lag' in col:
                    # For lag features, forward fill then use median
                    result_df[col] = result_df.groupby('WorkType')[col].fillna(method='ffill').fillna(
                        result_df.groupby('WorkType')[col].median()
                    )
                elif 'rolling' in col:
                    # For rolling features, use expanding mean until enough data
                    result_df[col] = result_df.groupby('WorkType')[col].fillna(
                        result_df.groupby('WorkType')[col.replace('rolling', 'expanding')].mean()
                    )
                else:
                    # For other features, use median by worktype
                    result_df[col] = result_df.groupby('WorkType')[col].fillna(
                        result_df.groupby('WorkType')[col].median()
                ).fillna(0)  # Final fallback
        
        logger.info(f"✅ Created lag features. Shape: {result_df.shape}")
        logger.info(f"📊 Columns include: Date={('Date' in result_df.columns)}, WorkType={('WorkType' in result_df.columns)}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error in create_lag_features: {str(e)}")
        logger.error(traceback.format_exc())
        return df  # Return original on error

class EnhancedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Config-driven transformer for enhanced feature engineering
    Uses the same config-driven approach as create_enhanced_features()
    Follows sklearn transformer pattern for seamless pipeline integration
    """
    
    def __init__(self):
        # Read parameters from config file (same as your approach)
        self.lag_days = ESSENTIAL_LAGS if FEATURE_GROUPS.get('LAG_FEATURES', False) else []
        self.rolling_windows = ESSENTIAL_WINDOWS if FEATURE_GROUPS.get('ROLLING_FEATURES', False) else []
        self.lag_columns = LAG_FEATURES_COLUMNS if hasattr(config, 'LAG_FEATURES_COLUMNS') else ['Hours']
        self.rolling_columns = ROLLING_FEATURES_COLUMNS if hasattr(config, 'ROLLING_FEATURES_COLUMNS') else ['Hours']
        self.cyclical_features = CYCLICAL_FEATURES if FEATURE_GROUPS.get('CYCLICAL_FEATURES', False) else {}
        self.fitted_features_ = None
        
        # Log active feature groups (same as your approach)
        enabled_groups = [k for k, v in FEATURE_GROUPS.items() if v]
        logger.info(f"📊 Config-driven EnhancedFeatureTransformer - Active Feature Groups: {enabled_groups}")
        
    def fit(self, X, y=None):
        # ensure DataFrame
        X = pd.DataFrame(X).copy()
        # remember how many days back we’ll need
        self.max_lag = max(self.lag_days or [0])
        # store the end of each series for each WorkType
        self._history = (
            X.sort_values(['WorkType','Date'])
            .groupby('WorkType')
            .tail(self.max_lag)
            .reset_index(drop=True)
        )
        self.fitted_features_ = self._get_expected_features(X)
        return self
    
    def transform(self, X):
        """
        Transform the data by applying config-driven enhanced feature engineering.
        Prepends the last max_lag rows from fit() so that early lags/windows are correct.
        """
        # 1) Ensure DataFrame and reset index
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy().reset_index(drop=True)

        # 2) Concatenate history (from fit) + new rows
        #    (history has the last self.max_lag rows per WorkType)
        full = pd.concat([self._history, X], ignore_index=True)

        # 3) Apply all FE steps to the full series
        full = self._add_date_features(full)
        full = self._add_lag_features(full)
        full = self._add_rolling_features(full)
        full = self._add_cyclical_features(full)
        full = self._add_system_features(full)
        full = self._add_trend_features(full)
        full = self._add_pattern_features(full)
        full = self._add_interaction_features(full)

        # 4) Extract just the transformed “new” rows
        transformed = full.iloc[-len(X):].reset_index(drop=True)

        # 5) Ensure every expected feature exists
        for feat in self.fitted_features_:
            if feat not in transformed.columns:
                transformed[feat] = 0.0

        # 6) Preserve raw columns the model may need (e.g. Hours) 
        essential = [c for c in ['WorkType', 'Quantity', 'Hours'] if c in transformed.columns]

        # 6.5) Encode WorkType if present
        if 'WorkType' in transformed.columns:
            transformed['WorkType'] = transformed['WorkType'].astype('category').cat.codes

        # 7) Final column ordering: essentials first, then all fitted_features_
        cols = essential + [f for f in self.fitted_features_ if f not in essential]
        return transformed[cols].fillna(0)

    
    def _get_expected_features(self, X):
        """
        Get the list of features this transformer will create
        Config-driven approach (same as your create_enhanced_features)
        """
        features = []
        
        # Date features (config-driven)
        if FEATURE_GROUPS.get('DATE_FEATURES', False):
            features.extend(['DayOfWeek', 'Month', 'WeekNo', 'IsWeekend', 'Quarter', 'Year', 'Day'])
            # Add categorical date features if defined in config
            if hasattr(config, 'DATE_FEATURES') and isinstance(DATE_FEATURES, dict):
                features.extend(DATE_FEATURES.get('categorical', []))
                features.extend(DATE_FEATURES.get('numeric', []))
        
        # Lag features (config-driven)
        if FEATURE_GROUPS.get('LAG_FEATURES', False):
            for col in self.lag_columns:
                for lag in self.lag_days:
                    features.append(f'{col}_lag_{lag}')
            
        # Rolling features (config-driven)
        if FEATURE_GROUPS.get('ROLLING_FEATURES', False):
            for col in self.rolling_columns:
                for window in self.rolling_windows:
                    features.append(f'{col}_rolling_mean_{window}')
                    # Add rolling std if your config includes it
                    features.append(f'{col}_rolling_std_{window}')
        
        # Cyclical features (config-driven)
        if FEATURE_GROUPS.get('CYCLICAL_FEATURES', False):
            for feature, period in self.cyclical_features.items():
                features.extend([f'{feature}_sin', f'{feature}_cos'])
        
        # Productivity features (config-driven)
        if FEATURE_GROUPS.get('PRODUCTIVITY_FEATURES', False):
            features.extend(['SystemHours', 'SystemKPI'])
            # Add other productivity features from config if available
            if hasattr(config, 'PRODUCTIVITY_FEATURES') and isinstance(PRODUCTIVITY_FEATURES, list):
                features.extend(PRODUCTIVITY_FEATURES)
        
        # Trend features (config-driven)
        if FEATURE_GROUPS.get('TREND_FEATURES', False):
            features.extend(['Cumulative_Quantity'])  # Example from your pattern
        
        # Pattern features (config-driven)  
        if FEATURE_GROUPS.get('PATTERN_FEATURES', False):
            features.extend(['Quantity_3d_avg'])  # Example from your pattern

            
        
        # Keep original features that might be needed
        original_features = []
        for feat in original_features:
            if feat in X.columns:
                features.append(feat)
        
        return features
    
    def _add_date_features(self, df):
        """Add date-based features (config-driven)"""
        if FEATURE_GROUPS.get('DATE_FEATURES', False):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
                df['Month'] = df['Date'].dt.month
                df['WeekNo'] = df['Date'].dt.isocalendar().week
                df['IsWeekend'] = (df['Date'].dt.dayofweek >= 5).astype(int)
                df['Quarter'] = df['Date'].dt.quarter
                df['Year'] = df['Date'].dt.year
                df['Day'] = df['Date'].dt.day
            else:
                # For prediction, use current date if Date not provided
                from datetime import datetime
                current_date = datetime.now()
                df['DayOfWeek'] = current_date.weekday() + 1
                df['Month'] = current_date.month
                df['WeekNo'] = current_date.isocalendar().week
                df['IsWeekend'] = 1 if current_date.weekday() >= 5 else 0
                df['Quarter'] = current_date.quarter
                df['Year'] = current_date.year
                df['Day'] = df['Date'].dt.day
            
        return df
    
    def _add_lag_features(self, df):
        """Add lag features (config-driven)"""
        if FEATURE_GROUPS.get('LAG_FEATURES', False):
            # Support both Hours (new) and NoOfMan (legacy)
            target_col = 'Hours' if 'Hours' in df.columns else 'NoOfMan'
            if target_col in df.columns and 'WorkType' in df.columns:
                df = df.sort_values(['WorkType', 'Date'] if 'Date' in df.columns else ['WorkType'])
                
                for col in self.lag_columns:
                    if col in df.columns:
                        for lag in self.lag_days:
                            df[f'{col}_lag_{lag}'] = df.groupby('WorkType')[col].shift(lag)
            
            if 'Date' in df.columns:
                df['DayOfWeek_num'] = df['Date'].dt.dayofweek
                for col in self.lag_columns:
                    if col in df.columns:
                        for weeks_back in [1, 2, 4]:  # Same weekday 1,2,4 weeks ago
                            df[f'{col}_same_dow_{weeks_back}w'] = (
                                df.groupby(['WorkType', 'DayOfWeek_num'])[col]
                                .shift(weeks_back)
                            )
                df = df.drop('DayOfWeek_num', axis=1)  # Clean up temp column

            else:
                # For prediction, fill with reasonable defaults
                for col in self.lag_columns:
                    for lag in self.lag_days:
                        df[f'{col}_lag_{lag}'] = 0
                        
        return df
    
    def _add_rolling_features(self, df):
        """Add rolling window features (config-driven)"""
        if FEATURE_GROUPS.get('ROLLING_FEATURES', False):
            if 'NoOfMan' in df.columns and 'WorkType' in df.columns:
                for col in self.rolling_columns:
                    if col in df.columns:
                        for window in self.rolling_windows:
                            rolling = df.groupby('WorkType')[col].rolling(window, min_periods=1)
                            df[f'{col}_rolling_mean_{window}'] = rolling.mean().reset_index(0, drop=True)
                            df[f'{col}_rolling_std_{window}'] = rolling.std().reset_index(0, drop=True)
            else:
                # For prediction, fill with reasonable defaults
                for col in self.rolling_columns:
                    for window in self.rolling_windows:
                        df[f'{col}_rolling_mean_{window}'] = 0
                        df[f'{col}_rolling_std_{window}'] = 0
                        
        return df
    
    def _add_cyclical_features(self, df):
        """Add cyclical encoding for temporal features (config-driven)"""
        if FEATURE_GROUPS.get('CYCLICAL_FEATURES', False):
            for feature, period in self.cyclical_features.items():
                if feature in df.columns:
                    df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / period)
                    df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / period)
                    
        return df
    
    def _add_system_features(self, df):
        """Add system-related features (config-driven)"""
        if FEATURE_GROUPS.get('PRODUCTIVITY_FEATURES', False):
            if 'SystemHours' not in df.columns:
                df['SystemHours'] = 8.0  # Default working hours
                
            if 'SystemKPI' not in df.columns:
                df['SystemKPI'] = 1.0  # Default KPI
                
        return df
    
    def _add_trend_features(self, df):
        """Add trend features (config-driven)"""
        if FEATURE_GROUPS.get('TREND_FEATURES', False):
            if 'Quantity' in df.columns:
                df = df.sort_values('Date' if 'Date' in df.columns else df.index)
                df['Cumulative_Quantity'] = df['Quantity'].cumsum()
        return df
    
    # def _add_pattern_features(self, df):
    #     """Add pattern features (config-driven)"""
    #     if FEATURE_GROUPS.get('PATTERN_FEATURES', False):
    #         if 'Quantity' in df.columns:
    #             df = df.sort_values('Date' if 'Date' in df.columns else df.index)
    #             df['Quantity_3d_avg'] = df['Quantity'].rolling(window=3, min_periods=1).mean()
    #     return df

    def _add_pattern_features(self, df):
        """Enhanced pattern features with config-driven logic and punch-code specificity"""
        df = df.copy()

        # Skip if PATTERN_FEATURES is not explicitly False
        if FEATURE_GROUPS.get('PATTERN_FEATURES') is not False:
            return df

        # Config-driven pattern features
        if 'Quantity' in df.columns:
            df = df.sort_values('Date' if 'Date' in df.columns else df.index)
            df['Quantity_3d_avg'] = df['Quantity'].rolling(window=3, min_periods=1).mean()

        # Special features for problematic punch codes
        for punch_code in [210, 217]:
            punch_mask = df['punch_code'] == punch_code
            if punch_mask.any():
                grouped = df.loc[punch_mask].groupby('punch_code')['hours']
                df.loc[punch_mask, f'volatility_{punch_code}'] = (
                    grouped.rolling(7).std().reset_index(0, drop=True)
                )
                df.loc[punch_mask, f'stability_{punch_code}'] = (
                    grouped.rolling(14)
                    .apply(lambda x: x.std() / (x.mean() + 1e-8))
                    .reset_index(0, drop=True)
                )

        return df.fillna(0)
    
    def _add_interaction_features(self, df):
        """Add interaction features for complex patterns"""
        if FEATURE_GROUPS.get('INTERACTION_FEATURES', False):
            # DayOfWeek × WorkType interactions (workforce varies by day and type)
            if 'DayOfWeek' in df.columns and 'WorkType' in df.columns:
                # Create worktype-specific day patterns
                for work_type in df['WorkType'].unique():
                    wt_mask = df['WorkType'] == work_type
                    df.loc[wt_mask, f'DayPattern_{work_type}'] = df.loc[wt_mask, 'DayOfWeek']
            
            # Month × Quantity interaction (seasonal workload)
            if 'Month' in df.columns and 'Quantity' in df.columns:
                df['Month_Quantity_interaction'] = df['Month'] * df['Quantity'] / 12
            
            # Weekend × WorkType (different weekend patterns per type)
            if 'IsWeekend' in df.columns and 'Hours' in df.columns:
                df['Weekend_Hours_ratio'] = df['IsWeekend'] * df['Hours']
        
        return df
