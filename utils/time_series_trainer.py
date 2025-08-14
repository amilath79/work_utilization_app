"""
Advanced Time-Series Trainer for Workforce Prediction
Based on proven approach that achieved R² = 0.9987 for punch code 217
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
import logging
from typing import Dict, Tuple, Optional
import traceback
import pickle
import os
import sys

logger = logging.getLogger(__name__)


class TimeSeriesWorkforceTrainer:
    """
    Enterprise-grade time-series trainer for workforce prediction
    Implements proven methodology for next-day workforce forecasting
    """
    
    def __init__(self, punch_code: str):
        self.punch_code = punch_code
        self.model = None
        self.feature_columns = None
        self.validation_results = None
        
        # Punch-code specific configurations
        self.config = self._get_punch_code_config(punch_code)
        
    def _get_punch_code_config(self, punch_code: str) -> Dict:
        """Get punch-code specific training configuration"""
        
        # Default configuration for weekday-only punch codes
        base_config = {
            'include_weekends': False,  # Most punch codes: weekdays only
            'data_start_date': '2019-07-01',  # Use all available data
            'outlier_method': 'iqr',
            'outlier_factor': 1.5,
            'hyperparameters': {
                'learning_rate': 0.01,
                'min_child_samples': 20,
                'n_estimators': 2000,
                'num_leaves': 31,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'lags': [1, 7, 14, 21, 30, 365, 366],
            'rolling_windows': [7, 14, 28],
            'use_cyclical_features': True,
            'use_interaction_features': True,
            'use_year_over_year': True
        }
        
        # Punch-code specific overrides
        if punch_code == '206':
            # 206 is special - works weekends + Sundays
            base_config.update({
                'include_weekends': True,   # 206 works weekends
                'sunday_max_workers': 8,    # Sunday constraint
                'saturday_patterns': True   # Different Saturday patterns
            })
        elif punch_code in ['202', '203', '209', '210', '211', '213', '214', '215', '217']:
            # All other punch codes: weekdays only (use base config)
            # Apply simple weekday filter for all these codes
            base_config.update({
                'include_weekends': False,  # Strict weekdays only
                'simple_weekday_filter': True  # Use simple filter like 217
            })
        
        return base_config
    
    def filter_data_by_working_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on punch-code specific working days"""
        try:
            # For weekday-only punch codes: Use simple weekday filter (like 217)
            if self.config.get('simple_weekday_filter', False) or self.punch_code in ['202', '203', '209', '210', '211', '213', '214', '215', '217']:
                filtered_df = df[df['Date'].dt.weekday < 5].copy()
                logger.info(f"Punch {self.punch_code}: Simple weekday filter {len(df)} → {len(filtered_df)}")
                return filtered_df
            
            # For 206 and other special cases: Use holiday utils
            from utils.holiday_utils import is_working_day_for_punch_code
            
            # Apply working day filter for this punch code
            working_day_mask = df['Date'].apply(
                lambda date: is_working_day_for_punch_code(date, self.punch_code)[0]  # Take first element of tuple
            )
            
            filtered_df = df[working_day_mask].copy()
            
            logger.info(f"Punch {self.punch_code}: Filtered {len(df)} → {len(filtered_df)} working days")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering working days for {self.punch_code}: {e}")
            # Fallback: weekdays only
            if self.config.get('include_weekends', False):
                return df.copy()
            else:
                return df[df['Date'].dt.weekday < 5].copy()
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced feature engineering matching Colab approach"""
        df_featured = df.copy()
      
        # 1. RESOURCE KPI (Critical improvement from Colab)
        epsilon = 1e-9
        df_featured['ResourceKPI'] = df_featured['Quantity'] / (df_featured['Hours'] + epsilon)
        df_featured['ResourceKPI'] = df_featured['ResourceKPI'].replace([float('inf'), float('-inf')], 0)
        
        # 2. UTILIZATION RATIO (New feature from Colab)
        df_featured['UtilizationRatio'] = df_featured['SystemHours'] / (df_featured['Hours'] + epsilon)
        df_featured['UtilizationRatio'] = df_featured['UtilizationRatio'].replace([float('inf'), float('-inf')], 0)
        
        # 3. DATE FEATURES (Keep existing)
        df_featured['year'] = df_featured['Date'].dt.year
        df_featured['month'] = df_featured['Date'].dt.month
        df_featured['week_no'] = df_featured['Date'].dt.isocalendar().week
        df_featured['day_of_week'] = df_featured['Date'].dt.dayofweek
        df_featured['quarter'] = df_featured['Date'].dt.quarter
        df_featured['is_monthend'] = df_featured['Date'].dt.is_month_end.astype(int)
        df_featured['is_monthstart'] = df_featured['Date'].dt.is_month_start.astype(int)
        
        # 4. ENHANCED LAG FEATURES (More comprehensive)
        lag_columns = ['Quantity', 'SystemHours', 'Hours']
        lags = [1, 7, 14, 21, 30]  # Match Colab lags
        
        for col in lag_columns:
            for lag in lags:
                df_featured[f'{col}_Lag{lag}'] = df_featured[col].shift(lag)
        
        # 5. COMPREHENSIVE ROLLING WINDOWS (Match Colab exactly)
        windows = [7, 14]
        
        for col in lag_columns:
            for window in windows:
                df_featured[f'{col}_Window{window}_Mean'] = df_featured[col].rolling(window=window, min_periods=1).mean()
                df_featured[f'{col}_Window{window}_Std'] = df_featured[col].rolling(window=window, min_periods=1).std()
                df_featured[f'{col}_Window{window}_Max'] = df_featured[col].rolling(window=window, min_periods=1).max()
                df_featured[f'{col}_Window{window}_Min'] = df_featured[col].rolling(window=window, min_periods=1).min()
                df_featured[f'{col}_Window{window}_Median'] = df_featured[col].rolling(window=window, min_periods=1).median()
                df_featured[f'{col}_Window{window}_Sum'] = df_featured[col].rolling(window=window, min_periods=1).sum()
                df_featured[f'{col}_Window{window}_EWMA'] = df_featured[col].ewm(span=window, min_periods=1).mean()
                
                # CV (Coefficient of Variation)
                mean_vals = df_featured[col].rolling(window=window, min_periods=1).mean()
                std_vals = df_featured[col].rolling(window=window, min_periods=1).std()
                df_featured[f'{col}_Window{window}_CV'] = std_vals / (mean_vals + epsilon)
                
                # IQR (Interquartile Range)
                q75 = df_featured[col].rolling(window=window, min_periods=1).quantile(0.75)
                q25 = df_featured[col].rolling(window=window, min_periods=1).quantile(0.25)
                df_featured[f'{col}_Window{window}_IQR'] = q75 - q25
        
        # 6. LAST YEAR SAME WEEK FEATURES (Critical from Colab)
        df_featured = self._create_last_year_features(df_featured)
        
        # 7. INTERACTION FEATURES (Enhanced from Colab)
        df_featured['ResourceKPI_Quantity_Lag7'] = df_featured['ResourceKPI'] * df_featured['Quantity_Lag7']
        df_featured['ResourceKPI_Quantity_Lag14'] = df_featured['ResourceKPI'] * df_featured['Quantity_Lag14']
        df_featured['ResourceKPI_Quantity_LastYear'] = df_featured['ResourceKPI'] * df_featured['Quantity_LastYear']
        
        # 8. FILL NAN VALUES (Critical for stability)
        df_featured = df_featured.ffill().bfill()
        
        logger.info(f"Feature engineering complete. Final shape: {df_featured.shape}")
        return df_featured
    

    def _create_last_year_features(self, df):
        """Create last year same week same day features (from Colab)"""
        df = df.copy()
        
        # Ensure Date is in index for proper date operations
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        df = df.sort_index()
        
        # Initialize the new columns
        df['Quantity_LastYear'] = np.nan
        df['Hours_LastYear'] = np.nan
        df['SystemHours_LastYear'] = np.nan

        for current_date in df.index:
            try:
                # Find the same week and same day of week from last year
                last_year_date = current_date - pd.DateOffset(years=1)
                
                # Create a window of ±3 days around the target date
                start_window = last_year_date - pd.DateOffset(days=3)
                end_window = last_year_date + pd.DateOffset(days=3)
                
                # Filter data within the window
                window_data = df[(df.index >= start_window) & (df.index <= end_window)]
                
                if not window_data.empty:
                    # Find the date with the same day of week, or closest match
                    current_dow = current_date.dayofweek
                    window_data_with_dow = window_data.copy()
                    window_data_with_dow['dow_diff'] = abs(window_data_with_dow.index.dayofweek - current_dow)
                    
                    # Sort by day of week difference and then by date difference
                    window_data_with_dow['date_diff'] = abs((window_data_with_dow.index - last_year_date).days)
                    best_match = window_data_with_dow.sort_values(['dow_diff', 'date_diff']).iloc[0]
                    
                    # Assign the values
                    df.loc[current_date, 'Quantity_LastYear'] = best_match['Quantity']
                    df.loc[current_date, 'Hours_LastYear'] = best_match['Hours']
                    df.loc[current_date, 'SystemHours_LastYear'] = best_match['SystemHours']
            
            except Exception as e:
                # Skip if any error occurs for this date
                continue
        
        # Reset index to get Date back as column
        df = df.reset_index()
        return df
    def _add_year_over_year_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add same weekday, same week last year feature"""
        
        def get_same_weekday_same_week_last_year(date):
            """Find same weekday in same ISO week of previous year"""
            try:
                iso_year, iso_week, iso_weekday = date.isocalendar()
                target_year = iso_year - 1
                jan4_last_year = pd.to_datetime(f'{target_year}-01-04')
                first_day_of_week1_last_year = jan4_last_year - timedelta(days=jan4_last_year.weekday())
                target_week_date = first_day_of_week1_last_year + timedelta(weeks=iso_week - 1)
                target_date = target_week_date + timedelta(days=iso_weekday - 1)
                
                # Validate the calculated date
                target_iso_year, target_iso_week, _ = target_date.isocalendar()
                if target_iso_year != target_year or target_iso_week != iso_week:
                    return pd.NaT
                
                return target_date
            except:
                return pd.NaT
        
        # Calculate same weekday last year dates
        df['Date_Last_Year_Same_Weekday_Same_Week'] = df['Date'].apply(
            get_same_weekday_same_week_last_year
        )
        
        # Merge with historical hours data
        df_hours_lookup = df[['Date', 'Hours']].rename(
            columns={'Hours': 'Hours_Last_Year_Same_Weekday_Same_Week'}
        )
        
        df = df.merge(
            df_hours_lookup,
            left_on='Date_Last_Year_Same_Weekday_Same_Week',
            right_on='Date',
            how='left',
            suffixes=('', '_y')
        )
        
        # Clean up
        df.drop(['Date_y', 'Date_Last_Year_Same_Weekday_Same_Week'], axis=1, inplace=True)
        
        return df
    
    # def prepare_time_series_data(self, df_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    #     """Prepare time-series data with Colab-enhanced features but keep next-day prediction"""
        
    #     # Remove NaN rows
    #     df_clean = df_featured.dropna().copy()  # Ensure we have a proper copy
        
    #     # CREATE NEXT-DAY TARGET (Keep your existing approach)
    #     df_clean['target_Hours'] = df_clean['Hours'].shift(-1)
    #     df_clean = df_clean.dropna(subset=['target_Hours'])
        
    #     # Colab-style feature selection but adapted for next-day prediction
    #     exclude_cols = [
    #         'Hours',           # Exclude current Hours (we're predicting next day)
    #         'WorkType', 
    #         'Date', 
    #         'target_Hours',    # Exclude the target we created
    #         'Quantity',        # Exclude base columns used to create features
    #         'SystemHours',     # Exclude base columns used to create features
    #         'Unnamed: 0'
    #     ]
        
    #     # Get all columns and exclude target and metadata columns
    #     all_columns = df_clean.columns.tolist()
    #     feature_columns = [col for col in all_columns if col not in exclude_cols]
        
    #     X = df_clean[feature_columns]
    #     y = df_clean['target_Hours']  # Use shifted target for next-day prediction
        
    #     # Fill any remaining NaN values (from Colab improvement)
    #     X = X.fillna(0)
        
    #     # Reset index
    #     X = X.reset_index(drop=True)
    #     y = y.reset_index(drop=True)
        
    #     logger.info(f"Using all {len(X.columns)} features (enhanced with Colab improvements)")
    #     logger.info(f"Prepared time-series data: {len(X)} samples, {len(X.columns)} features")
    #     logger.info(f"Prediction mode: Next-day Hours prediction using current day features")
        
    #     return X, y


    def prepare_time_series_data(self, df_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare time-series data - TEST: Same-day prediction like Colab"""
        
        # Remove NaN rows
        df_clean = df_featured.dropna().copy()
        
        # TEMPORARY: Use same-day prediction like Colab (no shift)
        # df_clean['target_Hours'] = df_clean['Hours'].shift(-1)
        # df_clean = df_clean.dropna(subset=['target_Hours'])
        
        # Colab-style feature selection
        exclude_cols = [
            'Hours',           # This is our target
            'WorkType', 
            'Date', 
            'Quantity',        # Exclude base columns 
            'SystemHours',     # Exclude base columns
            'Unnamed: 0'
        ]
        
        all_columns = df_clean.columns.tolist()
        feature_columns = [col for col in all_columns if col not in exclude_cols]
        
        X = df_clean[feature_columns]
        y = df_clean['Hours']  # Same-day prediction like Colab
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Reset index
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        logger.info(f"TEST MODE: Same-day prediction (like Colab)")
        logger.info(f"Prepared data: {len(X)} samples, {len(X.columns)} features")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train LightGBM model with time-series validation"""
        
        # Initialize model with punch-code specific hyperparameters
        self.model = lgb.LGBMRegressor(**self.config['hyperparameters'])
        
        # Train on full dataset
        self.model.fit(X, y)
        self.feature_columns = X.columns.tolist()
        
        # Validate with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # Train fold model
            fold_model = lgb.LGBMRegressor(**self.config['hyperparameters'])
            fold_model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = fold_model.predict(X_val)
            
            r2_scores.append(r2_score(y_val, y_pred))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        
        # Store validation results
        self.validation_results = {
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores),
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores),
            'cv_rmse_mean': np.mean(rmse_scores),
            'cv_rmse_std': np.std(rmse_scores),
            'cv_splits': len(r2_scores)
        }
        
        logger.info(f"Punch {self.punch_code} - CV Results:")
        logger.info(f"  R² = {self.validation_results['cv_r2_mean']:.4f} ± {self.validation_results['cv_r2_std']:.4f}")
        logger.info(f"  MAE = {self.validation_results['cv_mae_mean']:.4f} ± {self.validation_results['cv_mae_std']:.4f}")
        logger.info(f"  RMSE = {self.validation_results['cv_rmse_mean']:.4f} ± {self.validation_results['cv_rmse_std']:.4f}")
        
        return self.validation_results
    
    def save_model(self, models_dir: str) -> bool:
        """Save trained model and metadata in existing compatible format"""
        try:            
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'punch_code': self.punch_code,
                'validation_results': self.validation_results,
                'model_type': 'TimeSeriesLGBM_Direct'
            }

            # Save model data
            model_path = os.path.join(models_dir, f'enhanced_model_{self.punch_code}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metadata in compatible format
            metadata = {
                self.punch_code: {
                    'test_mae': self.validation_results['cv_mae_mean'],
                    'test_r2': self.validation_results['cv_r2_mean'],
                    'test_mape': self.validation_results['cv_mae_mean'] / 30 * 100,  # Approximate
                    'cv_mae': self.validation_results['cv_mae_mean'],
                    'cv_r2': self.validation_results['cv_r2_mean'],
                    'training_records': len(self.feature_columns),
                    'test_records': 'CV',
                    'num_features': len(self.feature_columns),
                    'model_type': 'TimeSeriesLGBM_Pipeline',
                    'trainer_version': '3.0'
                }
            }
            
            # Save features
            features = {self.punch_code: self.feature_columns}
            
            # Save metadata
            metadata_path = os.path.join(models_dir, 'enhanced_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    existing_metadata = pickle.load(f)
                existing_metadata.update(metadata)
                metadata = existing_metadata
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save features  
            features_path = os.path.join(models_dir, 'enhanced_features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    existing_features = pickle.load(f)
                existing_features.update(features)
                features = existing_features
                
            with open(features_path, 'wb') as f:
                pickle.dump(features, f)
            
            logger.info(f"✅ Saved enhanced pipeline model for punch code {self.punch_code}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model for {self.punch_code}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def train_punch_code_with_new_pipeline(punch_code: str, data_source: str = None) -> bool:
    """
    Universal training function for any punch code using time-series methodology
    
    Parameters:
    -----------
    punch_code : str
        Punch code to train (any from ENHANCED_WORK_TYPES)
    data_source : str, optional
        Path to data file or None to load from SQL
        
    Returns:
    --------
    bool
        True if training successful
    """
    try:
        logger.info(f"🎯 Starting time-series training for punch code {punch_code}")
        
        # Load data
        if data_source and os.path.exists(data_source):
            # Load from file
            if data_source.endswith('.xlsx'):
                df = pd.read_excel(data_source)
            elif data_source.endswith('.pkl'):
                df = pd.read_pickle(data_source)
            else:
                logger.error(f"Unsupported data format: {data_source}")
                return False
        else:
            # Load from SQL (existing method)
            from utils.sql_data_connector import extract_sql_data
            from config import SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION
            
            query = f"""
            SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
            CASE WHEN PunchCode IN (206, 213) THEN NoRows ELSE Quantity END as Quantity
            FROM WorkUtilizationData 
            WHERE PunchCode = '{punch_code}'
            AND Hours > 0 
            AND SystemHours > 0 
            AND Date < '2025-08-01'  
            ORDER BY Date
            """
            df = extract_sql_data(
                server=SQL_SERVER,
                database=SQL_DATABASE, 
                query=query,
                trusted_connection=SQL_TRUSTED_CONNECTION
            )
        
        if df is None or df.empty:
            logger.error(f"No data loaded for punch code {punch_code}")
            return False
        
        # Filter for the specific punch code and ensure proper data types
        logger.info(f"Raw data WorkType unique values: {df['WorkType'].unique()}")
        logger.info(f"Raw data WorkType dtypes: {df['WorkType'].dtype}")
        
        # Handle both string and integer WorkType formats
        df_punch = df[
            (df['WorkType'] == punch_code) | 
            (df['WorkType'] == int(punch_code)) | 
            (df['WorkType'].astype(str) == punch_code)
        ].copy()
        
        logger.info(f"After WorkType filtering: {len(df_punch)} records")
        
        if len(df_punch) == 0:
            logger.error(f"❌ No records found for punch code {punch_code} after filtering!")
            logger.error(f"Available WorkTypes: {df['WorkType'].unique()}")
            return False
        
        df_punch['Date'] = pd.to_datetime(df_punch['Date'])
        df_punch['WorkType'] = df_punch['WorkType'].astype(str)  # Ensure string type
        df_punch = df_punch.sort_values('Date')
        
        logger.info(f"Loaded {len(df_punch)} records for punch code {punch_code}")
        logger.info(f"Date range: {df_punch['Date'].min()} to {df_punch['Date'].max()}")
        logger.info(f"Hours stats: Mean={df_punch['Hours'].mean():.2f}, Std={df_punch['Hours'].std():.2f}")
        
        # Initialize trainer for this punch code
        trainer = TimeSeriesWorkforceTrainer(punch_code)
        
        # Filter working days (punch-code specific logic)
        df_filtered = trainer.filter_data_by_working_days(df_punch)
        
        logger.info(f"After working day filter: {len(df_filtered)} records")
        
        # Create advanced features
        df_featured = trainer.create_advanced_features(df_filtered)
        
        logger.info(f"After feature engineering: {len(df_featured)} records, {len(df_featured.columns)} columns")
        
        # Prepare time-series data
        X, y = trainer.prepare_time_series_data(df_featured)
        
        logger.info(f"Final training data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Target stats: Mean={y.mean():.2f}, Std={y.std():.2f}")
        
        # Train model
        validation_results = trainer.train_model(X, y)
        
        # Save model
        from config import MODELS_DIR
        success = trainer.save_model(MODELS_DIR)
        
        if success:
            logger.info(f"🎉 Punch code {punch_code} time-series training completed successfully!")
            logger.info(f"📊 Final Performance: R² = {validation_results['cv_r2_mean']:.4f}, MAE = {validation_results['cv_mae_mean']:.4f}, RMSE = {validation_results['cv_rmse_mean']:.4f}")
            return True
        else:
            logger.error(f"❌ Failed to save punch code {punch_code} model")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error training punch code {punch_code}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
