"""
Enhanced Model Training for Punch Codes 206 & 213
Enterprise-Grade Time Series Model Training with Advanced Feature Engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from collections import defaultdict
import json

# Import utilities
from utils.feature_engineering import (
    add_rolling_features_by_group, 
    add_lag_features_by_group, 
    add_cyclical_features,
    add_trend_features,
    add_pattern_features
)
from utils.holiday_utils import is_non_working_day
from utils.sql_data_connector import extract_sql_data

from config import (
    MODELS_DIR,
    DEFAULT_MODEL_PARAMS,
    FEATURE_GROUPS,
    SQL_SERVER, 
    SQL_DATABASE, 
    SQL_TRUSTED_CONNECTION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "enhanced_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_train_models")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_training_data():
    """Load training data for punch codes 206 and 213"""
    try:
        logger.info("Loading training data for enhanced models (206, 213)")
        
        query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, NoRows as Quantity, SystemKPI 
        FROM WorkUtilizationData 
        WHERE PunchCode IN (206, 213) 
        AND Hours > 0 
        AND NoOfMan > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        AND Date < '2025-05-01'
        ORDER BY Date
        """
        
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is None or df.empty:
            logger.error("No data returned from database")
            return None
            
        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Handle decimals appropriately
        df['NoOfMan'] = df['NoOfMan'].round(0).astype(int)
        df['SystemHours'] = df['SystemHours'].round(1)
        df['SystemKPI'] = df['SystemKPI'].round(2)
        df['Hours'] = df['Hours'].round(1)
        df['Quantity'] = df['Quantity'].round(0).astype(int)
        
        logger.info(f"Loaded {len(df)} records for training")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Punch codes: {df['WorkType'].unique()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_enhanced_features(df):
    """Create enhanced features for model training"""
    try:
        logger.info("Starting enhanced feature creation")
        df_enhanced = df.copy()

        # Log enabled feature groups
        enabled = [k for k, v in FEATURE_GROUPS.items() if v]
        logger.info(f"Active Feature Groups: {enabled}")

        # Temporal Features
        if FEATURE_GROUPS.get('DATE_FEATURES'):
            df_enhanced['DayOfWeek'] = df_enhanced['Date'].dt.dayofweek
            df_enhanced['Month'] = df_enhanced['Date'].dt.month
            df_enhanced['WeekNo'] = df_enhanced['Date'].dt.isocalendar().week
            df_enhanced['Year'] = df_enhanced['Date'].dt.year
            df_enhanced['Quarter'] = df_enhanced['Date'].dt.quarter

        # Schedule-based Binary Features
        df_enhanced['ScheduleType'] = np.where(df_enhanced['WorkType'] == '206', '6DAY', '5DAY')
        df_enhanced['CanWorkSunday'] = np.where(df_enhanced['WorkType'] == '206', 1, 0)
        df_enhanced['IsSunday'] = (df_enhanced['DayOfWeek'] == 6).astype(int)
        df_enhanced['IsWeekend'] = (df_enhanced['DayOfWeek'] >= 5).astype(int)
        df_enhanced['IsMonday'] = (df_enhanced['DayOfWeek'] == 0).astype(int)
        df_enhanced['IsFriday'] = (df_enhanced['DayOfWeek'] == 4).astype(int)

        # Apply Lag/Rolling/Trend/Pattern per WorkType
        for work_type in ['206', '213']:
            work_data = df_enhanced[df_enhanced['WorkType'] == work_type].copy()

            if len(work_data) > 30:
                if FEATURE_GROUPS.get('LAG_FEATURES'):
                    work_data = add_lag_features_by_group(work_data)

                if FEATURE_GROUPS.get('ROLLING_FEATURES'):
                    work_data = add_rolling_features_by_group(work_data)

                if FEATURE_GROUPS.get('TREND_FEATURES'):
                    work_data = add_trend_features(work_data)

                if FEATURE_GROUPS.get('PATTERN_FEATURES'):
                    work_data = add_pattern_features(work_data)

                df_enhanced.loc[df_enhanced['WorkType'] == work_type] = work_data

        # Holiday Feature
        df_enhanced['IsHoliday'] = df_enhanced['Date'].apply(lambda d: is_non_working_day(d.date()))

        # Cyclical Encoding
        if FEATURE_GROUPS.get('CYCLICAL_FEATURES'):
            df_enhanced = add_cyclical_features(df_enhanced)

        # Cleanup
        df_enhanced.replace([np.inf, -np.inf], np.nan, inplace=True)

        for col in df_enhanced.select_dtypes(include=[np.number]).columns:
            if df_enhanced[col].isna().any():
                median = df_enhanced[col].median() if not pd.isna(df_enhanced[col].median()) else 0.0
                df_enhanced[col].fillna(median, inplace=True)
                logger.info(f"Filled {col} NaNs with median: {median}")

        # Convert boolean columns to int
        bool_cols = ['IsHoliday', 'IsSunday', 'IsWeekend', 'IsMonday', 'IsFriday', 'CanWorkSunday']
        for col in bool_cols:
            if col in df_enhanced.columns:
                try:
                    df_enhanced[col] = df_enhanced[col].astype(int)
                except (TypeError, ValueError):
                    # If conversion fails, fill with 0 or handle appropriately
                    df_enhanced[col] = 0
                    logger.warning(f"Could not convert {col} to int, filled with 0")

        # Convert categorical columns to string
        cat_cols = ['ScheduleType']
        for col in cat_cols:
            df_enhanced[col] = df_enhanced[col].astype(str)

        logger.info(f"Final enhanced DataFrame shape: {df_enhanced.shape}")

        return df_enhanced

    except Exception as e:
        logger.error(f"Error in feature creation: {e}")
        logger.error(traceback.format_exc())
        return df

def select_features_with_time_series_validation(X, y, work_type, n_splits=5, max_features=15):
    try:
        logger.info(f"Feature selection for WorkType {work_type} using TimeSeriesSplit")
        
        # Calculate actual max features we can select
        n_available_features = len(X.columns)
        actual_max_features = min(max_features, n_available_features)
        
        if actual_max_features < max_features:
            logger.warning(f"Reducing max_features from {max_features} to {actual_max_features} (available features)")
        
        # Define numeric and categorical features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Ensure categorical features are properly formatted
        for col in categorical_features:
            X[col] = X[col].astype(str)
        
        # Create preprocessor
        preprocessors = []
        if numeric_features:
            preprocessors.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            preprocessors.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features))
        
        if not preprocessors:
            logger.error("No valid features found for preprocessing")
            return [], None, None, []  # Return empty results instead of continue
                
        preprocessor = ColumnTransformer(transformers=preprocessors)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        feature_scores = defaultdict(list)
        fold_performance = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create model pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', SelectFromModel(
                    RandomForestRegressor(**DEFAULT_MODEL_PARAMS),
                    max_features=actual_max_features
                )),
                ('model', RandomForestRegressor(**DEFAULT_MODEL_PARAMS))
            ])
            
           
            # Fit and evaluate
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            fold_performance.append({'fold': fold + 1, 'MAE': mae, 'R2': r2})
            
            # Get feature importances
            try:
                selector = pipeline.named_steps['feature_selection']
                selected_features = selector.get_support()
                feature_importance = selector.estimator_.feature_importances_
                
                feature_names = numeric_features + categorical_features
                for i, (feature, selected, importance) in enumerate(zip(feature_names, selected_features, feature_importance)):
                    if selected:
                        feature_scores[feature].append(importance)
                        
            except Exception as feat_error:
                logger.warning(f"Could not extract feature importance for fold {fold + 1}: {feat_error}")
        
        # Calculate average performance
        avg_mae = np.mean([fp['MAE'] for fp in fold_performance])
        avg_r2 = np.mean([fp['R2'] for fp in fold_performance])
        
        logger.info(f"Cross-validation results - MAE: {avg_mae:.3f}, R²: {avg_r2:.3f}")
        
        # Select features based on average importance
        avg_feature_scores = {
            feature: np.mean(scores) 
            for feature, scores in feature_scores.items()
        }
        
        sorted_features = sorted(avg_feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:max_features]]
        
        logger.info(f"Selected {len(selected_features)} features for {work_type}")
        logger.info(f"Top features: {selected_features[:5]}")
        
        return selected_features, avg_mae, avg_r2, fold_performance
        
    except Exception as e:
        logger.error(f"Error in feature selection for {work_type}: {str(e)}")
        logger.error(traceback.format_exc())
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_features[:max_features], None, None, []

def train_enhanced_model(work_type_data, work_type):
    """Train enhanced model for specific work type"""
    try:
        logger.info(f"Training enhanced model for WorkType {work_type}")
        
        if len(work_type_data) < 50:
            logger.warning(f"Insufficient data for {work_type}: {len(work_type_data)} records")
            return None, None, None
        
        # Prepare features and target
        feature_columns = [col for col in work_type_data.columns 
                         if col not in ['Date', 'Hours', 'WorkType']]
        
        X = work_type_data[feature_columns]
        y = work_type_data['Hours']
        
        logger.info(f"Initial features: {len(X.columns)}")
        logger.info(f"Data points: {len(X)}")
        
        # Feature selection
        selected_features, cv_mae, cv_r2, fold_results = select_features_with_time_series_validation(
            X, y, work_type, n_splits=5, max_features=15
        )
        
        if not selected_features:
            logger.error(f"No features selected for {work_type}")
            return None, None, None
        
        # Prepare final training data
        X_selected = X[selected_features]
        
        # Data validation
        logger.info(f"Final data validation for {work_type}")
        
        # Handle infinity and NaN values
        if np.isinf(X_selected.select_dtypes(include=[np.number])).any().any():
            logger.warning(f"Infinite values detected in {work_type}, replacing with 0")
            X_selected = X_selected.replace([np.inf, -np.inf], 0)
        
        if X_selected.isnull().any().any():
            logger.warning(f"NaN values detected in {work_type}, filling with median")
            X_selected = X_selected.fillna(X_selected.median())
        
        logger.info(f"Data validation complete for {work_type}")
        
        # Create preprocessing pipeline
        numeric_features = X_selected.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_features:
            X_selected[col] = X_selected[col].astype(str)
        
        preprocessors = []
        if numeric_features:
            preprocessors.append(('num', StandardScaler(), numeric_features))
        if categorical_features:
            preprocessors.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features))
        
        if not preprocessors:
            logger.error(f"No valid features found for {work_type}")
            return None, None, None
            
        preprocessor = ColumnTransformer(transformers=preprocessors)
        
        # Create final model pipeline
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(**DEFAULT_MODEL_PARAMS))
        ])
        
        # Train final model
        final_pipeline.fit(X_selected, y)
        
        # Model evaluation
        y_pred_final = final_pipeline.predict(X_selected)
        final_mae = mean_absolute_error(y, y_pred_final)
        final_r2 = r2_score(y, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))
        mape = np.mean(np.abs((y - y_pred_final) / y)) * 100
        
        logger.info(f"Final model trained for {work_type}")
        logger.info(f"Performance:")
        logger.info(f"MAE: {final_mae:.3f}")
        logger.info(f"R²: {final_r2:.3f}")
        logger.info(f"RMSE: {final_rmse:.3f}")
        logger.info(f"MAPE: {mape:.2f}%")
        
        # Store model metadata
        model_metadata = {
            'work_type': work_type,
            'features': selected_features,
            'training_records': len(work_type_data),
            'cv_mae': cv_mae,
            'cv_r2': cv_r2,
            'final_mae': final_mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'mape': mape,
            'fold_results': fold_results,
            'training_date': datetime.now().isoformat()
        }
        
        return final_pipeline, model_metadata, selected_features
        
    except Exception as e:
        logger.error(f"Error training model for {work_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def save_enhanced_models(models, metadata, features):
    """Save enhanced models and metadata"""
    try:
        logger.info("Saving enhanced models and metadata")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for work_type, model in models.items():
            if model is not None:
                model_filename = f"enhanced_model_{work_type}_{timestamp}.pkl"
                model_path = os.path.join(MODELS_DIR, model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Also save with standard name for loading
                standard_filename = f"enhanced_model_{work_type}.pkl"
                standard_path = os.path.join(MODELS_DIR, standard_filename)
                
                with open(standard_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"Saved model for {work_type}: {model_filename}")
        
        # Save metadata
        metadata_filename = f"enhanced_models_metadata_{timestamp}.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save features mapping
        features_filename = f"enhanced_features_{timestamp}.json"
        features_path = os.path.join(MODELS_DIR, features_filename)
        
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        # Save performance summary
        performance_summary = []
        for work_type, meta in metadata.items():
            if meta:
                performance_summary.append({
                    'WorkType': work_type,
                    'Final_MAE': meta['final_mae'],
                    'Final_R2': meta['final_r2'],
                    'MAPE': meta['mape'],
                    'CV_MAE': meta['cv_mae'],
                    'Training_Records': meta['training_records'],
                    'Features_Count': len(meta['features'])
                })
        
        performance_df = pd.DataFrame(performance_summary)
        performance_filename = f"enhanced_models_performance_{timestamp}.xlsx"
        performance_path = os.path.join(MODELS_DIR, performance_filename)
        
        performance_df.to_excel(performance_path, index=False)
        
        logger.info(f"All enhanced models and metadata saved")
        logger.info(f"Performance summary: {performance_filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving enhanced models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run enhanced model training"""
    try:
        logger.info("Starting Enhanced Model Training for Punch Codes 206 & 213")
        
        # Load training data
        df = load_training_data()
        if df is None:
            logger.error("Failed to load training data. Exiting.")
            return
        
        # Create enhanced features
        df_enhanced = create_enhanced_features(df)
        
        # Check data distribution
        logger.info("Data distribution:")
        for work_type in df_enhanced['WorkType'].unique():
            wt_data = df_enhanced[df_enhanced['WorkType'] == work_type]
            logger.info(f"WorkType {work_type}: {len(wt_data)} records")
            logger.info(f"Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
            logger.info(f"Hours avg: {wt_data['Hours'].mean():.2f}")
        
        # Train models for each work type
        models = {}
        metadata = {}
        features = {}
        
        for work_type in ['206', '213']:
            logger.info(f"Processing WorkType {work_type}")
            
            work_data = df_enhanced[df_enhanced['WorkType'] == work_type].copy()
            work_data = work_data.sort_values('Date')  # Ensure temporal order
            
            if len(work_data) < 50:
                logger.warning(f"Skipping {work_type}: Insufficient data ({len(work_data)} records)")
                continue
            
            # Train enhanced model
            model, model_metadata, selected_features = train_enhanced_model(work_data, work_type)
            
            if model is not None:
                models[work_type] = model
                metadata[work_type] = model_metadata
                features[work_type] = selected_features
                
                logger.info(f"Successfully trained enhanced model for {work_type}")
            else:
                logger.error(f"Failed to train model for {work_type}")
        
        # Save models and metadata
        if models:
            success = save_enhanced_models(models, metadata, features)
            
            if success:
                logger.info("ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
                logger.info(f"Trained models: {list(models.keys())}")
                
                # Print performance summary
                for work_type, meta in metadata.items():
                    logger.info(f"{work_type} Performance Summary:")
                    logger.info(f"MAE: {meta['final_mae']:.3f}")
                    logger.info(f"R²: {meta['final_r2']:.3f}")
                    logger.info(f"MAPE: {meta['mape']:.2f}%")
                    logger.info(f"Features: {len(meta['features'])}")
            else:
                logger.error("Failed to save enhanced models")
        else:
            logger.error("No models were successfully trained")
            
    except Exception as e:
        logger.error(f"Error in main training process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()