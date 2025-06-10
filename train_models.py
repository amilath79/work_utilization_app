"""
Enhanced Module for training workforce prediction models with intelligent feature selection.
Includes K-Fold Time Series Cross-Validation and Enterprise MLflow Integration.
All original functionality preserved with advanced enhancements.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.impute import SimpleImputer
from collections import defaultdict
import json

# Import feature engineering functions from utils
from utils.feature_engineering import engineer_features, create_lag_features

from config import (
    MODELS_DIR, DATA_DIR, LAG_DAYS, ROLLING_WINDOWS, 
    CHUNK_SIZE, DEFAULT_MODEL_PARAMS,
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION,
    SQL_USERNAME, SQL_PASSWORD,
    FEATURE_GROUPS, PRODUCTIVITY_FEATURES, DATE_FEATURES, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS,
    enterprise_logger
)

# Enterprise MLflow integration
from utils.enterprise_mlflow import mlflow_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_train_models")

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(file_path):
    """
    Load and preprocess the work utilization data
    
    Parameters:
    -----------
    file_path : str
        Path to the data file (Excel or CSV)
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use Excel (.xlsx/.xls) or CSV (.csv)")
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Ensure Date column is datetime 
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle PunchCode as WorkType if it exists
        if 'PunchCode' in df.columns and 'WorkType' not in df.columns:
            df = df.rename(columns={'PunchCode': 'WorkType'})
            logger.info("Renamed 'PunchCode' column to 'WorkType'")
        
        # Ensure WorkType is treated as string
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Process all numeric columns
        numeric_columns = ['Hours', 'NoOfMan', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].replace('-', 0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Sort by Date
        df = df.sort_values('Date')
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to load data: {str(e)}")

def diagnose_training_data(df, work_type):
    """
    Enhanced diagnosis of potential issues in training data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    work_type : str
        Work type to diagnose
        
    Returns:
    --------
    float
        Average NoOfMan value for the work type
    """
    wt_data = df[df['WorkType'] == work_type]
    
    print(f"\n=== ENHANCED DIAGNOSIS for {work_type} ===")
    print(f"Total records: {len(wt_data)}")
    print(f"Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
    print(f"NoOfMan statistics:")
    print(f"  Mean: {wt_data['NoOfMan'].mean():.2f}")
    print(f"  Median: {wt_data['NoOfMan'].median():.2f}")
    print(f"  Min: {wt_data['NoOfMan'].min():.2f}")
    print(f"  Max: {wt_data['NoOfMan'].max():.2f}")
    print(f"  Std: {wt_data['NoOfMan'].std():.2f}")
    
    # Check for outliers using IQR method
    q1 = wt_data['NoOfMan'].quantile(0.25)
    q3 = wt_data['NoOfMan'].quantile(0.75)
    iqr = q3 - q1
    outliers = wt_data[(wt_data['NoOfMan'] < q1 - 1.5*iqr) | (wt_data['NoOfMan'] > q3 + 1.5*iqr)]
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(wt_data)*100:.1f}%)")
    
    # Check recent trend (last 30 records)
    recent = wt_data.tail(30)
    print(f"Recent 30 records average: {recent['NoOfMan'].mean():.2f}")
    
    # Enhanced data quality checks
    print(f"Data Quality Assessment:")
    zero_count = (wt_data['NoOfMan'] == 0).sum()
    low_count = (wt_data['NoOfMan'] < 1).sum()
    print(f"  Zero values: {zero_count} ({zero_count/len(wt_data)*100:.1f}%)")
    print(f"  Values < 1: {low_count} ({low_count/len(wt_data)*100:.1f}%)")
    print(f"  Missing dates: {wt_data['Date'].isna().sum()}")
    
    # Check for temporal gaps
    date_diff = wt_data['Date'].diff().dt.days
    large_gaps = (date_diff > 7).sum()
    print(f"  Large time gaps (>7 days): {large_gaps}")
    
    return wt_data['NoOfMan'].mean()

def validate_model_performance(model, X, y, work_type):
    """
    Enhanced validation to check if model is learning properly
    
    Parameters:
    -----------
    model : sklearn.Pipeline
        Trained model pipeline
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    work_type : str
        Work type being validated
        
    Returns:
    --------
    float
        Prediction to target ratio
    """
    
    # Check training performance
    train_pred = model.predict(X)
    train_mae = mean_absolute_error(y, train_pred)
    train_r2 = r2_score(y, train_pred)
    
    print(f"\n{work_type} ENHANCED Training Performance Validation:")
    print(f"  Training MAE: {train_mae:.3f}")
    print(f"  Training R²: {train_r2:.3f}")
    print(f"  Target mean: {y.mean():.3f}")
    print(f"  Target std: {y.std():.3f}")
    print(f"  Prediction mean: {train_pred.mean():.3f}")
    print(f"  Prediction std: {train_pred.std():.3f}")
    print(f"  Prediction/Target ratio: {train_pred.mean()/y.mean():.3f}")
    
    # Check for systematic bias
    bias = train_pred.mean() - y.mean()
    bias_pct = (bias / y.mean()) * 100 if y.mean() > 0 else 0
    print(f"  Systematic bias: {bias:.3f} ({bias_pct:.1f}%)")
    
    # Check prediction variance capture
    pred_std = train_pred.std()
    target_std = y.std()
    variance_ratio = pred_std/target_std if target_std > 0 else 0
    print(f"  Variance capture ratio: {variance_ratio:.3f}")
    
    # Check for potential overfitting indicators
    residuals = y - train_pred
    residual_std = residuals.std()
    print(f"  Residual std: {residual_std:.3f}")
    
    # Warning flags
    if train_r2 > 0.99:
        print(f"  ⚠️  WARNING: Very high R² ({train_r2:.3f}) - possible overfitting")
    if abs(bias_pct) > 10:
        print(f"  ⚠️  WARNING: High bias ({bias_pct:.1f}%) - model may be systematically wrong")
    if variance_ratio < 0.5:
        print(f"  ⚠️  WARNING: Low variance capture ({variance_ratio:.3f}) - model may be too simple")
    
    return train_pred.mean() / y.mean() if y.mean() > 0 else 1.0

def select_features_with_time_series_validation(X, y, model_params, n_splits=5, max_features=12):
    """
    Enhanced feature selection using K-fold time series cross-validation to prevent data leakage
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    model_params : dict
        RandomForest parameters
    n_splits : int
        Number of time series splits
    max_features : int
        Maximum number of features to select
        
    Returns:
    --------
    list
        Selected feature names based on cross-validation performance
    """
    try:
        logger.info(f"🔍 Starting K-fold time series feature selection with {len(X.columns)} features using {n_splits} folds")
        
        # Track feature importance and selection frequency across folds
        feature_scores = defaultdict(list)
        feature_selection_counts = defaultdict(int)
        fold_performance = []
        
        # Initialize TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Perform feature selection within each fold to prevent data leakage
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing K-fold {fold + 1}/{n_splits} for feature selection")
            
            # Split data for this fold
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Create a RandomForest for feature selection (simpler than full pipeline)
            rf_selector = RandomForestRegressor(
                n_estimators=100,  # Smaller for speed during selection
                max_depth=10,
                random_state=42 + fold,  # Different seed per fold
                n_jobs=-1
            )
            
            # Fit model on fold training data
            rf_selector.fit(X_train_fold, y_train_fold)
            
            # Get feature importances for this fold
            importances = rf_selector.feature_importances_
            
            # Store importance scores for averaging later
            for i, feature in enumerate(X.columns):
                feature_scores[feature].append(importances[i])
            
            # Test performance of this fold
            fold_pred = rf_selector.predict(X_test_fold)
            fold_mae = mean_absolute_error(y_test_fold, fold_pred)
            fold_performance.append(fold_mae)
            
            # Use SelectFromModel to identify important features in this fold
            try:
                selector = SelectFromModel(
                    rf_selector, 
                    threshold='median',  # Use median as threshold
                    max_features=max_features
                )
                
                selector.fit(X_train_fold, y_train_fold)
                selected_features = X.columns[selector.get_support()].tolist()
                
                # Count how many times each feature was selected across folds
                for feature in selected_features:
                    feature_selection_counts[feature] += 1
                    
                logger.info(f"Fold {fold + 1}: Selected {len(selected_features)} features, MAE: {fold_mae:.4f}")
                
            except Exception as e:
                logger.warning(f"SelectFromModel failed in fold {fold + 1}: {e}, using RFE fallback")
                # Fallback: use Recursive Feature Elimination
                rfe = RFE(rf_selector, n_features_to_select=min(max_features, len(X.columns)))
                rfe.fit(X_train_fold, y_train_fold)
                selected_features = X.columns[rfe.support_].tolist()
                
                for feature in selected_features:
                    feature_selection_counts[feature] += 1
        
        # Calculate average importance across all folds
        avg_importance = {}
        for feature, scores in feature_scores.items():
            avg_importance[feature] = np.mean(scores)
        
        # Calculate average performance across folds
        avg_fold_performance = np.mean(fold_performance)
        logger.info(f"Average K-fold MAE during feature selection: {avg_fold_performance:.4f}")
        
        # Combine selection frequency and average importance for final ranking
        feature_ranking = []
        for feature in X.columns:
            selection_frequency = feature_selection_counts[feature] / n_splits
            avg_imp = avg_importance[feature]
            # Weight frequency more heavily than raw importance
            combined_score = (selection_frequency * 0.6) + (avg_imp * 0.4)
            
            feature_ranking.append({
                'feature': feature,
                'avg_importance': avg_imp,
                'selection_frequency': selection_frequency,
                'combined_score': combined_score
            })
        
        # Sort by combined score (descending)
        feature_ranking.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Select features that appeared in at least 40% of folds (robust selection)
        selected_features = []
        min_frequency = 0.4  # Must appear in at least 40% of folds
        
        for item in feature_ranking:
            if len(selected_features) >= max_features:
                break
            if item['selection_frequency'] >= min_frequency:
                selected_features.append(item['feature'])
        
        # Ensure we have minimum viable features
        if len(selected_features) < 5:
            logger.warning(f"Only {len(selected_features)} features met frequency threshold, taking top features by score")
            # Take top features by combined score regardless of frequency
            selected_features = [item['feature'] for item in feature_ranking[:max(5, max_features//2)]]
        
        # Log detailed feature selection results
        logger.info(f"✅ K-fold feature selection completed: {len(selected_features)} features selected")
        logger.info("Selected features with validation scores:")
        for item in feature_ranking[:len(selected_features)]:
            logger.info(f"  {item['feature']}: importance={item['avg_importance']:.4f}, "
                       f"frequency={item['selection_frequency']:.2f}, score={item['combined_score']:.4f}")
        
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in K-fold time series feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback: return top features by simple importance
        logger.warning("Falling back to simple feature importance ranking")
        rf_fallback = RandomForestRegressor(**model_params)
        rf_fallback.fit(X, y)
        importances = rf_fallback.feature_importances_
        top_indices = np.argsort(importances)[-max_features:]
        return X.columns[top_indices].tolist()

def validate_features_performance(X, y, selected_features, model_params, n_splits=5):
    """
    Validate that selected features actually improve performance using K-fold validation
    
    Parameters:
    -----------
    X : pd.DataFrame
        Full feature matrix
    y : pd.Series
        Target variable
    selected_features : list
        Selected feature names
    model_params : dict
        Model parameters
    n_splits : int
        Number of validation splits
        
    Returns:
    --------
    tuple
        (validated_features, performance_metrics)
    """
    try:
        logger.info(f"🎯 Validating performance of {len(selected_features)} selected features using K-fold")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Test selected features performance
        X_selected = X[selected_features]
        selected_scores = []
        
        # Create preprocessing pipeline for consistent validation
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(**model_params))
        ])
        
        # K-fold validation with selected features
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected)):
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            selected_scores.append(mae)
        
        avg_selected_mae = np.mean(selected_scores)
        std_selected_mae = np.std(selected_scores)
        
        # Compare with using all features (if significant difference in feature count)
        improvement_pct = 0
        if len(X.columns) > len(selected_features) * 1.5:  # Only compare if substantial reduction
            logger.info(f"Comparing selected features vs all features performance")
            all_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                all_scores.append(mae)
            
            avg_all_mae = np.mean(all_scores)
            improvement_pct = (avg_all_mae - avg_selected_mae) / avg_all_mae * 100
            
            logger.info(f"K-fold Feature Selection Validation Results:")
            logger.info(f"  All features MAE: {avg_all_mae:.4f} ± {np.std(all_scores):.4f}")
            logger.info(f"  Selected features MAE: {avg_selected_mae:.4f} ± {std_selected_mae:.4f}")
            logger.info(f"  Performance improvement: {improvement_pct:.2f}%")
        else:
            logger.info(f"Selected features K-fold validation MAE: {avg_selected_mae:.4f} ± {std_selected_mae:.4f}")
        
        # Remove highly correlated features if we have many (reduce redundancy)
        validated_features = selected_features
        if len(selected_features) > 8:  # Only if we have many features
            logger.info(f"Removing redundant features from {len(selected_features)} selected features")
            validated_features = remove_redundant_features(X[selected_features], y, selected_features, model_params)
        
        performance_metrics = {
            'selected_mae': avg_selected_mae,
            'selected_mae_std': std_selected_mae,
            'n_features': len(validated_features),
            'improvement_pct': improvement_pct
        }
        
        logger.info(f"✅ Feature validation completed: {len(validated_features)} final features")
        return validated_features, performance_metrics
        
    except Exception as e:
        logger.error(f"Error validating features performance: {str(e)}")
        logger.error(traceback.format_exc())
        return selected_features, {'selected_mae': 0, 'selected_mae_std': 0, 'n_features': len(selected_features), 'improvement_pct': 0}

def remove_redundant_features(X, y, features, model_params, max_correlation=0.9):
    """
    Remove highly correlated features to reduce redundancy and improve generalization
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix with selected features
    y : pd.Series
        Target variable
    features : list
        Feature names
    model_params : dict
        Model parameters
    max_correlation : float
        Maximum allowed correlation between features
        
    Returns:
    --------
    list
        Features with redundant ones removed
    """
    try:
        if len(features) <= 5:  # Don't remove if we have few features
            logger.info(f"Keeping all {len(features)} features (minimum threshold)")
            return features
            
        logger.info(f"Analyzing feature correlations for redundancy removal (threshold: {max_correlation})")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if correlation > max_correlation:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], correlation))
        
        if not high_corr_pairs:
            logger.info("No highly correlated features found")
            return features
        
        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # Get individual feature importance to decide which correlated feature to keep
        rf_temp = RandomForestRegressor(**model_params)
        rf_temp.fit(X, y)
        importances = dict(zip(X.columns, rf_temp.feature_importances_))
        
        # For each correlated pair, remove the one with lower importance
        features_to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            if importances[feat1] > importances[feat2]:
                features_to_remove.add(feat2)
                logger.info(f"Removing {feat2} (corr={corr:.3f} with {feat1}, importance: {importances[feat2]:.4f} < {importances[feat1]:.4f})")
            else:
                features_to_remove.add(feat1)
                logger.info(f"Removing {feat1} (corr={corr:.3f} with {feat2}, importance: {importances[feat1]:.4f} < {importances[feat2]:.4f})")
        
        # Create final feature list
        final_features = [f for f in features if f not in features_to_remove]
        
        # Ensure we maintain minimum viable features
        if len(final_features) < 5:
            logger.warning(f"Redundancy removal left only {len(final_features)} features, restoring to minimum 5")
            # Keep top 5 by importance
            sorted_features = sorted(features, key=lambda x: importances[x], reverse=True)
            final_features = sorted_features[:5]
        
        logger.info(f"After redundancy removal: {len(final_features)} features (removed {len(features_to_remove)})")
        return final_features
        
    except Exception as e:
        logger.error(f"Error removing redundant features: {str(e)}")
        logger.error(traceback.format_exc())
        return features

def build_models(processed_data, work_types=None, n_splits=5):
    """
    Build and train models with enhanced K-fold time series feature selection and validation
    
    Parameters:
    -----------
    processed_data : pd.DataFrame
        DataFrame with all engineered features
    work_types : list, optional
        List of work types to build models for
    n_splits : int
        Number of splits for K-fold time series cross-validation
        
    Returns:
    --------
    tuple
        (models, feature_importances, metrics)
    """
    try:
        print('XXXXXX')
        processed_data.to_excel('work_type_data.xlsx')
        # Initialize enterprise MLflow tracking
        mlflow_initialized = mlflow_manager.initialize()
        
        if not mlflow_initialized:
            enterprise_logger.warning("MLflow tracking not available, proceeding without tracking")
        
        # If work_types is not provided, get them from the data
        if work_types is None:
            work_types = sorted(processed_data['WorkType'].unique())
        
        enterprise_logger.info(f"🚀 Building ENHANCED models for {len(work_types)} work types with K-fold time series feature selection")
        
        # Log which feature groups are enabled
        active_groups = [group for group, enabled in FEATURE_GROUPS.items() if enabled]
        enterprise_logger.info(f"Active feature groups: {active_groups}")
        
        # Start enterprise training session
        session_params = {
            "n_splits": n_splits,
            "total_work_types": len(work_types),
            "feature_groups": active_groups,
            "model_params": DEFAULT_MODEL_PARAMS,
            "environment": "production",
            "data_shape": processed_data.shape,
            "training_timestamp": datetime.now().isoformat(),
            "feature_selection_enabled": True,
            "validation_method": "k_fold_time_series"
        }
        
        with mlflow_manager.start_run(
            run_name=f"enhanced_k_fold_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"session_type": "k_fold_enhanced_training", "enterprise": "true", "feature_selection": "k_fold_time_series"}
        ) as parent_run:
            
            if parent_run:
                mlflow_manager.log_training_parameters(session_params)
                enterprise_logger.info(f"Started enhanced K-fold training session: {parent_run.info.run_id}")
            
            models = {}
            feature_importances = {}
            metrics = {}
            
            # Build comprehensive feature lists using config-driven approach
            numeric_features = []
            categorical_features = []
            
            # Essential lag features (most important for time series)
            if FEATURE_GROUPS['LAG_FEATURES']:
                for lag in ESSENTIAL_LAGS:
                    numeric_features.append(f'NoOfMan_lag_{lag}')
                # Add quantity lag for productivity analysis
                if FEATURE_GROUPS['PRODUCTIVITY_FEATURES']:
                    numeric_features.append('Quantity_lag_1')
            
            # Essential rolling features (smoothed patterns)
            if FEATURE_GROUPS['ROLLING_FEATURES']:
                for window in ESSENTIAL_WINDOWS:
                    numeric_features.append(f'NoOfMan_rolling_mean_{window}')
            
            # Date features from config (temporal patterns)
            if FEATURE_GROUPS['DATE_FEATURES']:
                categorical_features.extend(DATE_FEATURES['categorical'])
                numeric_features.extend(DATE_FEATURES['numeric'])
                numeric_features.append('DayOfMonth')  # Add day of month for monthly patterns
            
            # Productivity features from config (efficiency metrics)
            if FEATURE_GROUPS['PRODUCTIVITY_FEATURES']:
                numeric_features.extend(PRODUCTIVITY_FEATURES)
            
            # Pattern features (seasonal patterns - optional)
            if FEATURE_GROUPS['PATTERN_FEATURES']:
                numeric_features.append('NoOfMan_same_dow_lag')
            
            # Trend features (momentum - optional)  
            if FEATURE_GROUPS['TREND_FEATURES']:
                numeric_features.append('NoOfMan_trend_7d')
            
            # Modified MAPE calculation with threshold
            def modified_mape(y_true, y_pred, epsilon=1.0):
                """Calculate MAPE with a minimum threshold to avoid division by zero"""
                denominator = np.maximum(np.abs(y_true), epsilon)
                return np.mean(np.abs(y_pred - y_true) / denominator) * 100
            
            # Log initial feature configuration
            enterprise_logger.info(f"Initial feature configuration: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
            enterprise_logger.info(f"Numeric features: {numeric_features}")
            enterprise_logger.info(f"Categorical features: {categorical_features}")
            
            # Train model for each work type
            for work_type in work_types:
                # Enhanced model training with full pipeline and tracking
                with mlflow_manager.start_run(
                    run_name=f"k_fold_enhanced_model_{work_type}",
                    nested=True,
                    tags={"work_type": work_type, "model_type": "random_forest", "feature_selection": "k_fold_time_series"}
                ) as model_run:
                    
                    try:
                        enterprise_logger.info(f"🎯 Building K-FOLD ENHANCED model for WorkType: {work_type}")
                        
                        # ✅ PHASE 0: DIAGNOSIS AND DATA VALIDATION
                        avg_value = diagnose_training_data(processed_data, work_type)
                        print(f'Work Type : {work_type} - Average Value : {avg_value}')
                        
                        # Log work type specific parameters for MLflow
                        if model_run:
                            work_type_params = {
                                "work_type": work_type,
                                "avg_target_value": avg_value,
                                "data_points": len(processed_data[processed_data['WorkType'] == work_type])
                            }
                            mlflow_manager.log_training_parameters(work_type_params)
                        
                        # Filter data for this WorkType
                        work_type_data = processed_data[processed_data['WorkType'] == work_type] 
                        
                        if len(work_type_data) < 50:  # Increased minimum data requirement for robust validation
                            logger.warning(f"Skipping {work_type}: Not enough data ({len(work_type_data)} records, minimum 50 required)")
                            continue
                        
                        # Sort data by date to ensure proper time series splitting
                        work_type_data = work_type_data.sort_values('Date')
                        
                        # Check which features are available in the dataset
                        available_numeric = [f for f in numeric_features if f in work_type_data.columns]
                        available_categorical = [f for f in categorical_features if f in work_type_data.columns]
                        
                        enterprise_logger.info(f"Available features for {work_type}: {len(available_numeric)} numeric, {len(available_categorical)} categorical")
                        
                        # Log missing features for debugging
                        missing_numeric = [f for f in numeric_features if f not in work_type_data.columns]
                        missing_categorical = [f for f in categorical_features if f not in work_type_data.columns]
                        
                        if missing_numeric:
                            logger.debug(f"Missing numeric features for {work_type}: {missing_numeric}")
                        if missing_categorical:
                            logger.debug(f"Missing categorical features for {work_type}: {missing_categorical}")
                        
                        # Skip if no features are available
                        if len(available_numeric) == 0 and len(available_categorical) == 0:
                            logger.warning(f"Skipping {work_type}: No features available")
                            continue
                        
                        # Prepare initial feature set and target
                        all_available_features = available_numeric + available_categorical
                        X_initial = work_type_data[all_available_features]
                        y = work_type_data['NoOfMan']
                        
                        enterprise_logger.info(f"=== K-FOLD ENHANCED FEATURE PROCESSING - {work_type} ===")
                        enterprise_logger.info(f"Initial features: {len(all_available_features)}")
                        enterprise_logger.info(f"Numeric: {len(available_numeric)}, Categorical: {len(available_categorical)}")
                        enterprise_logger.info(f"Target range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}")
                        
                        # ✅ PHASE 1: K-FOLD TIME SERIES FEATURE SELECTION
                        logger.info(f"🔍 Phase 1: K-Fold Time Series Feature Selection for {work_type}")
                        selected_features = select_features_with_time_series_validation(
                            X_initial, y, DEFAULT_MODEL_PARAMS, n_splits=n_splits, max_features=12
                        )
                        
                        # ✅ PHASE 2: FEATURE PERFORMANCE VALIDATION
                        logger.info(f"✅ Phase 2: K-Fold Feature Performance Validation for {work_type}")
                        validated_features, validation_metrics = validate_features_performance(
                            X_initial, y, selected_features, DEFAULT_MODEL_PARAMS, n_splits=n_splits
                        )
                        
                        # Log comprehensive feature selection results
                        enterprise_logger.info(f"K-Fold Feature Selection Results for {work_type}:")
                        enterprise_logger.info(f"  Initial features: {len(all_available_features)}")
                        enterprise_logger.info(f"  K-fold selected: {len(selected_features)}")
                        enterprise_logger.info(f"  Final validated: {len(validated_features)}")
                        enterprise_logger.info(f"  Feature reduction: {(1 - len(validated_features) / len(all_available_features)) * 100:.1f}%")
                        enterprise_logger.info(f"  Validated features: {validated_features}")
                        
                        # Use validated features for final model training
                        X = X_initial[validated_features]
                        
                        # Define preprocessing pipeline for final features
                        validated_numeric = [f for f in validated_features if f in available_numeric]
                        validated_categorical = [f for f in validated_features if f in available_categorical]
                        
                        transformers = []
                        
                        if validated_numeric:
                            numeric_pipeline = Pipeline([
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())  # Critical for mixed feature types
                            ])
                            transformers.append(('num', numeric_pipeline, validated_numeric))
                            
                        if validated_categorical:
                            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), validated_categorical))
                        
                        preprocessor = ColumnTransformer(transformers=transformers)
                        
                        # Create final model pipeline with validated features
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('model', RandomForestRegressor(**DEFAULT_MODEL_PARAMS))
                        ])
                        
                        # ✅ PHASE 3: ENHANCED K-FOLD TIME SERIES CROSS-VALIDATION
                        logger.info(f"📊 Phase 3: Final K-Fold Time Series Cross-Validation for {work_type}")
                        
                        tscv = TimeSeriesSplit(n_splits=n_splits)
                        
                        # Initialize metrics lists for cross-validation
                        mae_scores = []
                        rmse_scores = []
                        r2_scores = []
                        mape_scores = []
                        
                        # Enhanced cross-validation with detailed fold-by-fold analysis
                        fold_results = []
                        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                            
                            # Train model on this fold
                            pipeline.fit(X_train, y_train)
                            
                            # Validate model learning on first fold
                            if fold == 0:
                                bias_ratio = validate_model_performance(pipeline, X_train, y_train, work_type)
                                print(f'Bias Ratio for {work_type} : {bias_ratio}')
                            
                            # Make predictions on test set
                            y_pred = pipeline.predict(X_test)
                            
                            # Calculate comprehensive metrics for this fold
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            mape = modified_mape(y_test, y_pred, epsilon=1.0)
                            
                            # Store scores
                            mae_scores.append(mae)
                            rmse_scores.append(rmse)
                            r2_scores.append(r2)
                            mape_scores.append(mape)
                            
                            # Store detailed fold results
                            fold_results.append({
                                'fold': fold + 1,
                                'mae': mae,
                                'rmse': rmse,
                                'r2': r2,
                                'mape': mape,
                                'train_size': len(X_train),
                                'test_size': len(X_test),
                                'train_period': f"{work_type_data.iloc[train_idx]['Date'].min()} to {work_type_data.iloc[train_idx]['Date'].max()}",
                                'test_period': f"{work_type_data.iloc[test_idx]['Date'].min()} to {work_type_data.iloc[test_idx]['Date'].max()}"
                            })
                            
                            logger.info(f"  K-Fold {fold + 1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}% | Train: {len(X_train)}, Test: {len(X_test)}")
                        
                        # ✅ PHASE 4: FINAL MODEL TRAINING ON ALL DATA
                        logger.info(f"🏆 Phase 4: Final Model Training on All Data for {work_type}")
                        pipeline.fit(X, y)
                        
                        # Calculate comprehensive average metrics with stability indicators
                        avg_metrics = {
                            'MAE': np.mean(mae_scores),
                            'RMSE': np.mean(rmse_scores),
                            'R²': np.mean(r2_scores),
                            'MAPE': np.mean(mape_scores),
                            'MAE_std': np.std(mae_scores),
                            'RMSE_std': np.std(rmse_scores),
                            'R²_std': np.std(r2_scores),
                            'MAPE_std': np.std(mape_scores),
                            'CV_MAE': np.std(mae_scores) / np.mean(mae_scores),  # Coefficient of variation
                            'n_features_initial': len(all_available_features),
                            'n_features_selected': len(selected_features),
                            'n_features_final': len(validated_features),
                            'feature_reduction_pct': (1 - len(validated_features) / len(all_available_features)) * 100
                        }
                        
                        # Store model with metadata
                        models[work_type] = pipeline
                        models[work_type].selected_features_ = validated_features  # Store for prediction use
                        models[work_type].fold_results_ = fold_results  # Store detailed fold results
                        models[work_type].feature_selection_method_ = 'k_fold_time_series'
                        
                        # Get comprehensive feature importances from final model
                        final_model = pipeline.named_steps['model']
                        if hasattr(final_model, 'feature_importances_'):
                            # Start with validated features
                            feature_names = validated_features.copy()
                            
                            # Handle categorical feature expansion
                            if validated_categorical:
                                try:
                                    preprocessor_fitted = pipeline.named_steps['preprocessor']
                                    transformer_names = [name for name, _, _ in preprocessor_fitted.transformers_]
                                    
                                    if 'cat' in transformer_names:
                                        # Get expanded categorical feature names
                                        ohe = preprocessor_fitted.named_transformers_['cat']
                                        expanded_names = validated_numeric.copy()  # Start with numeric features
                                        
                                        for i, feature in enumerate(validated_categorical):
                                            categories = ohe.categories_[i]
                                            for category in categories:
                                                expanded_names.append(f"{feature}_{category}")
                                        
                                        feature_names = expanded_names
                                        
                                except Exception as cat_error:
                                    logger.warning(f"Could not expand categorical features for {work_type}: {cat_error}")
                                    # Keep original feature names as fallback
                            
                            # Create feature importance dictionary
                            importances = final_model.feature_importances_
                            if len(feature_names) == len(importances):
                                feature_importances[work_type] = dict(zip(feature_names, importances))
                            else:
                                logger.warning(f"Feature name count mismatch for {work_type}: {len(feature_names)} names vs {len(importances)} importances")
                                # Fallback: use validated features with available importances
                                min_length = min(len(validated_features), len(importances))
                                feature_importances[work_type] = dict(zip(validated_features[:min_length], importances[:min_length]))
                        
                        # Store comprehensive metrics
                        metrics[work_type] = avg_metrics
                        
                        # Enhanced MLflow logging with K-fold results
                        if model_run:
                            try:
                                # Log enhanced metrics including K-fold statistics
                                enhanced_metrics = avg_metrics.copy()
                                enhanced_metrics.update({
                                    'validation_mae': validation_metrics.get('selected_mae', 0),
                                    'validation_improvement_pct': validation_metrics.get('improvement_pct', 0)
                                })
                                
                                # Log individual fold scores for detailed analysis
                                cv_scores = {
                                    'MAE': mae_scores,
                                    'RMSE': rmse_scores,
                                    'R2': r2_scores,
                                    'MAPE': mape_scores
                                }
                                
                                mlflow_manager.log_model_metrics(work_type, enhanced_metrics, cv_scores)
                                mlflow_manager.log_model_artifact(pipeline, work_type, feature_importances.get(work_type))
                                
                                # Log detailed feature selection and validation parameters
                                mlflow_manager.log_training_parameters({
                                    'selected_features': validated_features,
                                    'n_features_final': len(validated_features),
                                    'feature_selection_method': 'k_fold_time_series',
                                    'n_splits': n_splits,
                                    'fold_results': fold_results
                                })
                                
                            except Exception as mlflow_error:
                                logger.warning(f"MLflow logging failed for {work_type}: {mlflow_error}")
                                # Continue without failing the training
                        
                        # Log comprehensive training completion
                        enterprise_logger.info(f"✅ K-Fold Enhanced model completed for {work_type}:")
                        enterprise_logger.info(f"   MAE: {avg_metrics['MAE']:.4f} ± {avg_metrics['MAE_std']:.4f} (CV: {avg_metrics['CV_MAE']:.3f})")
                        enterprise_logger.info(f"   R²: {avg_metrics['R²']:.4f} ± {avg_metrics['R²_std']:.4f}")
                        enterprise_logger.info(f"   MAPE: {avg_metrics['MAPE']:.2f}% ± {avg_metrics['MAPE_std']:.2f}%")
                        enterprise_logger.info(f"   Features: {len(validated_features)} final (reduced {avg_metrics['feature_reduction_pct']:.1f}% from {len(all_available_features)})")
                        
                        # Log top 5 most important features for interpretability
                        if work_type in feature_importances and feature_importances[work_type]:
                            sorted_features = sorted(feature_importances[work_type].items(), key=lambda x: x[1], reverse=True)
                            logger.info(f"Top 5 most important features for {work_type}:")
                            for feature, importance in sorted_features[:5]:
                                logger.info(f"  {feature}: {importance:.4f}")
                        
                        # Log detailed K-fold results for analysis
                        logger.info(f"Detailed K-fold results for {work_type}:")
                        for result in fold_results:
                            logger.info(f"  Fold {result['fold']}: MAE={result['mae']:.4f}, R²={result['r2']:.4f} | {result['test_period']}")
                        
                    except Exception as e:
                        logger.error(f"Error training K-fold enhanced model for WorkType {work_type}: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Continue with next work type instead of failing completely
                        continue
            
            # Log final comprehensive session metrics
            if parent_run:
                final_session_metrics = {
                    "total_models_trained": len(models),
                    "training_success_rate": len(models) / len(work_types) if len(work_types) > 0 else 0,
                    "feature_selection_method": "k_fold_time_series_enhanced",
                    "avg_feature_reduction": np.mean([metrics[wt].get('feature_reduction_pct', 0) for wt in metrics]) if metrics else 0,
                    "avg_mae": np.mean([metrics[wt].get('MAE', 0) for wt in metrics]) if metrics else 0,
                    "avg_r2": np.mean([metrics[wt].get('R²', 0) for wt in metrics]) if metrics else 0
                }
                mlflow_manager.log_training_parameters(final_session_metrics)
            
            enterprise_logger.info(f"🎉 K-Fold Enhanced training completed successfully:")
            enterprise_logger.info(f"   Models trained: {len(models)}/{len(work_types)}")
            enterprise_logger.info(f"   Success rate: {len(models)/len(work_types)*100:.1f}%")
            if metrics:
                enterprise_logger.info(f"   Average MAE: {np.mean([m.get('MAE', 0) for m in metrics.values()]):.4f}")
                enterprise_logger.info(f"   Average R²: {np.mean([m.get('R²', 0) for m in metrics.values()]):.4f}")
            
            return models, feature_importances, metrics
        
    except Exception as e:
        logger.error(f"Error building K-fold enhanced models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, {}

def train_from_sql(connection_string=None, sql_query=None):
    """
    Train enhanced models using data from SQL with K-fold time series feature selection
    
    Parameters:
    -----------
    connection_string : str, optional
        SQL Server connection string
    sql_query : str, optional
        SQL query to execute
        
    Returns:
    --------
    tuple
        (models, feature_importances, metrics)
    """
    try:
        import pyodbc
        
        # Use default connection string from config if not provided
        if connection_string is None:
            connection_string = (
                f"DRIVER={{SQL Server}};"
                f"SERVER={SQL_SERVER};"
                f"DATABASE={SQL_DATABASE};"
            )
            
            if SQL_TRUSTED_CONNECTION:
                connection_string += "Trusted_Connection=yes;"
            else:
                connection_string += f"UID={SQL_USERNAME};PWD={SQL_PASSWORD};"
        
        # Default enhanced query if none provided
        if sql_query is None:
            sql_query = """
                SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
                FROM WorkUtilizationData 
                WHERE PunchCode IN (206, 213) 
                AND Hours > 0 
                AND NoOfMan > 0 
                AND SystemHours > 0 
                AND Quantity > 0
                AND Date > '2025-05-01'
                ORDER BY Date
            """
        
        logger.info(f"Connecting to database {SQL_DATABASE} on server {SQL_SERVER} for enhanced training")
        conn = pyodbc.connect(connection_string)
        logger.info(f"Executing SQL query for enhanced K-fold training")
        
        # Handle large datasets with chunking for memory efficiency
        chunks = []
        for chunk in pd.read_sql(sql_query, conn, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
            logger.info(f"Read chunk of {len(chunk)} rows")
        
        conn.close()
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Data loaded successfully from SQL. Total rows: {len(df)}")
        else:
            logger.warning("No data returned from SQL query")
            return None, None, None
        
        # Handle PunchCode as WorkType mapping
        if 'PunchCode' in df.columns and 'WorkType' not in df.columns:
            df = df.rename(columns={'PunchCode': 'WorkType'})
            logger.info("Mapped 'PunchCode' column to 'WorkType'")
        
        # Ensure Date column is properly converted to datetime
        if 'Date' in df.columns:
            logger.info("Converting Date column to datetime format")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Check for and remove null dates
            null_dates = df['Date'].isna().sum()
            if null_dates > 0:
                logger.warning(f"Found {null_dates} null dates after conversion. Removing these rows.")
                df = df.dropna(subset=['Date'])
        
        # Ensure WorkType is treated as string for consistency
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Ensure all numeric columns are properly formatted
        numeric_columns = ['NoOfMan', 'Hours', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Process data using enhanced feature engineering pipeline
        logger.info("Engineering features with enhanced configuration...")
        feature_df = engineer_features(df)

        logger.info("Creating lag features with K-fold optimized configuration...")
        lag_features_df = create_lag_features(
            feature_df,
            lag_days=ESSENTIAL_LAGS,
            rolling_windows=ESSENTIAL_WINDOWS
        )
        
        work_types = lag_features_df['WorkType'].unique()
        logger.info(f"Found {len(work_types)} unique work types for enhanced training")
        
        # Use enhanced K-fold model building
        models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
        
        # Save enhanced models
        save_models(models, feature_importances, metrics)
        
        return models, feature_importances, metrics
        
    except Exception as e:
        logger.error(f"Error in enhanced SQL training: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def save_models(models, feature_importances, metrics):
    """
    Save trained models with enhanced metadata and K-fold results
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    feature_importances : dict
        Dictionary of feature importances
    metrics : dict
        Dictionary of model metrics including K-fold statistics
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        logger.info(f"Saving {len(models)} enhanced models with K-fold metadata to {MODELS_DIR}")
        
        # Save core model files
        with open(os.path.join(MODELS_DIR, "work_utilization_models.pkl"), "wb") as f:
            pickle.dump(models, f)
            
        with open(os.path.join(MODELS_DIR, "work_utilization_feature_importances.pkl"), "wb") as f:
            pickle.dump(feature_importances, f)
            
        with open(os.path.join(MODELS_DIR, "work_utilization_metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
        
        # Create comprehensive performance summary with enhanced metrics
        performance_summary = []
        for work_type, metric in metrics.items():
            # Get selected features count and other metadata
            n_features_final = metric.get('n_features_final', 'N/A')
            n_features_initial = metric.get('n_features_initial', 'N/A')
            feature_reduction = metric.get('feature_reduction_pct', 0)
            
            # Check if model has fold results stored
            fold_results_available = hasattr(models.get(work_type, {}), 'fold_results_')
            
            performance_summary.append({
                'WorkType': work_type,
                'MAE': metric.get('MAE', np.nan),
                'MAE_std': metric.get('MAE_std', np.nan),
                'RMSE': metric.get('RMSE', np.nan),
                'RMSE_std': metric.get('RMSE_std', np.nan),
                'R²': metric.get('R²', np.nan),
                'R²_std': metric.get('R²_std', np.nan),
                'MAPE': metric.get('MAPE', np.nan),
                'MAPE_std': metric.get('MAPE_std', np.nan),
                'CV_MAE': metric.get('CV_MAE', np.nan),
                'Features_Initial': n_features_initial,
                'Features_Final': n_features_final,
                'Feature_Reduction_%': feature_reduction,
                'K_Fold_Results_Available': fold_results_available,
                'Training_Method': 'K_Fold_Time_Series_Enhanced'
            })
        
        performance_df = pd.DataFrame(performance_summary)
        
        # Sort by MAE for easy analysis
        performance_df = performance_df.sort_values('MAE')
        
        performance_file = os.path.join(MODELS_DIR, "enhanced_k_fold_model_performance.xlsx")
        performance_df.to_excel(performance_file, index=False)
        
        # Save enhanced configuration info with K-fold details
        config_info = {
            'feature_groups': FEATURE_GROUPS,
            'active_groups': [group for group, enabled in FEATURE_GROUPS.items() if enabled],
            'training_date': datetime.now().isoformat(),
            'models_count': len(models),
            'feature_selection_enabled': True,
            'feature_selection_method': 'k_fold_time_series',
            'essential_lags': ESSENTIAL_LAGS,
            'essential_windows': ESSENTIAL_WINDOWS,
            'model_params': DEFAULT_MODEL_PARAMS,
            'validation_method': 'TimeSeriesSplit',
            'n_splits': 5,
            'performance_summary': {
                'avg_mae': performance_df['MAE'].mean() if not performance_df.empty else 0,
                'avg_r2': performance_df['R²'].mean() if not performance_df.empty else 0,
                'avg_feature_reduction': performance_df['Feature_Reduction_%'].mean() if not performance_df.empty else 0,
                'models_with_low_mae': len(performance_df[performance_df['MAE'] < 0.5]) if not performance_df.empty else 0
            }
        }
        
        config_file = os.path.join(MODELS_DIR, "enhanced_k_fold_training_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_info, f, indent=2, default=str)  # default=str to handle datetime serialization
        
        # Save detailed fold results for each model (for advanced analysis)
        fold_results_summary = {}
        for work_type, model in models.items():
            if hasattr(model, 'fold_results_'):
                fold_results_summary[work_type] = model.fold_results_
        
        if fold_results_summary:
            fold_results_file = os.path.join(MODELS_DIR, "k_fold_detailed_results.json")
            with open(fold_results_file, 'w') as f:
                json.dump(fold_results_summary, f, indent=2, default=str)
            logger.info(f"Detailed K-fold results saved to {fold_results_file}")
        
        # Log comprehensive save completion
        logger.info(f"✅ Enhanced model files saved successfully:")
        logger.info(f"   📊 Performance summary: {performance_file}")
        logger.info(f"   ⚙️  Training configuration: {config_file}")
        logger.info(f"   📈 Models with MAE < 0.5: {config_info['performance_summary']['models_with_low_mae']}/{len(models)}")
        logger.info(f"   🎯 Average feature reduction: {config_info['performance_summary']['avg_feature_reduction']:.1f}%")
        
        return True
    except Exception as e:
        logger.error(f"Error saving enhanced models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main function to run the complete enhanced training process with K-fold time series validation
    """
    try:
        logger.info("🚀 Starting COMPLETE K-FOLD ENHANCED model training with time series feature selection")
        
        # Log comprehensive feature group configuration
        active_groups = [group for group, enabled in FEATURE_GROUPS.items() if enabled]
        enterprise_logger.info(f"Active feature groups: {active_groups}")
        enterprise_logger.info(f"Essential lags: {ESSENTIAL_LAGS}")
        enterprise_logger.info(f"Essential windows: {ESSENTIAL_WINDOWS}")
        enterprise_logger.info(f"Model parameters: {DEFAULT_MODEL_PARAMS}")
        
        # Check command line arguments for data source
        import sys
        use_sql = len(sys.argv) > 1 and sys.argv[1].lower() == 'sql'
        
        if use_sql:
            logger.info("🔗 Training from SQL database with K-fold enhanced feature selection")
            # Train using SQL data with enhanced K-fold features
            result = train_from_sql()
            if result[0] is None:
                logger.error("❌ Enhanced SQL training failed. Check logs for details.")
                return False
            else:
                logger.info("✅ SQL training completed successfully")
        else:
            logger.info("📁 Training from file with K-fold enhanced feature selection")
            # Train using file data
            file_path = os.path.join(DATA_DIR, "work_utilization_melted1.csv")
            
            # Check if default file exists
            if not os.path.exists(file_path):
                logger.warning(f"Default file not found: {file_path}")
                file_path = input("Enter path to data file (.xlsx or .csv): ")
                
                # Validate user-provided file path
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    return False
            
            # Load and process data with complete enhanced pipeline
            logger.info("📊 Loading and processing data...")
            df = load_data(file_path)
            
            logger.info("🔧 Engineering features...")
            feature_df = engineer_features(df)
            
            logger.info("⏱️  Creating lag features with enhanced configuration...")
            lag_features_df = create_lag_features(
                feature_df,
                lag_days=ESSENTIAL_LAGS,
                rolling_windows=ESSENTIAL_WINDOWS
            )
            
            # Get work types for training
            work_types = lag_features_df['WorkType'].unique()
            logger.info(f"Found {len(work_types)} unique work types: {sorted(work_types)}")
            
            # Build enhanced models with complete K-fold pipeline
            logger.info("🤖 Building K-fold enhanced models...")
            models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
            
            # Save comprehensive results
            logger.info("💾 Saving enhanced models and metadata...")
            save_success = save_models(models, feature_importances, metrics)
            
            if not save_success:
                logger.error("❌ Failed to save models")
                return False
            else:
                logger.info("✅ File training completed successfully")
        
        # Log final comprehensive summary
        enterprise_logger.info("🎉 COMPLETE K-FOLD ENHANCED TRAINING FINISHED SUCCESSFULLY")
        enterprise_logger.info("=" * 80)
        enterprise_logger.info("📈 TRAINING SUMMARY:")
        enterprise_logger.info(f"   🎯 Feature Selection: K-Fold Time Series Cross-Validation")
        enterprise_logger.info(f"   📊 Validation Method: {DEFAULT_MODEL_PARAMS.get('n_estimators', 500)}-tree RandomForest with {5}-fold TimeSeriesSplit")
        enterprise_logger.info(f"   🔧 Feature Engineering: {len(active_groups)} active feature groups")
        enterprise_logger.info(f"   💾 Models Saved: Enhanced with fold-level metadata")
        enterprise_logger.info(f"   📁 Results Location: {MODELS_DIR}")
        enterprise_logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error in complete K-fold enhanced training process: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Cleanup MLflow resources
        if 'mlflow_manager' in globals():
            try:
                mlflow_manager.cleanup()
                logger.info("🧹 MLflow resources cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"MLflow cleanup warning: {cleanup_error}")

if __name__ == "__main__":
    # Set up proper logging directory
    os.makedirs("logs", exist_ok=True)
    
    # Run main training process
    success = main()
    
    # Exit with appropriate code
    if success:
        logger.info("🎯 Enhanced K-fold training completed successfully!")
        exit(0)
    else:
        logger.error("❌ Enhanced K-fold training failed!")
        exit(1)