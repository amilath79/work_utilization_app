"""
Enhanced Model Training for Punch Codes 206 & 213
Enterprise-Grade Time Series Model Training with Advanced Feature Engineering
Uses Complete Pipeline Approach Only
"""

# import pandas as pd
# import numpy as np
# from datetime import datetime
# import pickle
# import os
# import logging
# import traceback
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import TimeSeriesSplit
# import json


import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import traceback
from lightgbm import LGBMRegressor                    # Replace RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import json
<<<<<<< HEAD
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from lightgbm import early_stopping, log_evaluation

=======
from sklearn.feature_selection import RFECV
from lightgbm import log_evaluation, early_stopping, LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
>>>>>>> 7f9e72d (2025-06-23  05 commit)
# IMPACT: LightGBM import for superior gradient boosting performance



# Import utilities - PIPELINE APPROACH ONLY
from utils.feature_engineering import EnhancedFeatureTransformer
from utils.holiday_utils import is_non_working_day
from utils.sql_data_connector import extract_sql_data
from config import ENHANCED_WORK_TYPES

from config import (
    MODELS_DIR,
    DEFAULT_MODEL_PARAMS,
    FEATURE_GROUPS,
    ESSENTIAL_LAGS, 
    ESSENTIAL_WINDOWS,
    SQL_SERVER, 
    SQL_DATABASE, 
    SQL_TRUSTED_CONNECTION
)


# Enhanced time series cross-validation with gap
from sklearn.model_selection import TimeSeriesSplit
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
class GapTimeSeriesSplit(TimeSeriesSplit):
    """TimeSeriesSplit with gap to prevent data leakage"""
    def __init__(self, n_splits=10, gap=7):
        super().__init__(n_splits=n_splits)
        self.gap = gap
    
    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in super().split(X, y, groups):
            # Add gap between train and test
            if len(train_idx) > self.gap:
                train_idx = train_idx[:-self.gap]
            yield train_idx, test_idx

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
        logger.info(f"Loading training data for enhanced models {ENHANCED_WORK_TYPES}")
        
        query = """
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
		CASE WHEN PunchCode IN (206, 213) THEN NoRows
		ELSE Quantity END as Quantity, 
		SystemKPI
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
		AND Hours > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        AND Date < '2025-05-06'
        ORDER BY Date
        """
        
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is None or df.empty:
            logger.error("No data returned from SQL query")
            return None
        
        # Convert date column and work type
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        
        logger.info(f"Loaded {len(df)} records for enhanced training")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None
def add_target_lags_before_pipeline(df, work_type):
    """
    Add Hours lag features before pipeline processing
    """
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Create Hours lag features
    for lag in ESSENTIAL_LAGS:
        df[f'Hours_lag_{lag}'] = df['Hours'].shift(lag).fillna(0)
    
    # Create Hours rolling features
    for window in ESSENTIAL_WINDOWS:
        df[f'Hours_rolling_mean_{window}'] = (
            df['Hours'].rolling(window=window, min_periods=1).mean()
        )
        df[f'Hours_rolling_std_{window}'] = (
            df['Hours'].rolling(window=window, min_periods=2).std().fillna(0)
        )
    
    return df


# Add these imports at the top of train_models2.py
from lightgbm import early_stopping, log_evaluation

def train_enhanced_model(df, work_type):
    """
    Train enhanced model with loss logging, feature importance, and RFE
    """
    try:
<<<<<<< HEAD
        logger.info(f"Training enhanced LightGBM model for WorkType {work_type} using complete pipeline")
        df = detect_and_handle_outliers(df, 'Hours', n_std=3)  # Changed to 3 for better outlier handling
=======
        logger.info(f"Training enhanced LightGBM model for WorkType {work_type} with comprehensive analysis")
        df = detect_and_handle_outliers(df, 'Hours', n_std=3)
        
        # Prepare target
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        y = df['Hours'].values
        
        # Import LightGBM utilities
        from utils.lightgbm_utils import optimize_lightgbm_for_worktype, validate_lightgbm_params
        
<<<<<<< HEAD
        # The pipeline will handle all feature engineering
        basic_features = ['Date', 'WorkType', 'Quantity', 'SystemHours', 'SystemKPI']
=======
        # Basic features for pipeline

        # Add target lags before pipeline
        df = add_target_lags_before_pipeline(df, work_type)
        
        # Now include the lag features in basic_features
        hours_lag_features = [f'Hours_lag_{lag}' for lag in ESSENTIAL_LAGS]
        hours_rolling_features = []
        for window in ESSENTIAL_WINDOWS:
            hours_rolling_features.extend([
                f'Hours_rolling_mean_{window}',
                f'Hours_rolling_std_{window}'
            ])

        basic_features = ['Date', 'WorkType', 'Quantity', 'SystemHours', 'SystemKPI'] + \
                    hours_lag_features + hours_rolling_features
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        available_basic = [f for f in basic_features if f in df.columns]
        X_basic = df[available_basic].copy()
        
        # First, create full feature set to perform RFE
        logger.info("Creating full feature set for RFE analysis...")
        feature_transformer = EnhancedFeatureTransformer()
        X_full_features = feature_transformer.fit_transform(X_basic)
        feature_names = X_full_features.columns.tolist()
        
        # STEP 1: Perform RFE-CV to find optimal features
        optimal_features, feature_ranking, rfe_results = perform_rfe_cv(
            X_full_features, y, work_type, n_features_to_select=10, cv_splits=3
        )
        
        # Store RFE results for later use
        rfe_metadata = {
            'optimal_features': optimal_features,
            'feature_ranking': feature_ranking,
            'n_features_tested': len(feature_names)
        }
        
        # STEP 2: Optimize LightGBM parameters
        optimized_params = optimize_lightgbm_for_worktype(X_basic, y, work_type)
        validated_params = validate_lightgbm_params({**DEFAULT_MODEL_PARAMS, **optimized_params})
        
<<<<<<< HEAD
        # Time series cross-validation
        tscv = GapTimeSeriesSplit(n_splits=5, gap=14)  # Reduced splits, increased gap
        fold_scores = []
        train_scores = []  # Track training scores to detect overfitting
=======
        # Initialize storage for cross-validation results
        tscv = GapTimeSeriesSplit(n_splits=5, gap=14)
        fold_scores = []
        train_scores = []
        all_feature_importances = []
        all_loss_histories = []
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        
        logger.info("Performing time series cross-validation with detailed logging...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_basic)):
<<<<<<< HEAD
            # Skip if validation set too small
            if len(val_idx) < 30:
                logger.warning(f"Skipping fold {fold+1}: validation set too small ({len(val_idx)} samples)")
=======
            if len(val_idx) < 30:
                logger.warning(f"Skipping fold {fold+1}: validation set too small")
>>>>>>> 7f9e72d (2025-06-23  05 commit)
                continue
                
            X_train_fold = X_basic.iloc[train_idx]
            X_val_fold = X_basic.iloc[val_idx] 
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
<<<<<<< HEAD
            # Create a NEW pipeline for each fold (important!)
=======
            # Create pipeline for this fold
>>>>>>> 7f9e72d (2025-06-23  05 commit)
            fold_pipeline = Pipeline([
                ('feature_engineering', EnhancedFeatureTransformer()),
                ('model', LGBMRegressor(**validated_params))
            ])
            
<<<<<<< HEAD
            # Fit feature engineering on training fold
=======
            # Transform features
>>>>>>> 7f9e72d (2025-06-23  05 commit)
            feature_eng = fold_pipeline.named_steps['feature_engineering']
            X_train_transformed = feature_eng.fit_transform(X_train_fold)
            X_val_transformed = feature_eng.transform(X_val_fold)
            
<<<<<<< HEAD
            # Train the model with early stopping
=======
            # Create logging callback
            loss_logger = DetailedLoggingCallback()
            
            # Train model with detailed logging
>>>>>>> 7f9e72d (2025-06-23  05 commit)
            lgb_model = fold_pipeline.named_steps['model']
            lgb_model.fit(
                X_train_transformed, 
                y_train_fold,
<<<<<<< HEAD
                eval_set=[(X_val_transformed, y_val_fold)],
                callbacks=[
                    early_stopping(stopping_rounds=20, verbose=False),
                    log_evaluation(period=0)  # Suppress output
                ]
            )
            
            # Predict on validation using the fitted model
            y_pred_val = lgb_model.predict(X_val_transformed)
            
            # Also predict on training to check overfitting
=======
                eval_set=[(X_train_transformed, y_train_fold), 
                          (X_val_transformed, y_val_fold)],
                eval_names=['training', 'valid_0'],  # Explicit names
                callbacks=[
                    early_stopping(stopping_rounds=20, verbose=False),
                    loss_logger
                ]
            )
            
            # Store loss history
            all_loss_histories.append({
                'fold': fold + 1,
                'train_losses': loss_logger.train_losses,
                'val_losses': loss_logger.val_losses,
                'iterations': loss_logger.iterations,
                'best_iteration': lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else len(loss_logger.iterations)
            })
            
            # Calculate feature importance for this fold
            feature_importance = calculate_feature_importance(
                lgb_model, 
                X_train_transformed.columns.tolist(),
                importance_type='gain'
            )
            all_feature_importances.append(feature_importance)
            
            # Make predictions
            y_pred_val = lgb_model.predict(X_val_transformed)
>>>>>>> 7f9e72d (2025-06-23  05 commit)
            y_pred_train = lgb_model.predict(X_train_transformed)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_fold, y_pred_train)
<<<<<<< HEAD
            val_mae = mean_absolute_error(y_val_fold, y_pred_val)
=======
            train_loss = mean_squared_error(y_train_fold, y_pred_train)
            val_mae = mean_absolute_error(y_val_fold, y_pred_val)
            val_loss = mean_squared_error(y_val_fold, y_pred_val)
>>>>>>> 7f9e72d (2025-06-23  05 commit)
            val_r2 = r2_score(y_val_fold, y_pred_val)
            
            # Calculate overfitting ratio
            overfit_ratio = train_mae / val_mae if val_mae > 0 else 0
            
<<<<<<< HEAD
            fold_scores.append({'MAE': val_mae, 'R2': val_r2})
            train_scores.append({'train_MAE': train_mae, 'overfit_ratio': overfit_ratio})
            
            logger.info(f"  Fold {fold+1}: Train MAE={train_mae:.3f}, Val MAE={val_mae:.3f}, "
                       f"R²={val_r2:.3f}, Overfit Ratio={overfit_ratio:.3f}")
            
            # Warning if severe overfitting detected
            if overfit_ratio < 0.5:
                logger.warning(f"  ⚠️ Severe overfitting detected in fold {fold+1}")
        
        # Check overall overfitting
        avg_overfit_ratio = np.mean([s['overfit_ratio'] for s in train_scores]) if train_scores else 1.0
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores]) if fold_scores else 0
        
        # If still overfitting, increase regularization
        if avg_overfit_ratio < 0.7 or avg_cv_r2 > 0.95:
            logger.warning(f"⚠️ Model shows signs of overfitting (avg ratio: {avg_overfit_ratio:.3f}, R²: {avg_cv_r2:.3f})")
            validated_params['lambda_l1'] *= 2
            validated_params['lambda_l2'] *= 2
            validated_params['num_leaves'] = max(10, validated_params.get('num_leaves', 31) // 2)
            logger.info("  Increased regularization parameters")
        
        # Train final pipeline on all data with adjusted parameters
=======
            fold_scores.append({
                'MAE': val_mae, 
                'R2': val_r2,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            train_scores.append({
                'train_MAE': train_mae, 
                'overfit_ratio': overfit_ratio
            })
            
            logger.info(f"  Fold {fold+1} Summary:")
            logger.info(f"    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"    Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}")
            logger.info(f"    R²: {val_r2:.3f}, Overfit Ratio: {overfit_ratio:.3f}")
            logger.info(f"    Best Iteration: {lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else 'N/A'}")
        
        # STEP 3: Calculate average feature importance across folds
        logger.info("Calculating average feature importance across all folds...")
        avg_feature_importance = {}
        all_features = set()
        for fold_importance in all_feature_importances:
            all_features.update(fold_importance.keys())
        
        for feature in all_features:
            importances = [fold_imp.get(feature, 0) for fold_imp in all_feature_importances]
            avg_feature_importance[feature] = np.mean(importances)
        
        # Sort by average importance
        sorted_avg_importance = dict(sorted(avg_feature_importance.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
        
        # Log top 20 features
        logger.info("📊 Top 20 Most Important Features (averaged across CV folds):")
        for i, (feature, importance) in enumerate(list(sorted_avg_importance.items())[:20], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
        
        # STEP 4: Train final model on all data
        avg_overfit_ratio = np.mean([s['overfit_ratio'] for s in train_scores]) if train_scores else 1.0
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores]) if fold_scores else 0
        
        # Adjust parameters if overfitting detected
        if avg_overfit_ratio < 0.7 or avg_cv_r2 > 0.95:
            logger.warning(f"⚠️ Overfitting detected. Increasing regularization...")
            validated_params['lambda_l1'] *= 2
            validated_params['lambda_l2'] *= 2
            validated_params['num_leaves'] = max(10, validated_params.get('num_leaves', 31) // 2)
        
        # Create final pipeline
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        logger.info("Training final pipeline on all data...")
        complete_pipeline = Pipeline([
            ('feature_engineering', EnhancedFeatureTransformer()),
            ('model', LGBMRegressor(**validated_params))
        ])
        
<<<<<<< HEAD
        # For final training, use validation split for early stopping
=======
        # Split for final training with validation
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        val_size = int(len(X_basic) * 0.2)
        X_train_final = X_basic.iloc[:-val_size]
        y_train_final = y[:-val_size]
        X_val_final = X_basic.iloc[-val_size:]
        y_val_final = y[-val_size:]
        
<<<<<<< HEAD
        # Fit feature engineering on training data
=======
        # Transform features
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        fe = complete_pipeline.named_steps['feature_engineering']
        X_train_transformed_final = fe.fit_transform(X_train_final)
        X_val_transformed_final = fe.transform(X_val_final)
        
<<<<<<< HEAD
        # Train model with early stopping
=======
        # Create final loss logger
        final_loss_logger = DetailedLoggingCallback()
        
        # Train final model
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        final_model = complete_pipeline.named_steps['model']
        final_model.fit(
            X_train_transformed_final,
            y_train_final,
<<<<<<< HEAD
            eval_set=[(X_val_transformed_final, y_val_final)],
            callbacks=[
                early_stopping(stopping_rounds=20, verbose=True),
=======
            eval_set=[(X_train_transformed_final, y_train_final, 'training'),
                      (X_val_transformed_final, y_val_final)],
            eval_metric='l2',
            callbacks=[
                early_stopping(stopping_rounds=20, verbose=True),
                final_loss_logger,
>>>>>>> 7f9e72d (2025-06-23  05 commit)
                log_evaluation(period=10)
            ]
        )
        
<<<<<<< HEAD
        # Final evaluation on ALL data
=======
        # Calculate final feature importance
        final_feature_importance = calculate_feature_importance(
            final_model,
            X_train_transformed_final.columns.tolist(),
            importance_type='gain'
        )
        
        # Final evaluation
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        y_pred_final = complete_pipeline.predict(X_basic)
        final_mae = mean_absolute_error(y, y_pred_final)
        final_r2 = r2_score(y, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))
        mape = np.mean(np.abs((y - y_pred_final) / np.where(y == 0, 1, y))) * 100
        
<<<<<<< HEAD
        # Calculate average CV metrics
        avg_cv_mae = np.mean([score['MAE'] for score in fold_scores]) if fold_scores else final_mae
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores]) if fold_scores else final_r2
        
        # Create metadata
=======
        # Create comprehensive metadata
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        model_metadata = {
            'work_type': work_type,
            'training_records': len(df),
            'final_mae': final_mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'mape': mape,
            'cv_mae': np.mean([s['MAE'] for s in fold_scores]) if fold_scores else final_mae,
            'cv_r2': avg_cv_r2,
            'cv_folds': len(fold_scores),
            'avg_train_loss': np.mean([s['train_loss'] for s in fold_scores]),
            'avg_val_loss': np.mean([s['val_loss'] for s in fold_scores]),
            'input_features': basic_features,
            'pipeline_steps': [step[0] for step in complete_pipeline.steps],
            'model_type': 'complete_pipeline',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
<<<<<<< HEAD
            'regularization_applied': avg_overfit_ratio < 0.7 or avg_cv_r2 > 0.95
        }
        
        # Extract LightGBM-specific information
        lgb_model = complete_pipeline.named_steps['model']
        n_estimators_used = getattr(lgb_model, 'best_iteration_', lgb_model.n_estimators)
        
        logger.info(f"✅ Enhanced LightGBM pipeline trained for {work_type}")
        logger.info(f"   Final MAE: {final_mae:.3f}")
        logger.info(f"   Final R²: {final_r2:.3f}")
        logger.info(f"   CV MAE: {avg_cv_mae:.3f}")
        logger.info(f"   CV R²: {avg_cv_r2:.3f}")
        logger.info(f"   MAPE: {mape:.2f}%")
        logger.info(f"   Trees used: {n_estimators_used}/{lgb_model.n_estimators}")
=======
            'regularization_applied': avg_overfit_ratio < 0.7 or avg_cv_r2 > 0.95,
            'feature_importance': dict(list(sorted_avg_importance.items())[:50]),  # Top 50
            'final_feature_importance': dict(list(final_feature_importance.items())[:50]),
            'loss_histories': all_loss_histories,
            'rfe_results': rfe_metadata,
            'n_trees_used': final_model.best_iteration_ if hasattr(final_model, 'best_iteration_') else final_model.n_estimators
        }
        
        # Create loss history plot
        create_loss_history_plot(all_loss_histories, work_type)
        
        # Create feature importance plot
        create_feature_importance_plot(sorted_avg_importance, work_type)
        
        logger.info(f"✅ Enhanced model trained successfully for {work_type}")
        logger.info(f"   Final Metrics: MAE={final_mae:.3f}, R²={final_r2:.3f}, MAPE={mape:.2f}%")
        logger.info(f"   Trees used: {model_metadata['n_trees_used']}")
>>>>>>> 7f9e72d (2025-06-23  05 commit)
        
        return complete_pipeline, model_metadata, basic_features
        
    except Exception as e:
        logger.error(f"Error training enhanced model for {work_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None


def save_enhanced_models(models, metadata, features, df):
    """
    Save enhanced models and metadata with comprehensive analysis results
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models by work type
    metadata : dict
        Dictionary of metadata including loss histories, feature importance, RFE results
    features : dict
        Dictionary of feature lists by work type
    df : DataFrame
        Training data
    """
    try:
        logger.info("💾 Saving enhanced models and comprehensive metadata")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models with enhanced naming
        for work_type, model in models.items():
            if model is not None:
                model_filename = f"enhanced_model_{work_type}.pkl"
                model_path = os.path.join(MODELS_DIR, model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"  ✅ Saved model for {work_type}: {model_filename}")
                
                # Update metadata with model filename
                if work_type in metadata:
                    metadata[work_type]['model_filename'] = model_filename
        
        # Save comprehensive metadata with all analysis results
        metadata_filename = f"enhanced_models_metadata_{timestamp}.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        
        # Convert numpy arrays and non-serializable objects for JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return str(obj)
        
        # Prepare metadata for JSON serialization
        json_metadata = convert_for_json(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        logger.info(f"  ✅ Saved comprehensive metadata: {metadata_filename}")
        
        # Save features mapping
        features_filename = f"enhanced_features_{timestamp}.json"
        features_path = os.path.join(MODELS_DIR, features_filename)
        
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        logger.info(f"  ✅ Saved features mapping: {features_filename}")
        
        # Save detailed analysis report (human-readable format)
        report_filename = f"training_analysis_report_{timestamp}.txt"
        report_path = os.path.join("logs", report_filename)
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ENHANCED MODEL TRAINING ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for work_type, meta in metadata.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"WORK TYPE: {work_type}\n")
                f.write(f"{'='*60}\n\n")
                
                # Model Performance Summary
                f.write("MODEL PERFORMANCE SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Final MAE: {meta.get('final_mae', 'N/A'):.3f}\n")
                f.write(f"Final R²: {meta.get('final_r2', 'N/A'):.3f}\n")
                f.write(f"Final RMSE: {meta.get('final_rmse', 'N/A'):.3f}\n")
                f.write(f"MAPE: {meta.get('mape', 'N/A'):.2f}%\n")
                f.write(f"Training Records: {meta.get('training_records', 'N/A')}\n")
                f.write(f"Trees Used: {meta.get('n_trees_used', 'N/A')}\n\n")
                
                # Cross-Validation Summary
                f.write("CROSS-VALIDATION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"CV Folds: {meta.get('cv_folds', 'N/A')}\n")
                f.write(f"Average CV MAE: {meta.get('cv_mae', 'N/A'):.3f}\n")
                f.write(f"Average CV R²: {meta.get('cv_r2', 'N/A'):.3f}\n")
                f.write(f"Avg Train Loss: {meta.get('avg_train_loss', 'N/A'):.4f}\n")
                f.write(f"Avg Val Loss: {meta.get('avg_val_loss', 'N/A'):.4f}\n")
                f.write(f"Regularization Applied: {meta.get('regularization_applied', False)}\n\n")
                
                # RFE Results
                if 'rfe_results' in meta and meta['rfe_results']:
                    f.write("RECURSIVE FEATURE ELIMINATION RESULTS:\n")
                    f.write("-" * 40 + "\n")
                    rfe = meta['rfe_results']
                    f.write(f"Features Tested: {rfe.get('n_features_tested', 'N/A')}\n")
                    f.write(f"Optimal Features: {len(rfe.get('optimal_features', []))}\n")
                    if rfe.get('optimal_features'):
                        f.write("Selected Features:\n")
                        for i, feat in enumerate(rfe['optimal_features'][:10], 1):
                            f.write(f"  {i}. {feat}\n")
                        if len(rfe['optimal_features']) > 10:
                            f.write(f"  ... and {len(rfe['optimal_features'])-10} more\n")
                    f.write("\n")
                
                # Top Feature Importances
                if 'feature_importance' in meta and meta['feature_importance']:
                    f.write("TOP 20 FEATURE IMPORTANCES (CV Averaged):\n")
                    f.write("-" * 40 + "\n")
                    for i, (feat, imp) in enumerate(list(meta['feature_importance'].items())[:20], 1):
                        f.write(f"{i:2d}. {feat:<40} {imp:>10.4f}\n")
                    f.write("\n")
                
                # Loss History Summary
                if 'loss_histories' in meta and meta['loss_histories']:
                    f.write("LOSS HISTORY SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    for fold_hist in meta['loss_histories']:
                        fold_num = fold_hist.get('fold', 'N/A')
                        best_iter = fold_hist.get('best_iteration', 'N/A')
                        if fold_hist.get('train_losses') and fold_hist.get('val_losses'):
                            final_train_loss = fold_hist['train_losses'][-1]
                            final_val_loss = fold_hist['val_losses'][-1]
                            f.write(f"Fold {fold_num}: Best Iter={best_iter}, "
                                   f"Final Train Loss={final_train_loss:.4f}, "
                                   f"Final Val Loss={final_val_loss:.4f}\n")
                    f.write("\n")
        
        logger.info(f"  ✅ Saved training analysis report: {report_filename}")
        
        # Save training data for predictions (with timestamp for versioning)
        try:
            training_data_filename = f'enhanced_training_data_{timestamp}.pkl'
            training_data_path = os.path.join(MODELS_DIR, training_data_filename)
            df.to_pickle(training_data_path)
            logger.info(f"  ✅ Enhanced training data saved: {training_data_filename}")
            
            # Also save as latest for easy access
            latest_training_data_path = os.path.join(MODELS_DIR, 'enhanced_training_data_latest.pkl')
            df.to_pickle(latest_training_data_path)
            logger.info(f"  ✅ Latest training data link created")
            
        except Exception as e:
            logger.error(f"  ⚠️ Failed to save training data: {str(e)}")
        
        # Create summary CSV for quick reference
        summary_data = []
        for work_type, meta in metadata.items():
            summary_data.append({
                'WorkType': work_type,
                'FinalMAE': meta.get('final_mae', None),
                'FinalR2': meta.get('final_r2', None),
                'MAPE': meta.get('mape', None),
                'CVFolds': meta.get('cv_folds', None),
                'CVMAE': meta.get('cv_mae', None),
                'CVR2': meta.get('cv_r2', None),
                'TrainingRecords': meta.get('training_records', None),
                'TreesUsed': meta.get('n_trees_used', None),
                'Timestamp': timestamp
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_filename = f"model_performance_summary_{timestamp}.csv"
            summary_path = os.path.join("logs", summary_filename)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"  ✅ Saved performance summary CSV: {summary_filename}")
        
        # Log success summary
        logger.info(f"\n🎉 ENHANCED MODEL SAVING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"✅ Models saved: {len(models)}")
        logger.info(f"✅ Timestamp: {timestamp}")
        logger.info(f"✅ Models directory: {MODELS_DIR}")
        logger.info(f"✅ Logs directory: logs/")
        logger.info("=" * 60)
        
        # Print performance summary to console
        logger.info("\n📊 PERFORMANCE SUMMARY:")
        for work_type, meta in metadata.items():
            logger.info(f"\n{work_type}:")
            logger.info(f"  MAE: {meta.get('final_mae', 'N/A'):.3f}")
            logger.info(f"  R²: {meta.get('final_r2', 'N/A'):.3f}")
            logger.info(f"  MAPE: {meta.get('mape', 'N/A'):.2f}%")
            logger.info(f"  Top Features: {list(meta.get('feature_importance', {}).keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving enhanced models: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
def preprocess_by_punch_code(df):
    """Special handling for problematic punch codes"""
    df_processed = df.copy()
    
    # Punch 210 & 217: Apply log transformation for extreme outliers
    for punch_code in [210, 217]:
        mask = df_processed['WorkType'] == punch_code
        if mask.any():
            # Log transform to reduce extreme values
            df_processed.loc[mask, 'Hours'] = np.log1p(df_processed.loc[mask, 'Hours'])
            
            # Cap extreme outliers at 99th percentile
            p99 = df_processed.loc[mask, 'Hours'].quantile(0.99)
            df_processed.loc[mask, 'Hours'] = np.minimum(
                df_processed.loc[mask, 'Hours'], p99
            )
    
    return df_processed

def remove_extreme_outliers(X, y):
    """Aggressive outlier removal for punch codes 210 & 217"""
    mask = np.ones(len(X), dtype=bool)
    
    # Standard outlier removal for most punch codes
    for punch_code in [202, 203, 206, 209, 211, 213, 214, 215]:
        punch_mask = X['WorkType'] == punch_code
        if punch_mask.sum() > 10:
            Q1 = y[punch_mask].quantile(0.25)
            Q3 = y[punch_mask].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (y[punch_mask] >= lower_bound) & (y[punch_mask] <= upper_bound)
            mask[punch_mask] = outlier_mask
    
    # AGGRESSIVE outlier removal for 210 & 217
    for punch_code in [210, 217]:
        punch_mask = X['WorkType'] == punch_code
        if punch_mask.sum() > 10:
            # Remove top and bottom 10% of extreme values
            lower_bound = y[punch_mask].quantile(0.10)
            upper_bound = y[punch_mask].quantile(0.90)
            outlier_mask = (y[punch_mask] >= lower_bound) & (y[punch_mask] <= upper_bound)
            mask[punch_mask] = outlier_mask
    
    return X[mask], y[mask]


def detect_and_handle_outliers(df, target_col='Hours', n_std=4):
    """Remove extreme outliers that could hurt model training"""
    for work_type in df['WorkType'].unique():
        wt_mask = df['WorkType'] == work_type
        wt_data = df.loc[wt_mask, target_col]
        
        # Calculate bounds
        mean_val = wt_data.mean()
        std_val = wt_data.std()
        lower_bound = mean_val - n_std * std_val
        upper_bound = mean_val + n_std * std_val
        
        # Cap outliers instead of removing
        df.loc[wt_mask & (df[target_col] < lower_bound), target_col] = lower_bound
        df.loc[wt_mask & (df[target_col] > upper_bound), target_col] = upper_bound
    
    return df

# Add this custom callback class for detailed loss logging
class DetailedLoggingCallback:
    """Custom callback to log training and validation losses at each iteration"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.iterations = []
        
    def __call__(self, env):
        """Called after each iteration"""
        # Get current iteration
        iteration = env.iteration
        
        # Get evaluation results - Fixed to properly extract losses
        if env.evaluation_result_list:
            # LightGBM returns tuples of (dataset_name, metric_name, value, is_higher_better)
            for i, (dataset_name, metric_name, value, _) in enumerate(env.evaluation_result_list):
                # Handle both named and indexed datasets
                if 'training' in str(dataset_name) or i == 0:  # First dataset is usually training
                    if metric_name in ['l2', 'rmse', 'mse']:
                        if iteration >= len(self.train_losses):
                            self.train_losses.append(value)
                elif 'valid' in str(dataset_name) or i == 1:  # Second dataset is usually validation
                    if metric_name in ['l2', 'rmse', 'mse']:
                        if iteration >= len(self.val_losses):
                            self.val_losses.append(value)
            
            self.iterations.append(iteration)
            
            # Log every 10 iterations
            if iteration % 10 == 0 and self.train_losses and self.val_losses:
                train_loss = self.train_losses[-1]
                val_loss = self.val_losses[-1]
                logger.info(f"    Iteration {iteration}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")


# Add this function to calculate comprehensive feature importance
def calculate_feature_importance(model, feature_names, importance_type='gain'):
    """
    Calculate and return feature importance
    
    Parameters:
    -----------
    model : LGBMRegressor
        Trained LightGBM model
    feature_names : list
        List of feature names
    importance_type : str
        Type of importance: 'gain' or 'split'
    """
    try:
        # Get feature importance
        if importance_type == 'gain':
            importances = model.feature_importances_
        else:
            importances = model.feature_importance(importance_type=importance_type)
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        return sorted_importance
        
    except Exception as e:
        logger.error(f"Error calculating feature importance: {str(e)}")
        return {}

# Add this function for RFE with cross-validation
def perform_rfe_cv(X, y, work_type, n_features_to_select=None, cv_splits=3):
    """
    Perform Recursive Feature Elimination with Cross-Validation
    Fixed for Windows compatibility
    """
    logger.info(f"🔍 Starting RFE-CV for {work_type}")
    
    try:
        # Create base estimator with conservative parameters
        base_estimator = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=20,
            random_state=42,
            verbose=-1,
            n_jobs=1  # Use single job to avoid Windows multiprocessing issues
        )
        
        # Initialize RFE with cross-validation
        rfecv = RFECV(
            estimator=base_estimator,
            step=1,
            cv=GapTimeSeriesSplit(n_splits=cv_splits, gap=7),
            scoring='neg_mean_absolute_error',
            n_jobs=1,  # Changed from -1 to avoid Windows issues
            verbose=1,
            min_features_to_select=n_features_to_select or 5
        )
        
        # Rest of the function remains the same...
        logger.info(f"  Fitting RFE-CV with {X.shape[1]} initial features...")
        rfecv.fit(X, y)
        
        # Get results
        optimal_features = X.columns[rfecv.support_].tolist()
        feature_ranking = dict(zip(X.columns, rfecv.ranking_))
        
        logger.info(f"  ✅ Optimal number of features: {rfecv.n_features_}")
        logger.info(f"  📊 Best CV score: {-rfecv.cv_results_['mean_test_score'].max():.4f}")
        logger.info(f"  🎯 Selected features: {optimal_features[:10]}...")
        
        # Create plot with error handling
        try:
            plt.figure(figsize=(10, 6))
            mean_scores = -rfecv.cv_results_['mean_test_score']
            std_scores = rfecv.cv_results_['std_test_score']
            n_features = range(1, len(mean_scores) + 1)
            
            plt.plot(n_features, mean_scores, 'b-', label='CV Score (MAE)')
            plt.fill_between(n_features,
                            mean_scores - std_scores,
                            mean_scores + std_scores,
                            alpha=0.2)
            plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', 
                       label=f'Optimal: {rfecv.n_features_} features')
            plt.xlabel('Number of Features')
            plt.ylabel('Cross-Validation MAE')
            plt.title(f'RFE Cross-Validation Results - {work_type}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join("logs", f"rfe_cv_{work_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"  📈 RFE plot saved to: {plot_path}")
        except Exception as plot_error:
            logger.error(f"Error creating RFE plot: {plot_error}")
        
        return optimal_features, feature_ranking, rfecv.cv_results_
        
    except Exception as e:
        logger.error(f"Error in RFE-CV: {str(e)}")
        return None, None, None
    

# Add this function for outlier detection (if not already present)
def detect_and_handle_outliers(df, column, n_std=3):
    """
    Detect and handle outliers using z-score method
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column to check for outliers
    n_std : float
        Number of standard deviations for outlier threshold
    """
    try:
        df_copy = df.copy()
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(df_copy[column]))
        
        # Find outliers
        outlier_mask = z_scores > n_std
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            logger.info(f"  Found {n_outliers} outliers in {column} (>{n_std} std)")
            
            # Cap outliers at n_std threshold
            mean_val = df_copy[column].mean()
            std_val = df_copy[column].std()
            
            upper_limit = mean_val + (n_std * std_val)
            lower_limit = mean_val - (n_std * std_val)
            
            df_copy.loc[df_copy[column] > upper_limit, column] = upper_limit
            df_copy.loc[df_copy[column] < lower_limit, column] = lower_limit
            
            logger.info(f"  Capped outliers to range [{lower_limit:.2f}, {upper_limit:.2f}]")
        
        return df_copy
        
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        return df

# Add visualization functions
def create_loss_history_plot(loss_histories, work_type):
    """Create and save loss history plots with error handling"""
    try:
        # Filter out empty histories
        valid_histories = [h for h in loss_histories if h.get('train_losses') and h.get('val_losses')]
        
        if not valid_histories:
            logger.warning(f"No valid loss histories to plot for {work_type}")
            return
        
        # Determine subplot layout based on number of valid histories
        n_plots = min(len(valid_histories), 6)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        axes = axes.flatten()
        
        for i, history in enumerate(valid_histories[:n_plots]):
            ax = axes[i]
            
            # Ensure we have matching lengths
            train_losses = history['train_losses']
            val_losses = history['val_losses']
            iterations = history.get('iterations', list(range(len(train_losses))))
            
            # Truncate to minimum length if needed
            min_len = min(len(train_losses), len(val_losses), len(iterations))
            train_losses = train_losses[:min_len]
            val_losses = val_losses[:min_len]
            iterations = iterations[:min_len]
            
            if min_len > 0:
                ax.plot(iterations, train_losses, 'b-', label='Train Loss', alpha=0.7, linewidth=2)
                ax.plot(iterations, val_losses, 'r-', label='Val Loss', alpha=0.7, linewidth=2)
                
                # Add best iteration marker if available
                best_iter = history.get('best_iteration')
                if best_iter and best_iter < len(iterations):
                    ax.axvline(x=best_iter, color='g', linestyle='--', alpha=0.5, label='Best Iteration')
                
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss (RMSE)')
                ax.set_title(f'Fold {history.get("fold", i+1)} Loss History')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Fold {history.get("fold", i+1)} - No Data')
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Training and Validation Loss History - {work_type}', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join("logs", f"loss_history_{work_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Loss history plot saved to: {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating loss history plot: {str(e)}")
        logger.error(f"Loss histories info: {[(h.get('fold'), len(h.get('train_losses', [])), len(h.get('val_losses', []))) for h in loss_histories]}")

def create_feature_importance_plot(feature_importance, work_type, top_n=30):
    """Create and save feature importance plot"""
    try:
        # Get top N features
        top_features = dict(list(feature_importance.items())[:top_n])
        
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, color='steelblue', alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Average Feature Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importances - {work_type}')
        plt.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importances):
            plt.text(v + max(importances) * 0.01, i, f'{v:.2f}', 
                    va='center', fontsize=8)
        
        plt.tight_layout()
        
        plot_path = os.path.join("logs", f"feature_importance_{work_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Feature importance plot saved to: {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")

def main():
    """
    Main function to run enhanced model training
    """
    try:
        logger.info("🚀 Starting Enhanced Model Training for Punch Codes 206 & 213")
        logger.info("=" * 60)
        
        # Load training data
        df = load_training_data()
        if df is None:
            logger.error("❌ Failed to load training data. Exiting.")
            return
        
        # Check data distribution
        logger.info("📊 Data distribution:")
        for work_type in df['WorkType'].unique():
            wt_data = df[df['WorkType'] == work_type]
            logger.info(f"  WorkType {work_type}: {len(wt_data)} records")
            logger.info(f"    Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
            logger.info(f"    Hours avg: {wt_data['Hours'].mean():.2f}")
        
        # Train models for each work type
        models = {}
        metadata = {}
        features = {}
        
        for work_type in ENHANCED_WORK_TYPES:
            logger.info(f"\n🎯 Processing WorkType {work_type}")
            
            work_data = df[df['WorkType'] == work_type].copy()
            work_data = work_data.sort_values('Date')  # Ensure temporal order
            
            if len(work_data) < 50:
                logger.warning(f"Skipping {work_type}: Insufficient data ({len(work_data)} records)")
                continue
            work_data = preprocess_by_punch_code(work_data)

            # Train enhanced model using complete pipeline
            model, model_metadata, selected_features = train_enhanced_model(work_data, work_type)
            
            if model is not None:
                models[work_type] = model
                metadata[work_type] = model_metadata
                features[work_type] = selected_features
                
                logger.info(f"✅ Successfully trained enhanced model for {work_type}")
            else:
                logger.error(f"❌ Failed to train model for {work_type}")
        
        # Save models and metadata
        if models:
            success = save_enhanced_models(models, metadata, features, df)
            
            if success:
                logger.info("\n🎉 ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
                logger.info("=" * 60)
                logger.info(f"✅ Trained models: {list(models.keys())}")
                
                # Print performance summary
                for work_type, meta in metadata.items():
                    logger.info(f"\n📈 {work_type} Performance Summary:")
                    logger.info(f"   MAE: {meta['final_mae']:.3f}")
                    logger.info(f"   R²: {meta['final_r2']:.3f}")
                    logger.info(f"   MAPE: {meta['mape']:.2f}%")
                    logger.info(f"   Pipeline: {' -> '.join(meta['pipeline_steps'])}")
            else:
                logger.error("❌ Failed to save enhanced models")
        else:
            logger.error("❌ No models were successfully trained")
            
    except Exception as e:
        logger.error(f"❌ Error in main training process: {str(e)}")
        logger.error(traceback.format_exc())



if __name__ == "__main__":
    main()