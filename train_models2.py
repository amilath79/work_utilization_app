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
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from lightgbm import early_stopping, log_evaluation

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

# Add these imports at the top of train_models2.py
from lightgbm import early_stopping, log_evaluation

def train_enhanced_model(df, work_type):
    """
    Train enhanced model using COMPLETE PIPELINE approach with LightGBM
    """
    try:
        logger.info(f"Training enhanced LightGBM model for WorkType {work_type} using complete pipeline")
        df = detect_and_handle_outliers(df, 'Hours', n_std=3)  # Changed to 3 for better outlier handling
        y = df['Hours'].values
        
        # Import LightGBM utilities
        from utils.lightgbm_utils import optimize_lightgbm_for_worktype, validate_lightgbm_params
        
        # The pipeline will handle all feature engineering
        basic_features = ['Date', 'WorkType', 'Quantity', 'SystemHours', 'SystemKPI']
        available_basic = [f for f in basic_features if f in df.columns]
        X_basic = df[available_basic].copy()
        
        # Optimize LightGBM parameters for this specific work type
        optimized_params = optimize_lightgbm_for_worktype(X_basic, y, work_type)
        validated_params = validate_lightgbm_params({**DEFAULT_MODEL_PARAMS, **optimized_params})
        
        # Time series cross-validation
        tscv = GapTimeSeriesSplit(n_splits=5, gap=14)  # Reduced splits, increased gap
        fold_scores = []
        train_scores = []  # Track training scores to detect overfitting
        
        logger.info("Performing time series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_basic)):
            # Skip if validation set too small
            if len(val_idx) < 30:
                logger.warning(f"Skipping fold {fold+1}: validation set too small ({len(val_idx)} samples)")
                continue
                
            X_train_fold = X_basic.iloc[train_idx]
            X_val_fold = X_basic.iloc[val_idx] 
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Create a NEW pipeline for each fold (important!)
            fold_pipeline = Pipeline([
                ('feature_engineering', EnhancedFeatureTransformer()),
                ('model', LGBMRegressor(**validated_params))
            ])
            
            # Fit feature engineering on training fold
            feature_eng = fold_pipeline.named_steps['feature_engineering']
            X_train_transformed = feature_eng.fit_transform(X_train_fold)
            X_val_transformed = feature_eng.transform(X_val_fold)
            
            # Train the model with early stopping
            lgb_model = fold_pipeline.named_steps['model']
            lgb_model.fit(
                X_train_transformed, 
                y_train_fold,
                eval_set=[(X_val_transformed, y_val_fold)],
                callbacks=[
                    early_stopping(stopping_rounds=20, verbose=False),
                    log_evaluation(period=0)  # Suppress output
                ]
            )
            
            # Predict on validation using the fitted model
            y_pred_val = lgb_model.predict(X_val_transformed)
            
            # Also predict on training to check overfitting
            y_pred_train = lgb_model.predict(X_train_transformed)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train_fold, y_pred_train)
            val_mae = mean_absolute_error(y_val_fold, y_pred_val)
            val_r2 = r2_score(y_val_fold, y_pred_val)
            
            # Calculate overfitting ratio
            overfit_ratio = train_mae / val_mae if val_mae > 0 else 0
            
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
        logger.info("Training final pipeline on all data...")
        complete_pipeline = Pipeline([
            ('feature_engineering', EnhancedFeatureTransformer()),
            ('model', LGBMRegressor(**validated_params))
        ])
        
        # For final training, use validation split for early stopping
        val_size = int(len(X_basic) * 0.2)
        X_train_final = X_basic.iloc[:-val_size]
        y_train_final = y[:-val_size]
        X_val_final = X_basic.iloc[-val_size:]
        y_val_final = y[-val_size:]
        
        # Fit feature engineering on training data
        fe = complete_pipeline.named_steps['feature_engineering']
        X_train_transformed_final = fe.fit_transform(X_train_final)
        X_val_transformed_final = fe.transform(X_val_final)
        
        # Train model with early stopping
        final_model = complete_pipeline.named_steps['model']
        final_model.fit(
            X_train_transformed_final,
            y_train_final,
            eval_set=[(X_val_transformed_final, y_val_final)],
            callbacks=[
                early_stopping(stopping_rounds=20, verbose=True),
                log_evaluation(period=10)
            ]
        )
        
        # Final evaluation on ALL data
        y_pred_final = complete_pipeline.predict(X_basic)
        final_mae = mean_absolute_error(y, y_pred_final)
        final_r2 = r2_score(y, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))
        
        # Calculate MAPE
        mape = np.mean(np.abs((y - y_pred_final) / np.where(y == 0, 1, y))) * 100
        
        # Calculate average CV metrics
        avg_cv_mae = np.mean([score['MAE'] for score in fold_scores]) if fold_scores else final_mae
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores]) if fold_scores else final_r2
        
        # Create metadata
        model_metadata = {
            'work_type': work_type,
            'training_records': len(df),
            'final_mae': final_mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'mape': mape,
            'cv_mae': avg_cv_mae,
            'cv_r2': avg_cv_r2,
            'cv_folds': len(fold_scores),
            'input_features': basic_features,
            'pipeline_steps': [step[0] for step in complete_pipeline.steps],
            'model_type': 'complete_pipeline',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
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
        
        return complete_pipeline, model_metadata, basic_features
        
    except Exception as e:
        logger.error(f"Error training enhanced model for {work_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None


def save_enhanced_models(models, metadata, features, df):
    """Save enhanced models and metadata"""
    try:
        logger.info("Saving enhanced models and metadata")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for work_type, model in models.items():
            if model is not None:
                model_filename = f"enhanced_model_{work_type}.pkl"
                model_path = os.path.join(MODELS_DIR, model_filename)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"  ✅ Saved model for {work_type}: {model_filename}")
        
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
        
        # Save training data for predictions
        try:
            training_data_path = os.path.join(MODELS_DIR, 'enhanced_training_data.pkl')
            df.to_pickle(training_data_path)
            logger.info(f"✅ Enhanced training data saved: {training_data_path}")
        except Exception as e:
            logger.error(f"⚠️ Failed to save training data: {str(e)}")

        logger.info(f"✅ All enhanced models and metadata saved")
        
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