"""
Enhanced Model Training for Punch Codes 206 & 213
Enterprise-Grade Time Series Model Training with Advanced Feature Engineering
Uses Complete Pipeline Approach Only
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import json



# Import utilities - PIPELINE APPROACH ONLY
from utils.feature_engineering import EnhancedFeatureTransformer
from utils.holiday_utils import is_non_working_day
from utils.sql_data_connector import extract_sql_data

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
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, NoRows as Quantity, SystemKPI 
        FROM WorkUtilizationData 
        WHERE PunchCode IN (206, 213) 
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

def train_enhanced_model(df, work_type):
    """
    Train enhanced model using COMPLETE PIPELINE approach
    """
    try:
        logger.info(f"Training enhanced model for WorkType {work_type} using complete pipeline")
        
        # Prepare data - TARGET IS NOW HOURS
        y = df['Hours'].values
        
        
        # The pipeline will handle all feature engineering
        basic_features = ['Date', 'WorkType', 'Quantity', 'SystemHours', 'SystemKPI']
        available_basic_features = [f for f in basic_features if f in df.columns]
        
        X_basic = df[available_basic_features].copy()
        
        logger.info(f"Training with {len(X_basic)} records and {len(available_basic_features)} basic input features")

        # Log active feature groups (same as your create_enhanced_features approach)
        enabled_groups = [k for k, v in FEATURE_GROUPS.items() if v]
        logger.info(f"📊 Config-driven training - Active Feature Groups: {enabled_groups}")
        logger.info(f"📊 Using ESSENTIAL_LAGS: {ESSENTIAL_LAGS}")
        logger.info(f"📊 Using ESSENTIAL_WINDOWS: {ESSENTIAL_WINDOWS}")
        
        # Create COMPLETE pipeline (config-driven)
        complete_pipeline = Pipeline([
            # Step 1: Config-driven Feature Engineering
            ('feature_engineering', EnhancedFeatureTransformer()),  # No parameters - reads from config
            
            # Step 2: Preprocessing (scaling, encoding)
            ('preprocessing', StandardScaler()),
            
            # Step 3: Model
            ('model', RandomForestRegressor(**DEFAULT_MODEL_PARAMS))
        ])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []
        
        logger.info("Performing time series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_basic)):
            X_train_fold = X_basic.iloc[train_idx]
            X_val_fold = X_basic.iloc[val_idx] 
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Train pipeline on fold
            complete_pipeline.fit(X_train_fold, y_train_fold)
            
            # Predict on validation
            y_pred_fold = complete_pipeline.predict(X_val_fold)
            
            # Calculate metrics
            fold_mae = mean_absolute_error(y_val_fold, y_pred_fold)
            fold_r2 = r2_score(y_val_fold, y_pred_fold)
            fold_scores.append({'MAE': fold_mae, 'R2': fold_r2})
            
            logger.info(f"  Fold {fold+1}: MAE={fold_mae:.3f}, R²={fold_r2:.3f}")
        
        # Train final pipeline on all data
        logger.info("Training final pipeline on all data...")
        complete_pipeline.fit(X_basic, y)
        
        # Final evaluation
        y_pred_final = complete_pipeline.predict(X_basic)
        final_mae = mean_absolute_error(y, y_pred_final)
        final_r2 = r2_score(y, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))
        
        # Calculate MAPE
        mape = np.mean(np.abs((y - y_pred_final) / np.where(y == 0, 1, y))) * 100
        
        # Calculate average CV metrics
        avg_cv_mae = np.mean([score['MAE'] for score in fold_scores])
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores])
        
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
            'input_features': available_basic_features,
            'pipeline_steps': [step[0] for step in complete_pipeline.steps],
            'model_type': 'complete_pipeline',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        logger.info(f"✅ Enhanced complete pipeline trained for {work_type}")
        logger.info(f"   Final MAE: {final_mae:.3f}")
        logger.info(f"   Final R²: {final_r2:.3f}")
        logger.info(f"   CV MAE: {avg_cv_mae:.3f}")
        logger.info(f"   MAPE: {mape:.2f}%")
        
        return complete_pipeline, model_metadata, available_basic_features
        
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
        
        for work_type in ['206', '213']:
            logger.info(f"\n🎯 Processing WorkType {work_type}")
            
            work_data = df[df['WorkType'] == work_type].copy()
            work_data = work_data.sort_values('Date')  # Ensure temporal order
            
            if len(work_data) < 50:
                logger.warning(f"Skipping {work_type}: Insufficient data ({len(work_data)} records)")
                continue
            
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