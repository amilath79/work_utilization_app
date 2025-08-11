import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import traceback
import argparse  # Added for command-line argument parsing
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import json
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

from utils.feature_engineering import EnhancedFeatureTransformer
from utils.sql_data_connector import extract_sql_data
from config import ENHANCED_WORK_TYPES, MODELS_DIR, DEFAULT_MODEL_PARAMS, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, MAX_FEATURES_PER_MODEL
from utils.feature_selection import FeatureSelector

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

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def load_training_data():
    try:
        logger.info(f"Loading training data for enhanced models {ENHANCED_WORK_TYPES}")
        query = """
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
        CASE WHEN PunchCode IN (206, 213) THEN NoRows
        ELSE Quantity END as Quantity
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
        AND Hours > 0 
        AND SystemHours > 0 
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
            return None, None
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        logger.info(f"Loaded {len(df)} records for enhanced training")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def detect_and_handle_outliers(df, target_col='Hours', work_type=None):
    # Punch code 214 specific handling - preserve high values
    if work_type == 214:
        wt_mask = df['WorkType'] == work_type
        wt_data = df.loc[wt_mask, target_col]
        
        # Use IQR method instead of std for punch code 214
        Q1 = wt_data.quantile(0.25)
        Q3 = wt_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # More conservative bounds for high-variance punch code
        lower_bound = Q1 - 2.0 * IQR  # Less aggressive than 1.5*IQR
        upper_bound = Q3 + 3.0 * IQR  # More aggressive to preserve peaks
        
        # Only clip extreme outliers
        df.loc[wt_mask & (df[target_col] < lower_bound), target_col] = lower_bound
        df.loc[wt_mask & (df[target_col] > upper_bound), target_col] = upper_bound
        
        logger.info(f"Punch code 214: IQR outlier bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
    else:
        # Standard handling for other punch codes
        wt_mask = df['WorkType'] == work_type if work_type else True
        wt_data = df.loc[wt_mask, target_col]
        mean_val = wt_data.mean()
        std_val = wt_data.std()
        lower_bound = mean_val - 4 * std_val
        upper_bound = mean_val + 4 * std_val
        df.loc[wt_mask & (df[target_col] < lower_bound), target_col] = lower_bound
        df.loc[wt_mask & (df[target_col] > upper_bound), target_col] = upper_bound
    
    return df

def apply_target_transformation(df, work_type):
    """Apply optimal transformation based on punch code characteristics"""
    if work_type == 214:
        # Box-Cox-like transformation for high variance data
        df['transformed_Hours'] = np.sign(df['Hours']) * np.log1p(np.abs(df['Hours']))
        logger.info(f"Applied sign-log transformation for punch code {work_type}")
    else:
        # Standard log transformation for other punch codes
        df['transformed_Hours'] = np.log1p(df['Hours'])
        logger.info(f"Applied log1p transformation for punch code {work_type}")
    
    return df['transformed_Hours'].values

def train_enhanced_model(df, work_type):
    try:
        logger.info(f"Training enhanced LightGBM model for WorkType {work_type} using complete pipeline")
        df = detect_and_handle_outliers(df, 'Hours', work_type) 
        y = apply_target_transformation(df, work_type)

        basic_features = ['Date', 'WorkType', 'Quantity', 'SystemHours']
        available_basic = [f for f in basic_features if f in df.columns]
        X_basic = df[available_basic].copy()

        # CRITICAL FIX: Reserve final 20% for true out-of-sample testing
        test_size = int(len(X_basic) * 0.2)
        
        # Split data temporally - most recent data for testing
        X_train_cv = X_basic.iloc[:-test_size].copy()
        y_train_cv = y[:-test_size]
        X_test_final = X_basic.iloc[-test_size:].copy()
        y_test_final = y[-test_size:]
        
        logger.info(f"Data split - Train/CV: {len(X_train_cv)} samples, Final Test: {len(X_test_final)} samples")
        logger.info(f"Test period: {X_test_final['Date'].min()} to {X_test_final['Date'].max()}")

        # TimeSeriesSplit for CV on training data only
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []
        feature_importances = None

        logger.info("Performing time series cross-validation on training data...")

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_cv)):
            X_train_fold = X_train_cv.iloc[train_idx]
            X_val_fold = X_train_cv.iloc[val_idx]
            y_train_fold = y_train_cv[train_idx]
            y_val_fold = y_train_cv[val_idx]

            # Create pipeline for this fold
            fold_pipeline = Pipeline([
                ('feature_engineering', EnhancedFeatureTransformer()),
                ('model', LGBMRegressor(**DEFAULT_MODEL_PARAMS))
            ])

            # Fit and transform
            feature_eng = fold_pipeline.named_steps['feature_engineering']
            X_train_transformed = feature_eng.fit_transform(X_train_fold)
            X_val_transformed = feature_eng.transform(X_val_fold)

            # Train model
            lgb_model = fold_pipeline.named_steps['model']
            lgb_model.fit(
                X_train_transformed,
                y_train_fold,
                eval_set=[(X_val_transformed, y_val_fold)],
                callbacks=[]
            )

            # Evaluate on validation set
            y_pred_val_log = lgb_model.predict(X_val_transformed)
            y_pred_val = np.expm1(y_pred_val_log)
            y_val_true = np.expm1(y_val_fold)

            val_mae = mean_absolute_error(y_val_true, y_pred_val)
            val_r2 = r2_score(y_val_true, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val_true, y_pred_val))
            val_mape = np.mean(np.abs((y_val_true - y_pred_val) / np.maximum(y_val_true, 10))) * 100

            fold_scores.append({'MAE': val_mae, 'RMSE': val_rmse, 'R2': val_r2, 'MAPE': val_mape})
            
            # Accumulate feature importances
            current_importances = lgb_model.feature_importances_
            if feature_importances is None:
                feature_importances = current_importances
            else:
                feature_importances += current_importances

            # # Plot for fold (optional - can be commented out for speed)
            # plt.figure(figsize=(10,4))
            # plt.plot(y_val_true, label='Actual')
            # plt.plot(y_pred_val, label='Predicted')
            # plt.legend()
            # plt.title(f"CV Fold {fold+1} - WorkType {work_type}")
            # plt.tight_layout()
            # plt.savefig(os.path.join(MODELS_DIR, f"cv_fold_{work_type}_fold{fold+1}.png"))
            # plt.close()

        # Feature selection based on average importance
        feature_eng = EnhancedFeatureTransformer()
        X_train_cv_transformed = feature_eng.fit_transform(X_train_cv)
        feature_names = X_train_cv_transformed.columns if hasattr(X_train_cv_transformed, 'columns') else [f'f{i}' for i in range(X_train_cv_transformed.shape[1])]
        
        # Average importances across folds
        feature_importances = feature_importances / len(fold_scores)
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = importance_df.head(MAX_FEATURES_PER_MODEL)['feature'].tolist()
        
        # Add diagnostic logging
        logger.info(f"Trend features created: {len([f for f in feature_names if 'trend' in f.lower()])} - {[f for f in feature_names if 'trend' in f.lower()][:5]}")
        
        # Force include some trend features if they exist
        trend_features = [col for col in feature_names if 'trend' in col.lower()]
        if trend_features and len([f for f in selected_features if 'trend' in f.lower()]) < 5:
            selected_features = list(set(selected_features + trend_features[:5]))
            logger.info(f"Added trend features: {trend_features[:5]}")
        
        logger.info(f"Selected top {len(selected_features)} features for final model: {selected_features}")

        # CRITICAL: Train final model on ALL training data (not including test set)
        complete_pipeline = Pipeline([
            ('feature_engineering', EnhancedFeatureTransformer()),
            ('feature_selection', FeatureSelector(selected_features)),
            ('model', LGBMRegressor(**DEFAULT_MODEL_PARAMS))
        ])

        # Fit on all training data
        fe = complete_pipeline.named_steps['feature_engineering']
        fe.fit(X_train_cv)
        X_train_transformed = fe.transform(X_train_cv)
        
        fs = complete_pipeline.named_steps['feature_selection']
        fs.fit(X_train_transformed)
        X_train_selected = fs.transform(X_train_transformed)
        
        # Transform test set
        X_test_transformed = fe.transform(X_test_final)
        X_test_selected = fs.transform(X_test_transformed)
        
        # Train final model
        final_model = complete_pipeline.named_steps['model']
        final_model.fit(
            X_train_selected,
            y_train_cv,
            eval_set=[(X_test_selected, y_test_final)],
            callbacks=[]
        )

        # CRITICAL: Evaluate ONLY on held-out test set
        y_pred_test_log = final_model.predict(X_test_selected)
        y_pred_test = np.expm1(y_pred_test_log)
        y_true_test = np.expm1(y_test_final)
        
        # Calculate test metrics
        test_mae = mean_absolute_error(y_true_test, y_pred_test)
        test_r2 = r2_score(y_true_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
        test_mape = np.mean(np.abs((y_true_test - y_pred_test) / np.maximum(y_true_test, 10))) * 100

        # Calculate average CV metrics
        avg_cv_mae = np.mean([score['MAE'] for score in fold_scores]) if fold_scores else test_mae
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores]) if fold_scores else test_r2
        avg_cv_mape = np.mean([score['MAPE'] for score in fold_scores]) if fold_scores else test_mape

        # Plot test set predictions
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(y_true_test, label='Actual')
        plt.plot(y_pred_test, label='Predicted')
        plt.legend()
        plt.title(f"Test Set Predictions - WorkType {work_type}")
        
        plt.subplot(1,2,2)
        plt.scatter(y_true_test, y_pred_test, alpha=0.5)
        plt.plot([y_true_test.min(), y_true_test.max()], [y_true_test.min(), y_true_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f"Test Set Scatter - R²={test_r2:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f"test_predictions_{work_type}.png"))
        plt.close()

        # Create comprehensive metadata
        model_metadata = {
            'work_type': work_type,
            'training_records': len(X_train_cv),
            'test_records': len(X_test_final),
            'test_period': {
                'start': str(X_test_final['Date'].min()),
                'end': str(X_test_final['Date'].max())
            },
            # Test set metrics (true out-of-sample)
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            # Cross-validation metrics (average)
            'cv_mae': avg_cv_mae,
            'cv_r2': avg_cv_r2,
            'cv_mape': avg_cv_mape,
            'cv_folds': len(fold_scores),
            # Feature information
            'input_features': basic_features,
            'selected_features': selected_features,
            'num_features': len(selected_features),
            'trend_features_count': len([f for f in selected_features if 'trend' in f.lower()]),
            # Model configuration
            'pipeline_steps': [step[0] for step in complete_pipeline.steps],
            'model_type': 'complete_pipeline',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }

        logger.info(f"✅ Enhanced LightGBM pipeline trained for {work_type}")
        logger.info(f"   Test MAE: {test_mae:.3f} (on {len(X_test_final)} samples)")
        logger.info(f"   Test R²: {test_r2:.3f}")
        logger.info(f"   Test MAPE: {test_mape:.2f}%")
        logger.info(f"   CV MAE: {avg_cv_mae:.3f} (avg of {len(fold_scores)} folds)")
        logger.info(f"   CV R²: {avg_cv_r2:.3f}")

        return complete_pipeline, model_metadata, selected_features

    except Exception as e:
        logger.error(f"Error training enhanced model for {work_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def save_enhanced_models(models, metadata, features, df):
    try:
        logger.info("Saving enhanced models and metadata")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for work_type, model in models.items():
            if model is not None:
                model_filename = f"enhanced_model_{work_type}.pkl"
                model_path = os.path.join(MODELS_DIR, model_filename)
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"  ✅ Saved model for {work_type}: {model_filename}")
        metadata_filename = f"enhanced_models_metadata_{timestamp}.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        features_filename = f"enhanced_features_{timestamp}.json"
        features_path = os.path.join(MODELS_DIR, features_filename)
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2)
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
    
# AFTER save_enhanced_models function
def create_model_summary(metadata):
    """Create a comprehensive model summary after training"""
    try:
        summary_path = os.path.join(MODELS_DIR, 'model_summary.csv')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare summary data
        summary_data = []
        for work_type, meta in metadata.items():
            summary_data.append({
                'Timestamp': timestamp,
                'Work_Type': work_type,
                'MAE': meta['test_mae'],
                'R2': meta['test_r2'], 
                'MAPE': meta['test_mape'],
                'CV_MAE': meta['cv_mae'],
                'CV_R2': meta['cv_r2'],
                'Training_Records': meta['training_records'],
                'Test_Records': meta['test_records'],
                'Features_Count': meta['num_features']
            })
        
        # Create DataFrame
        new_summary = pd.DataFrame(summary_data)
        
        # Load existing summary if it exists, otherwise create new
        if os.path.exists(summary_path):
            existing_summary = pd.read_csv(summary_path)
            # Combine with new results
            combined_summary = pd.concat([existing_summary, new_summary], ignore_index=True)
        else:
            combined_summary = new_summary
        
        # Save updated summary
        combined_summary.to_csv(summary_path, index=False)
        
        # Log summary for immediate view
        logger.info("\n" + "="*60)
        logger.info("📊 MODEL PERFORMANCE SUMMARY")
        logger.info("="*60)
        for _, row in new_summary.iterrows():
            logger.info(f"Work Type {row['Work_Type']}:")
            logger.info(f"  MAE: {row['MAE']:.3f} | R²: {row['R2']:.3f} | MAPE: {row['MAPE']:.2f}%")
            logger.info(f"  Records: {row['Training_Records']} train, {row['Test_Records']} test")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating model summary: {str(e)}")
        return False

def main():
    # Add argument parsing for specific model training
    parser = argparse.ArgumentParser(description='Train workforce prediction models')
    parser.add_argument('--punch-code', '-p', type=str, help='Train specific punch code (e.g., 211)')
    parser.add_argument('--all', action='store_true', help='Train all punch codes (default behavior)')
    args = parser.parse_args()
    
    try:
        df = load_training_data()
        if df is None:
            logger.error("❌ Failed to load training data. Exiting.")
            return

        # Determine which work types to process
        if args.punch_code:
            if args.punch_code not in ENHANCED_WORK_TYPES:
                logger.error(f"❌ Punch code {args.punch_code} not in enhanced work types: {ENHANCED_WORK_TYPES}")
                return
            if args.punch_code not in df['WorkType'].unique():
                logger.error(f"❌ No data available for punch code {args.punch_code}")
                logger.info(f"Available work types in data: {list(df['WorkType'].unique())}")
                return
            work_types_to_process = [args.punch_code]
            logger.info(f"🎯 Training single model for punch code: {args.punch_code}")
        else:
            work_types_to_process = df['WorkType'].unique()
            logger.info(f"🎯 Training all available models: {list(work_types_to_process)}")

        logger.info("📊 Data distribution:")
        for work_type in work_types_to_process:
            if work_type in df['WorkType'].unique():
                wt_data = df[df['WorkType'] == work_type]
                logger.info(f"  WorkType {work_type}: {len(wt_data)} records")
                logger.info(f"    Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
                logger.info(f"    Hours avg: {wt_data['Hours'].mean():.2f}")

        models = {}
        metadata = {}
        features = {}

        for work_type in work_types_to_process:
            if work_type not in df['WorkType'].unique():
                logger.warning(f"⚠️ No data available for punch code {work_type}")
                continue
                
            logger.info(f"\n🎯 Processing WorkType {work_type}")
            work_data = df[df['WorkType'] == work_type].copy()
            work_data = work_data.sort_values('Date')
            
            if len(work_data) < 50:
                logger.warning(f"Skipping {work_type}: Insufficient data ({len(work_data)} records)")
                continue
                
            model, model_metadata, selected_features = train_enhanced_model(work_data, work_type)
            if model is not None:
                models[work_type] = model
                metadata[work_type] = model_metadata
                features[work_type] = selected_features
                logger.info(f"✅ Successfully trained enhanced model for {work_type}")
            else:
                logger.error(f"❌ Failed to train model for {work_type}")

        if models:
            success = save_enhanced_models(models, metadata, features, df)
            if success:
                create_model_summary(metadata)
                logger.info("\n🎉 ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
                logger.info("=" * 60)
                logger.info(f"✅ Trained models: {list(models.keys())}")
                for work_type, meta in metadata.items():
                    logger.info(f"\n📈 {work_type} Performance Summary:")
                    logger.info(f"   Test MAE: {meta['test_mae']:.3f} (True out-of-sample)")
                    logger.info(f"   Test R²: {meta['test_r2']:.3f}")
                    logger.info(f"   Test MAPE: {meta['test_mape']:.2f}%")
                    logger.info(f"   CV MAE: {meta['cv_mae']:.3f} (Cross-validation average)")
                    logger.info(f"   CV R²: {meta['cv_r2']:.3f}")
                    logger.info(f"   Features: {meta['num_features']} (including {meta['trend_features_count']} trend features)")
            else:
                logger.error("❌ Failed to save enhanced models")
        else:
            logger.error("❌ No models were successfully trained")

    except Exception as e:
        logger.error(f"❌ Error in main training process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()