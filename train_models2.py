import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import traceback
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import json
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

from utils.feature_engineering import EnhancedFeatureTransformer
from utils.sql_data_connector import extract_sql_data
from config import ENHANCED_WORK_TYPES, MODELS_DIR, DEFAULT_MODEL_PARAMS, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION
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

def detect_and_handle_outliers(df, target_col='Hours', n_std=4):
    for work_type in df['WorkType'].unique():
        wt_mask = df['WorkType'] == work_type
        wt_data = df.loc[wt_mask, target_col]
        mean_val = wt_data.mean()
        std_val = wt_data.std()
        lower_bound = mean_val - n_std * std_val
        upper_bound = mean_val + n_std * std_val
        df.loc[wt_mask & (df[target_col] < lower_bound), target_col] = lower_bound
        df.loc[wt_mask & (df[target_col] > upper_bound), target_col] = upper_bound
    return df

def train_enhanced_model(df, work_type):
    try:
        logger.info(f"Training enhanced LightGBM model for WorkType {work_type} using complete pipeline")
        df = detect_and_handle_outliers(df, 'Hours', n_std=4)
        df['log_Hours'] = np.log1p(df['Hours'])
        y = df['log_Hours'].values

        basic_features = ['Date', 'WorkType', 'Quantity', 'SystemHours', 'SystemKPI']
        available_basic = [f for f in basic_features if f in df.columns]
        X_basic = df[available_basic].copy()

        # TimeSeriesSplit for robust validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = []
        feature_importances = None

        logger.info("Performing time series cross-validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_basic)):
            X_train_fold = X_basic.iloc[train_idx]
            X_val_fold = X_basic.iloc[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            fold_pipeline = Pipeline([
                ('feature_engineering', EnhancedFeatureTransformer()),
                ('model', LGBMRegressor(**DEFAULT_MODEL_PARAMS))
            ])

            feature_eng = fold_pipeline.named_steps['feature_engineering']
            X_train_transformed = feature_eng.fit_transform(X_train_fold)
            X_val_transformed = feature_eng.transform(X_val_fold)

            lgb_model = fold_pipeline.named_steps['model']
            lgb_model.fit(
                X_train_transformed,
                y_train_fold,
                eval_set=[(X_val_transformed, y_val_fold)],
                callbacks=[]
            )

            y_pred_val_log = lgb_model.predict(X_val_transformed)
            y_pred_val = np.expm1(y_pred_val_log)
            y_val_true = np.expm1(y_val_fold)

            val_mae = mean_absolute_error(y_val_true, y_pred_val)
            val_r2 = r2_score(y_val_true, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val_true, y_pred_val))
            val_mape = np.mean(np.abs((y_val_true - y_pred_val) / np.maximum(y_val_true, 10))) * 100

            fold_scores.append({'MAE': val_mae, 'RMSE': val_rmse, 'R2': val_r2, 'MAPE': val_mape})

            # Feature importances
            current_importances = lgb_model.feature_importances_
            if feature_importances is None:
                feature_importances = current_importances
            else:
                feature_importances += current_importances

            plt.figure(figsize=(10,4))
            plt.plot(y_val_true, label='Actual')
            plt.plot(y_pred_val, label='Predicted')
            plt.legend()
            plt.title(f"Actual vs Predicted (Validation Fold {fold+1}) - WorkType {work_type}")
            plt.tight_layout()
            plt.savefig(os.path.join(MODELS_DIR, f"val_plot_{work_type}_fold{fold+1}.png"))
            plt.close()

        # Feature selection
        feature_eng = EnhancedFeatureTransformer()
        X_all_transformed = feature_eng.fit_transform(X_basic)
        feature_names = X_all_transformed.columns if hasattr(X_all_transformed, 'columns') else [f'f{i}' for i in range(X_all_transformed.shape[1])]
        feature_importances = feature_importances / len(fold_scores)
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        selected_features = importance_df.head(20)['feature'].tolist()
        logger.info(f"Selected top 20 features for final model: {selected_features}")

        # Final pipeline
        complete_pipeline = Pipeline([
            ('feature_engineering', EnhancedFeatureTransformer()),
            ('feature_selection', FeatureSelector(selected_features)),
            ('model', LGBMRegressor(**DEFAULT_MODEL_PARAMS))
        ])

        val_size = int(len(X_basic) * 0.2)
        X_train_final = X_basic.iloc[:-val_size]
        y_train_final = y[:-val_size]
        X_val_final = X_basic.iloc[-val_size:]
        y_val_final = y[-val_size:]

        fe = complete_pipeline.named_steps['feature_engineering']
        fe.fit(X_train_final)
        X_train_transformed_final = fe.transform(X_train_final)
        X_val_transformed_final = fe.transform(X_val_final)

        fs = complete_pipeline.named_steps['feature_selection']
        fs.fit(X_train_transformed_final)
        X_train_selected = fs.transform(X_train_transformed_final)
        X_val_selected = fs.transform(X_val_transformed_final)

        final_model = complete_pipeline.named_steps['model']
        final_model.fit(
            X_train_selected,
            y_train_final,
            eval_set=[(X_val_selected, y_val_final)],
            callbacks=[]
        )

        # Final evaluation on ALL data (original scale)
        X_all_transformed = fe.transform(X_basic)
        X_all_selected = fs.transform(X_all_transformed)
        y_pred_final_log = final_model.predict(X_all_selected)
        y_pred_final = np.expm1(y_pred_final_log)
        y_true_final = np.expm1(y)
        final_mae = mean_absolute_error(y_true_final, y_pred_final)
        final_r2 = r2_score(y_true_final, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
        mape = np.mean(np.abs((y_true_final - y_pred_final) / np.maximum(y_true_final, 10))) * 100

        avg_cv_mae = np.mean([score['MAE'] for score in fold_scores]) if fold_scores else final_mae
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores]) if fold_scores else final_r2

        plt.figure(figsize=(12,4))
        plt.plot(y_true_final, label='Actual')
        plt.plot(y_pred_final, label='Predicted')
        plt.legend()
        plt.title(f"Actual vs Predicted (All Data) - WorkType {work_type}")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f"final_plot_{work_type}.png"))
        plt.close()

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
            'selected_features': selected_features,
            'pipeline_steps': [step[0] for step in complete_pipeline.steps],
            'model_type': 'complete_pipeline',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }

        logger.info(f"✅ Enhanced LightGBM pipeline trained for {work_type}")
        logger.info(f"   Final MAE: {final_mae:.3f}")
        logger.info(f"   Final R²: {final_r2:.3f}")
        logger.info(f"   CV MAE: {avg_cv_mae:.3f}")
        logger.info(f"   CV R²: {avg_cv_r2:.3f}")
        logger.info(f"   MAPE: {mape:.2f}%")

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

def main():
    try:
        df = load_training_data()
        if df is None:
            logger.error("❌ Failed to load training data. Exiting.")
            return

        logger.info("📊 Data distribution:")
        for work_type in df['WorkType'].unique():
            wt_data = df[df['WorkType'] == work_type]
            logger.info(f"  WorkType {work_type}: {len(wt_data)} records")
            logger.info(f"    Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
            logger.info(f"    Hours avg: {wt_data['Hours'].mean():.2f}")

        models = {}
        metadata = {}
        features = {}

        for work_type in df['WorkType'].unique():
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
                logger.info("\n🎉 ENHANCED MODEL TRAINING COMPLETED SUCCESSFULLY")
                logger.info("=" * 60)
                logger.info(f"✅ Trained models: {list(models.keys())}")
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