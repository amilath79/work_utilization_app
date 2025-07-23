"""
Train LightGBM model for workforce prediction using database data.
Focus: Next-day prediction accuracy with minimal complexity.
"""
import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold

from utils.sql_data_connector import extract_sql_data
from config import (
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION,
    DEFAULT_MODEL_PARAMS, MODELS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkforcePredictor:
    """Simplified LightGBM predictor for next-day workforce prediction"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.data = None
        
    def load_data(self):
        """Load data from database using specified query"""
        query = """
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
        CASE WHEN PunchCode IN (206, 213) THEN NoRows
        ELSE Quantity END as Quantity
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
        AND Hours > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        AND Date < '2025-05-06'
        ORDER BY Date
        """
        
        logger.info("Loading data from database...")
        self.data = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if self.data is None or self.data.empty:
            raise ValueError("No data returned from database")
        
        # Convert data types
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['WorkType'] = self.data['WorkType'].astype(str)
        
        logger.info(f"Loaded {len(self.data)} records")
        logger.info(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        logger.info(f"Work types: {sorted(self.data['WorkType'].unique())}")
        
    def create_features(self, df):
        """Create features for prediction - simplified and focused"""
        df = df.copy()
        
        # Time features
        df['year'] = df['Date'].dt.year
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['dayofmonth'] = df['Date'].dt.day
        df['weekofyear'] = df['Date'].dt.isocalendar().week
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Simple flags
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = (df['dayofmonth'] <= 5).astype(int)
        df['is_month_end'] = (df['dayofmonth'] >= 25).astype(int)
        
        # Lag features for Hours (target)
        for lag in [1, 7, 14, 21, 28]:
            df[f'hours_lag_{lag}'] = df['Hours'].shift(lag)
            df[f'quantity_lag_{lag}'] = df['Quantity'].shift(lag)
            df[f'systemhours_lag_{lag}'] = df['SystemHours'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 28]:
                df[f'hours_roll_mean_{window}'] = df['Hours'].shift(1).rolling(window, min_periods=1).mean()
                df[f'hours_roll_std_{window}'] = df['Hours'].shift(1).rolling(window, min_periods=1).std().fillna(0)
                df[f'quantity_roll_mean_{window}'] = df['Quantity'].shift(1).rolling(window, min_periods=1).mean()
                df[f'systemhours_roll_mean_{window}'] = df['SystemHours'].shift(1).rolling(window, min_periods=1).mean()
        
        # Productivity features
        df['quantity_per_hour'] = df['Quantity'] / (df['SystemHours'] + 1)
        df['hours_system_ratio'] = df['Hours'] / (df['SystemHours'] + 1)
    
        
        return df
    
    def prepare_training_data(self, df):
        exclude_cols = ['Date', 'WorkType', 'Hours']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        X = df[self.feature_columns]
        y = df['Hours']
        # Remove near-constant features
        selector = VarianceThreshold(threshold=1e-5)
        X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
        self.feature_columns = list(X.columns)
        return X, y
    
    def train_model_for_worktype(self, work_type):
        """Train LightGBM model for a specific work type"""
        # Filter data
        wt_data = self.data[self.data['WorkType'] == work_type].copy()
        
        if len(wt_data) < 50:
            logger.warning(f"Insufficient data for WorkType {work_type}: {len(wt_data)} records")
            return None
        
        # Create features
        wt_data = self.create_features(wt_data)
        
        # Remove rows with too many NaN values (keep at least lag_28)
        wt_data = wt_data.dropna(subset=['hours_lag_28'])
        
        if len(wt_data) < 30:
            logger.warning(f"Insufficient data after feature creation for WorkType {work_type}")
            return None
        
        # Prepare training data
        X, y = self.prepare_training_data(wt_data)
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Time series cross-validation
        n_splits = min(5, len(X) // 50)
        if n_splits < 2:
            # Simple train-test split for small datasets
            split_idx = int(0.8 * len(X))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model = self.tune_lightgbm(X_train, y_train)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            logger.info(f"WorkType {work_type} - Validation MAE: {mae:.3f}, R²: {r2:.3f}")
            
        else:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = self.tune_lightgbm(X_train, y_train)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
            
            avg_mae = np.mean(cv_scores)
            logger.info(f"WorkType {work_type} - CV Average MAE: {avg_mae:.3f}")
        
        # Train final model on all data
        final_model = self.tune_lightgbm(X, y)
        final_model.fit(X, y)
        
        # Store model information
        model_info = {
            'model': final_model,
            'feature_columns': self.feature_columns,
            'last_date': wt_data['Date'].max(),
            'last_data': wt_data.tail(30),
            'mae': mae if 'mae' in locals() else avg_mae,
            'n_samples': len(wt_data)
        }
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 5 features for WorkType {work_type}:")
        logger.info(importance_df.head())
        
        return model_info
    
    def train_all_models(self):
        """Train models for all work types"""
        work_types = sorted(self.data['WorkType'].unique())
        
        for work_type in work_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training model for WorkType: {work_type}")
            logger.info(f"{'='*50}")
            
            model_info = self.train_model_for_worktype(work_type)
            
            if model_info is not None:
                self.models[work_type] = model_info
                logger.info(f"✓ Successfully trained model for WorkType {work_type}")
            else:
                logger.warning(f"✗ Failed to train model for WorkType {work_type}")
        
        logger.info(f"\nTraining complete. Models trained: {len(self.models)}/{len(work_types)}")
    
    def predict_next_day(self, work_type, next_date=None):
        """Predict hours for next day"""
        if work_type not in self.models:
            raise ValueError(f"No model available for WorkType {work_type}")
        
        model_info = self.models[work_type]
        model = model_info['model']
        
        # Get last data
        last_data = model_info['last_data'].copy()
        
        # Determine next date
        if next_date is None:
            next_date = model_info['last_date'] + timedelta(days=1)
        else:
            next_date = pd.to_datetime(next_date)
        
        # Create new row for prediction
        new_row = pd.DataFrame({
            'Date': [next_date],
            'WorkType': [work_type],
            'Hours': [np.nan],
            'SystemHours': [last_data['SystemHours'].iloc[-1]],  # Use last value
            'Quantity': [last_data['Quantity'].iloc[-1]]         # Use last value
        })
        
        # Combine with historical data
        combined = pd.concat([last_data, new_row], ignore_index=True)
        
        # Create features
        combined = self.create_features(combined)
        
        # Get features for prediction (last row)
        X_pred = combined.iloc[-1:][model_info['feature_columns']]
        
        # Fill NaN values
        X_pred = X_pred.fillna(0)
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        return {
            'work_type': work_type,
            'date': next_date,
            'predicted_hours': max(0, prediction),  # Ensure non-negative
            'model_mae': model_info['mae']
        }
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save models
        model_file = os.path.join(MODELS_DIR, 'lightgbm_models.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.models, f)
        
        logger.info(f"Models saved to {model_file}")
        
        # Save summary
        summary = []
        for work_type, info in self.models.items():
            summary.append({
                'WorkType': work_type,
                'MAE': info['mae'],
                'Samples': info['n_samples'],
                'LastDate': info['last_date']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(MODELS_DIR, 'model_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")


    def predict_week(self, work_type, start_date=None):
        """Predict hours for one week, passing predictions forward"""
        if work_type not in self.models:
            raise ValueError(f"No model available for WorkType {work_type}")

        model_info = self.models[work_type]
        last_data = model_info['last_data'].copy()
        predictions = []

        # Determine first prediction date
        if start_date is None:
            next_date = model_info['last_date'] + timedelta(days=1)
        else:
            next_date = pd.to_datetime(start_date)

        # Use a copy to accumulate predictions
        history = last_data.copy()

        for i in range(1):
            # Create new row for prediction
            new_row = pd.DataFrame({
                'Date': [next_date],
                'WorkType': [work_type],
                'Hours': [np.nan],
                'SystemHours': [history['SystemHours'].iloc[-1]],
                'Quantity': [history['Quantity'].iloc[-1]]
            })

            # Combine with historical data
            combined = pd.concat([history, new_row], ignore_index=True)
            combined = self.create_features(combined)
            X_pred = combined.iloc[-1:][model_info['feature_columns']]
            X_pred = X_pred.fillna(0)
            pred_hours = max(0, model_info['model'].predict(X_pred)[0])

            # Store prediction
            predictions.append({
                'date': next_date,
                'predicted_hours': pred_hours,
                'model_mae': model_info['mae']
            })

            # Update history for next prediction (simulate actual value with prediction)
            new_row['Hours'] = pred_hours
            history = pd.concat([history, new_row], ignore_index=True)
            next_date += timedelta(days=1)

        return predictions
    

    def tune_lightgbm(self, X_train, y_train):
        param_grid = {
            'num_leaves': [31, 50, 100],
            'max_depth': [5, 10, 20, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 300, 500]
        }
        model = lgb.LGBMRegressor()
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2)
        grid.fit(X_train, y_train)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_

def main():
    """Main training function"""
    logger.info("Starting LightGBM model training...")
    
    # Initialize predictor
    predictor = WorkforcePredictor()
    
    # Load data
    predictor.load_data()
    
    # Train models
    predictor.train_all_models()
    
    # Save models
    predictor.save_models()
    
    for wt in predictor.models.keys():
        week_preds = predictor.predict_week(wt)
        logger.info(f"\nWeekly predictions for WorkType {wt}:")
        for pred in week_preds:
            logger.info(
                f"Date: {pred['date'].strftime('%Y-%m-%d')}, "
                f"Predicted Hours: {pred['predicted_hours']:.2f}, "
                f"Model MAE: {pred['model_mae']:.3f}"
            )


if __name__ == "__main__":
    main()