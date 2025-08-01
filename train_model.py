"""
Enhanced train_model.py with comprehensive evaluation for unseen data
Focus: Robust validation, time series cross-validation, and business metrics
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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

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
    """Enhanced LightGBM predictor with robust evaluation capabilities"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.data = None
        self.evaluation_results = {}
        
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
        df['dayofyear'] = df['Date'].dt.dayofyear
        
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
            df[f'hours_roll_max_{window}'] = df['Hours'].shift(1).rolling(window, min_periods=1).max()
            df[f'hours_roll_min_{window}'] = df['Hours'].shift(1).rolling(window, min_periods=1).min()
            df[f'quantity_roll_mean_{window}'] = df['Quantity'].shift(1).rolling(window, min_periods=1).mean()
            df[f'systemhours_roll_mean_{window}'] = df['SystemHours'].shift(1).rolling(window, min_periods=1).mean()
        
        # Productivity features
        df['quantity_per_hour'] = df['Quantity'] / (df['SystemHours'] + 1)
        df['hours_system_ratio'] = df['Hours'] / (df['SystemHours'] + 1)
        
        # Trend features
        df['hours_ewm_7'] = df['Hours'].shift(1).ewm(span=7, adjust=False).mean()
        df['hours_ewm_28'] = df['Hours'].shift(1).ewm(span=28, adjust=False).mean()
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        # Define feature columns (exclude target and identifiers)
        exclude_cols = ['Date', 'WorkType', 'Hours']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Split features and target
        X = df[self.feature_columns]
        y = df['Hours']
        
        return X, y
    
    def adaptive_hyperparameter_tuning(self, X_train, y_train, work_type):
        """
        Adaptive hyperparameter tuning based on data characteristics
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        work_type : str
            Work type being trained
            
        Returns:
        --------
        dict
            Optimized parameters for LightGBM
        """
        # Start with base parameters from config
        params = DEFAULT_MODEL_PARAMS.copy()
        
        # 1. Adjust based on data size
        n_samples = len(X_train)
        
        if n_samples < 500:
            # Small dataset - prevent overfitting
            params.update({
                'num_leaves': 20,
                'max_depth': 5,
                'min_child_samples': 20,
                'n_estimators': 200,
                'subsample': 0.6,
                'colsample_bytree': 0.6
            })
            logger.info(f"WorkType {work_type}: Small dataset ({n_samples} samples) - using conservative parameters")
            
        elif n_samples < 2000:
            # Medium dataset
            params.update({
                'num_leaves': 31,
                'max_depth': 7,
                'min_child_samples': 15,
                'n_estimators': 500,
                'subsample': 0.7,
                'colsample_bytree': 0.7
            })
            logger.info(f"WorkType {work_type}: Medium dataset ({n_samples} samples) - using balanced parameters")
            
        else:
            # Large dataset - can handle more complexity
            params.update({
                'num_leaves': 50,
                'max_depth': -1,
                'min_child_samples': 30,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            })
            logger.info(f"WorkType {work_type}: Large dataset ({n_samples} samples) - using complex parameters")
        
        # 2. Adjust based on target variance
        cv = y_train.std() / (y_train.mean() + 1e-6)  # Coefficient of variation
        
        if cv > 0.5:
            # High variance - need more regularization
            params.update({
                'reg_alpha': 0.2,
                'reg_lambda': 0.2,
                'min_gain_to_split': 0.02
            })
            logger.info(f"WorkType {work_type}: High variance (CV={cv:.3f}) - increased regularization")
        
        # 3. Adjust based on feature count
        n_features = X_train.shape[1]
        
        if n_features > 50:
            # Many features - use feature subsampling
            params['colsample_bytree'] = min(0.7, 30 / n_features)
            logger.info(f"WorkType {work_type}: Many features ({n_features}) - adjusted feature sampling")
        
        # 4. Special handling for specific work types
        if work_type in ['206', '213']:
            # These use NoRows instead of Quantity - might need different tuning
            params['learning_rate'] = 0.02  # Slower learning
            logger.info(f"WorkType {work_type}: Special handling applied")
        
        return params
    
    def evaluate_on_holdout(self, model, X_test, y_test, work_type):
        """Comprehensive evaluation on holdout test set"""
        y_pred = model.predict(X_test)
        
        # Basic metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Business metrics
        # Handle division by zero
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = 0
            
        bias = np.mean(y_pred - y_test)
        
        # Accuracy within thresholds
        if mask.sum() > 0:
            within_5_pct = np.sum(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]) <= 0.05) / mask.sum() * 100
            within_10_pct = np.sum(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]) <= 0.10) / mask.sum() * 100
        else:
            within_5_pct = 0
            within_10_pct = 0
        
        # Store results
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Bias': bias,
            'Within_5%': within_5_pct,
            'Within_10%': within_10_pct,
            'Test_Size': len(y_test)
        }
        
        logger.info(f"WorkType {work_type} Test Results:")
        logger.info(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        logger.info(f"  MAPE: {mape:.2f}%, Bias: {bias:.3f}")
        logger.info(f"  Within 5%: {within_5_pct:.1f}%, Within 10%: {within_10_pct:.1f}%")
        
        return results
    
    def time_series_cv_evaluation(self, X, y, work_type, adaptive_params, n_splits=5):
        """Perform time series cross-validation with detailed metrics"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model with adaptive parameters and early stopping
            model = lgb.LGBMRegressor(**adaptive_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            fold_results = {
                'fold': fold + 1,
                'mae': mean_absolute_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'r2': r2_score(y_val, y_pred),
                'val_size': len(y_val)
            }
            cv_results.append(fold_results)
        
        # Calculate average metrics
        avg_mae = np.mean([r['mae'] for r in cv_results])
        avg_rmse = np.mean([r['rmse'] for r in cv_results])
        avg_r2 = np.mean([r['r2'] for r in cv_results])
        std_mae = np.std([r['mae'] for r in cv_results])
        
        logger.info(f"WorkType {work_type} CV Results:")
        logger.info(f"  Average MAE: {avg_mae:.3f} (±{std_mae:.3f})")
        logger.info(f"  Average RMSE: {avg_rmse:.3f}")
        logger.info(f"  Average R²: {avg_r2:.3f}")
        
        return cv_results, avg_mae
    
    def train_model_for_worktype(self, work_type):
        """Train LightGBM model with adaptive tuning and robust evaluation"""
        # Filter data
        wt_data = self.data[self.data['WorkType'] == work_type].copy()
        
        if len(wt_data) < 50:
            logger.warning(f"Insufficient data for WorkType {work_type}: {len(wt_data)} records")
            return None
        
        # Create features
        wt_data = self.create_features(wt_data)
        
        # Remove rows with too many NaN values
        wt_data = wt_data.dropna(subset=['hours_lag_28'])
        
        if len(wt_data) < 30:
            logger.warning(f"Insufficient data after feature creation for WorkType {work_type}")
            return None
        
        # Prepare training data
        X, y = self.prepare_training_data(wt_data)
        X = X.fillna(0)
        
        # Hold out last 20% for final testing
        test_size = int(0.2 * len(X))
        X_train_cv = X.iloc[:-test_size]
        y_train_cv = y.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]
        
        # Get adaptive parameters
        adaptive_params = self.adaptive_hyperparameter_tuning(X_train_cv, y_train_cv, work_type)
        
        # Perform time series cross-validation
        n_splits = min(5, len(X_train_cv) // 50)
        if n_splits >= 2:
            cv_results, avg_mae = self.time_series_cv_evaluation(
                X_train_cv, y_train_cv, work_type, adaptive_params, n_splits
            )
        else:
            # For very small datasets, use simple train-val split
            logger.info(f"Using simple train-val split for WorkType {work_type} due to limited data")
            split_idx = int(0.8 * len(X_train_cv))
            X_train = X_train_cv.iloc[:split_idx]
            y_train = y_train_cv.iloc[:split_idx]
            X_val = X_train_cv.iloc[split_idx:]
            y_val = y_train_cv.iloc[split_idx:]
            
            model = lgb.LGBMRegressor(**adaptive_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            avg_mae = mean_absolute_error(y_val, y_pred)
            cv_results = [{
                'fold': 1,
                'mae': avg_mae,
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'r2': r2_score(y_val, y_pred),
                'val_size': len(y_val)
            }]
        
        # Train final model on all training data with adaptive parameters
        final_model = lgb.LGBMRegressor(**adaptive_params)
        final_model.fit(X_train_cv, y_train_cv)
        
        # Evaluate on holdout test set
        test_results = self.evaluate_on_holdout(final_model, X_test, y_test, work_type)
        
        # Train production model on all data with adaptive parameters
        production_model = lgb.LGBMRegressor(**adaptive_params)
        production_model.fit(X, y)
        
        # Store model information
        model_info = {
            'model': production_model,
            'feature_columns': self.feature_columns,
            'last_date': wt_data['Date'].max(),
            'last_data': wt_data.tail(30),
            'cv_mae': avg_mae,
            'test_mae': test_results['MAE'],
            'test_metrics': test_results,
            'cv_results': cv_results,
            'n_samples': len(wt_data),
            'adaptive_params': adaptive_params
        }
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': production_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        model_info['feature_importance'] = importance_df
        
        logger.info(f"\nTop 5 features for WorkType {work_type}:")
        logger.info(importance_df.head())
        
        # Store evaluation results
        self.evaluation_results[work_type] = {
            'cv_results': cv_results,
            'test_results': test_results,
            'feature_importance': importance_df.head(10).to_dict(),
            'adaptive_params': adaptive_params
        }
        
        return model_info
    
    def train_all_models(self):
        """Train models for all work types with evaluation"""
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
        
        # Generate evaluation report
        self.generate_evaluation_report()
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        report_data = []
        
        for work_type, info in self.models.items():
            report_data.append({
                'WorkType': work_type,
                'CV_MAE': info['cv_mae'],
                'Test_MAE': info['test_mae'],
                'Test_R2': info['test_metrics']['R2'],
                'Test_MAPE': info['test_metrics']['MAPE'],
                'Within_5%': info['test_metrics']['Within_5%'],
                'Within_10%': info['test_metrics']['Within_10%'],
                'Samples': info['n_samples']
            })
        
        report_df = pd.DataFrame(report_data)
        report_file = os.path.join(MODELS_DIR, 'evaluation_report.csv')
        report_df.to_csv(report_file, index=False)
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Average Test MAE: {report_df['Test_MAE'].mean():.3f}")
        logger.info(f"Average Test R²: {report_df['Test_R2'].mean():.3f}")
        logger.info(f"Average MAPE: {report_df['Test_MAPE'].mean():.2f}%")
        logger.info(f"Models with MAE < 0.5: {len(report_df[report_df['Test_MAE'] < 0.5])}")
        logger.info(f"Models with R² > 0.85: {len(report_df[report_df['Test_R2'] > 0.85])}")
        logger.info("="*60)
        
        # Save detailed performance summary
        self.save_performance_plots(report_df)
        
        return report_df
    
    def save_performance_plots(self, report_df):
        """Generate and save performance visualization plots"""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: MAE by WorkType
            ax1 = axes[0, 0]
            ax1.bar(report_df['WorkType'], report_df['Test_MAE'])
            ax1.axhline(y=0.5, color='r', linestyle='--', label='Target MAE')
            ax1.set_xlabel('Work Type')
            ax1.set_ylabel('Test MAE')
            ax1.set_title('Model Performance: MAE by Work Type')
            ax1.legend()
            
            # Plot 2: R² by WorkType
            ax2 = axes[0, 1]
            ax2.bar(report_df['WorkType'], report_df['Test_R2'])
            ax2.axhline(y=0.85, color='r', linestyle='--', label='Target R²')
            ax2.set_xlabel('Work Type')
            ax2.set_ylabel('Test R²')
            ax2.set_title('Model Performance: R² by Work Type')
            ax2.legend()
            
            # Plot 3: Sample Size vs Performance
            ax3 = axes[1, 0]
            ax3.scatter(report_df['Samples'], report_df['Test_MAE'])
            ax3.set_xlabel('Number of Samples')
            ax3.set_ylabel('Test MAE')
            ax3.set_title('Sample Size vs Model Performance')
            
            # Plot 4: Accuracy within thresholds
            ax4 = axes[1, 1]
            x = np.arange(len(report_df))
            width = 0.35
            ax4.bar(x - width/2, report_df['Within_5%'], width, label='Within 5%')
            ax4.bar(x + width/2, report_df['Within_10%'], width, label='Within 10%')
            ax4.set_xlabel('Work Type')
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Prediction Accuracy Within Thresholds')
            ax4.set_xticks(x)
            ax4.set_xticklabels(report_df['WorkType'])
            ax4.legend()
            
            plt.tight_layout()
            plot_file = os.path.join(MODELS_DIR, 'model_performance_summary.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance plots saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate performance plots: {str(e)}")
    
    def predict_next_day(self, work_type, next_date=None):
        """Predict hours for next day with confidence intervals"""
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
            'SystemHours': [last_data['SystemHours'].iloc[-1]],
            'Quantity': [last_data['Quantity'].iloc[-1]]
        })
        
        # Combine with historical data
        combined = pd.concat([last_data, new_row], ignore_index=True)
        
        # Create features
        combined = self.create_features(combined)
        
        # Get features for prediction (last row)
        X_pred = combined.iloc[-1:][model_info['feature_columns']]
        X_pred = X_pred.fillna(0)
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        
        # Calculate confidence interval based on test MAE
        test_mae = model_info['test_mae']
        confidence_lower = max(0, prediction - 1.96 * test_mae)
        confidence_upper = prediction + 1.96 * test_mae
        
        return {
            'work_type': work_type,
            'date': next_date,
            'predicted_hours': max(0, prediction),
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'test_mae': test_mae,
            'cv_mae': model_info['cv_mae']
        }
    
    def predict_week(self, work_type, start_date=None):
        """Predict hours for one week with rolling predictions"""
        if work_type not in self.models:
            raise ValueError(f"No model available for WorkType {work_type}")

        predictions = []
        
        # Determine first prediction date
        if start_date is None:
            next_date = self.models[work_type]['last_date'] + timedelta(days=1)
        else:
            next_date = pd.to_datetime(start_date)

        # Make predictions for 7 days
        for i in range(7):
            pred = self.predict_next_day(work_type, next_date)
            predictions.append(pred)
            
            # Update last_data with the prediction for next iteration
            new_row = pd.DataFrame({
                'Date': [next_date],
                'WorkType': [work_type],
                'Hours': [pred['predicted_hours']],
                'SystemHours': [self.models[work_type]['last_data']['SystemHours'].iloc[-1]],
                'Quantity': [self.models[work_type]['last_data']['Quantity'].iloc[-1]]
            })
            
            # Update last_data in model info (keep only last 30 days)
            self.models[work_type]['last_data'] = pd.concat([
                self.models[work_type]['last_data'].iloc[1:],
                new_row
            ], ignore_index=True)
            
            next_date += timedelta(days=1)

        return pd.DataFrame(predictions)
    
    def save_models(self):
        """Save trained models and evaluation results"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save models
        model_file = os.path.join(MODELS_DIR, 'lightgbm_models.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.models, f)
        
        logger.info(f"Models saved to {model_file}")
        
        # Save evaluation results
        eval_file = os.path.join(MODELS_DIR, 'evaluation_results.pkl')
        with open(eval_file, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
        
        logger.info(f"Evaluation results saved to {eval_file}")
        
        # Save summary
        summary = []
        for work_type, info in self.models.items():
            summary.append({
                'WorkType': work_type,
                'CV_MAE': info['cv_mae'],
                'Test_MAE': info['test_mae'],
                'Samples': info['n_samples'],
                'LastDate': info['last_date']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(MODELS_DIR, 'model_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to {summary_file}")


def main():
    """Main training function with evaluation"""
    logger.info("Starting Enhanced LightGBM model training with evaluation...")
    
    # Initialize predictor
    predictor = WorkforcePredictor()
    
    # Load data
    predictor.load_data()
    
    # Train models with evaluation
    predictor.train_all_models()
    
    # Save models and results
    predictor.save_models()
    
    # Test prediction with confidence intervals
    test_wt = '206'
    if test_wt in predictor.models:
        pred = predictor.predict_next_day(test_wt)
        logger.info(f"\nTest prediction for WorkType {test_wt}:")
        logger.info(f"Date: {pred['date'].strftime('%Y-%m-%d')}")
        logger.info(f"Predicted Hours: {pred['predicted_hours']:.2f}")
        logger.info(f"95% Confidence Interval: [{pred['confidence_lower']:.2f}, {pred['confidence_upper']:.2f}]")
        logger.info(f"Test MAE: {pred['test_mae']:.3f}")


if __name__ == "__main__":
    main()