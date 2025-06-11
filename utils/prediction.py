"""
Prediction utilities for work utilization forecasting with multiple model types.
"""
import pandas as pd
import numpy as np
import logging
import pickle
import os
import traceback
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Update the import statement
from utils.holiday_utils import is_working_day_for_punch_code
from utils.feature_engineering import EnhancedFeatureTransformer
# Import the torch_utils module for neural network support
try:
    from utils.torch_utils import load_torch_models, predict_with_torch_model
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import from configuration
from config import (
    MODELS_DIR, DATA_DIR, LAG_DAYS, ROLLING_WINDOWS, CHUNK_SIZE, DEFAULT_MODEL_PARAMS,
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, SQL_USERNAME, SQL_PASSWORD,
    FEATURE_GROUPS, PRODUCTIVITY_FEATURES, DATE_FEATURES, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS
)

# Configure logger
logger = logging.getLogger(__name__)

# Import holiday utils
from utils.holiday_utils import is_swedish_holiday

def get_required_features():
    """Get required features based on config - simple and direct"""
    numeric_features = []
    categorical_features = []
    
    # Essential lag features
    if FEATURE_GROUPS['LAG_FEATURES']:
        for lag in ESSENTIAL_LAGS:
            numeric_features.append(f'NoOfMan_lag_{lag}')
        if FEATURE_GROUPS['PRODUCTIVITY_FEATURES']:
            numeric_features.append('Quantity_lag_1')
    
    # Essential rolling features  
    if FEATURE_GROUPS['ROLLING_FEATURES']:
        for window in ESSENTIAL_WINDOWS:
            numeric_features.append(f'NoOfMan_rolling_mean_{window}')
    
    # Date features from config
    if FEATURE_GROUPS['DATE_FEATURES']:
        categorical_features.extend(DATE_FEATURES['categorical'])
        numeric_features.extend(DATE_FEATURES['numeric'])
        numeric_features.append('DayOfMonth')
    
    # Productivity features from config
    if FEATURE_GROUPS['PRODUCTIVITY_FEATURES']:
        numeric_features.extend(PRODUCTIVITY_FEATURES)
    
    # Pattern features (optional)
    if FEATURE_GROUPS['PATTERN_FEATURES']:
        numeric_features.append('NoOfMan_same_dow_lag')
    
    # Trend features (optional)  
    if FEATURE_GROUPS['TREND_FEATURES']:
        numeric_features.append('NoOfMan_trend_7d')
    
    return numeric_features, categorical_features


def calculate_hours_prediction(df, work_type, no_of_man_prediction, date=None):
    """
    Calculate hours prediction based on NoOfMan prediction and historical patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with historical data
    work_type : str
        Work type (punch code)
    no_of_man_prediction : float
        Predicted number of workers
    date : datetime, optional
        Date for the prediction
    
    Returns:
    --------
    float
        Predicted hours
    """
    try:
        # Default hours per worker (8 hours per day)
        default_hours_per_worker = 8.0
        
        # Filter data for this work type
        wt_data = df[df['WorkType'] == work_type]
        
        if len(wt_data) < 5 or 'Hours' not in wt_data.columns:
            # Not enough data or no Hours column, use default
            return no_of_man_prediction * default_hours_per_worker
        
        # Calculate the average hours per worker for this work type
        wt_data = wt_data[(wt_data['NoOfMan'] > 0) & (wt_data['Hours'] > 0)]  # Filter for valid data
        
        if len(wt_data) == 0:
            return no_of_man_prediction * default_hours_per_worker
            
        # Calculate historical ratio
        hours_per_worker_ratios = wt_data['Hours'] / wt_data['NoOfMan']
        
        # Get average ratio
        avg_hours_per_worker = hours_per_worker_ratios.mean()
        
        # Handle extreme or invalid values
        if pd.isna(avg_hours_per_worker) or avg_hours_per_worker <= 0 or avg_hours_per_worker > 24:
            avg_hours_per_worker = default_hours_per_worker
        
        # If date is provided, check for day-of-week patterns
        if date is not None:
            # Get day of week
            day_of_week = date.weekday()
            
            # Filter data for this day of week
            dow_data = wt_data[wt_data['Date'].dt.weekday == day_of_week]
            
            if len(dow_data) >= 3:  # If we have enough data for this day of week
                # Calculate day-of-week specific ratio
                dow_ratios = dow_data['Hours'] / dow_data['NoOfMan'] 
                dow_avg = dow_ratios.mean()
                
                # Use day-of-week specific average if it's valid
                if not pd.isna(dow_avg) and dow_avg > 0 and dow_avg <= 24:
                    avg_hours_per_worker = dow_avg
        
        # Calculate predicted hours using the adaptive ratio
        predicted_hours = no_of_man_prediction * avg_hours_per_worker
        
        return predicted_hours
        
    except Exception as e:
        logger.error(f"Error calculating hours prediction: {str(e)}")
        # Fallback to simple calculation
        return no_of_man_prediction * 8.0



def load_neural_models():
    """
    Load neural network models, scalers, and metrics
    """
    try:
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch is not available. Neural network models will not be loaded.")
            return {}, {}, {}
            
        return load_torch_models(MODELS_DIR)
    
    except Exception as e:
        logger.error(f"Error loading neural network models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, {}

def predict_with_neural_network(df, nn_models, nn_scalers, work_type, date=None, sequence_length=7):
    """
    Make prediction using PyTorch neural network model with config-driven features
    """
    try:
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch is not available. Cannot use neural network for prediction.")
            return None
            
        if work_type not in nn_models or work_type not in nn_scalers:
            logger.warning(f"No neural network model available for WorkType: {work_type}")
            return None
        
        # Get model and scaler
        model = nn_models[work_type]
        scaler = nn_scalers[work_type]
        
        # Filter data for this WorkType
        work_type_data = df[df['WorkType'] == work_type]
        
        if len(work_type_data) < sequence_length:
            logger.warning(f"Not enough data for neural prediction for WorkType: {work_type}")
            return None
        
        # Sort by date and get the most recent data
        work_type_data = work_type_data.sort_values('Date', ascending=False)
        recent_data = work_type_data.head(sequence_length)
        
        # Get features using same config as training
        numeric_features, categorical_features = get_required_features()
        all_features = numeric_features + categorical_features
        
        # Filter to only include features that actually exist in the data
        available_features = [f for f in all_features if f in recent_data.columns]
        
        # Validation: Check if we have minimum required features
        if len(available_features) < 4:
            logger.warning(f"Not enough features available for neural prediction for WorkType: {work_type}. "
                         f"Available: {available_features}")
            return None
        
        # Log features being used
        active_groups = [group for group, enabled in FEATURE_GROUPS.items() if enabled]
        logger.info(f"Neural network using {len(available_features)} features from groups {active_groups} for {work_type}")
        
        # ✅ EXTRACT SEQUENCE USING AVAILABLE FEATURES
        try:
            sequence = recent_data[available_features].values
        except KeyError as e:
            logger.error(f"Error extracting features for neural network: {str(e)}")
            return None
        
        # ✅ VALIDATE SEQUENCE SHAPE
        if sequence.shape[1] != len(available_features):
            logger.warning(f"Sequence shape mismatch for {work_type}. Expected {len(available_features)} features, "
                         f"got {sequence.shape[1]}")
            return None
        
        # Ensure sequence is in chronological order (oldest to newest)
        sequence = sequence[::-1]
        

        # Check if model expects this input size (optional validation)
        try:
            # This is a rough check - you might need to adjust based on your model architecture
            test_input = torch.tensor(sequence.reshape(1, sequence_length, -1), dtype=torch.float32)
            
            # Use the utility function to make prediction
            prediction = predict_with_torch_model(model, scaler, sequence, sequence_length)
            
            logger.info(f"Neural network prediction successful for {work_type}: {prediction:.3f}")
            return prediction
            
        except Exception as model_error:
            logger.error(f"Model prediction error for {work_type}: {str(model_error)}")
            logger.error(f"Sequence shape: {sequence.shape}, Features used: {available_features}")
            return None
    
    except Exception as e:
        logger.error(f"Error predicting with neural network: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
def create_prediction_features(df, work_type, next_date, latest_date):
    """
    Create prediction features using EnhancedFeatureTransformer
    """
    try:
        # Process historical data ONLY
        work_type_data = df[df['WorkType'] == work_type].copy()
        
        # Initialize and use the transformer
        feature_transformer = EnhancedFeatureTransformer()
        
        # Fit and transform the data
        feature_transformer.fit(work_type_data)
        features_df = feature_transformer.transform(work_type_data)
        
        # Get latest row and update date features only
        latest_features = features_df.iloc[-1:].copy()
        latest_features['Date'] = next_date
        latest_features['DayOfWeek_feat'] = next_date.weekday()
        latest_features['Month_feat'] = next_date.month
        latest_features['IsWeekend_feat'] = 1 if next_date.weekday() == 5 else 0
        latest_features['DayOfMonth'] = next_date.day
        latest_features['Quarter'] = (next_date.month - 1) // 3 + 1
        
        return latest_features
        
    except Exception as e:
        logger.error(f"Error creating prediction features: {str(e)}")
        return None
    
        
def predict_next_day(df, models, date=None, use_neural_network=False):
    """
    Simplified prediction using complete pipelines
    No manual feature engineering needed - pipeline handles everything!
    """
    try:
        logger.info("🚀 Starting next day prediction with complete pipelines")
        
        # Determine prediction date
        if date is None:
            latest_date = df['Date'].max()
            next_date = latest_date + timedelta(days=1)
        else:
            next_date = pd.to_datetime(date)
        
        # Skip weekends
        while next_date.weekday() >= 5:  # Saturday=5, Sunday=6
            next_date += timedelta(days=1)
        
        logger.info(f"Predicting for date: {next_date.strftime('%Y-%m-%d')}")
        
        predictions = {}
        hours_predictions = {}
        
        for work_type in models.keys():
            try:
                # Check if this punch code should work on this date
                is_working, reason = is_working_day_for_punch_code(next_date, work_type)
                
                if not is_working:
                    logger.info(f"Date {next_date.strftime('%Y-%m-%d')} is non-working for punch code {work_type}: {reason}")
                    predictions[work_type] = 0
                    hours_predictions[work_type] = 0
                    continue
                
                # Get the complete pipeline model
                pipeline = models[work_type]
                
                # Create simple prediction input - pipeline handles all feature engineering!
                work_type_data = df[df['WorkType'] == work_type].copy()
                
                if len(work_type_data) == 0:
                    logger.warning(f"No historical data for WorkType {work_type}")
                    continue
                
                # Get the most recent record as base for prediction
                latest_record = work_type_data.iloc[-1:].copy()
                
                # Create prediction row with just the basic information
                prediction_row = latest_record.copy()
                prediction_row['Date'] = next_date
                prediction_row['WorkType'] = work_type
                
                # The pipeline will handle ALL feature engineering automatically!
                prediction = pipeline.predict(prediction_row)[0]
                
                # Ensure prediction is not negative
                prediction = max(0, prediction)
                
                predictions[work_type] = prediction
                hours_predictions[work_type] = calculate_hours_prediction(df, work_type, prediction, next_date)
                
                logger.info(f"✅ Pipeline predicted {prediction:.2f} workers for WorkType {work_type}")
                
            except Exception as e:
                logger.error(f"Error predicting for WorkType {work_type}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"🎯 Predictions completed for {len(predictions)} work types")
        return next_date, predictions, hours_predictions
        
    except Exception as e:
        logger.error(f"Error in predict_next_day: {str(e)}")
        logger.error(traceback.format_exc())
        return None, {}, {}


def predict_multiple_days(df, models, start_date, num_days, use_neural_network=False):
    """
    Simplified multi-day prediction using complete pipelines
    """
    
    try:
        all_predictions = {}
        current_df = df.copy()
        
        current_date = pd.to_datetime(start_date)
        
        for day in range(num_days):
            # Predict for current date
            pred_date, daily_predictions, daily_hours = predict_next_day(
                current_df, models, current_date, use_neural_network
            )
            
            if pred_date and daily_predictions:
                all_predictions[pred_date] = daily_predictions
                
                # Add predictions to dataframe for next iteration
                for work_type, prediction in daily_predictions.items():
                    new_row = pd.DataFrame({
                        'Date': [pred_date],
                        'WorkType': [work_type], 
                        'NoOfMan': [prediction],
                        'Quantity': [current_df[current_df['WorkType'] == work_type]['Quantity'].iloc[-1] if len(current_df[current_df['WorkType'] == work_type]) > 0 else 0]
                    })
                    current_df = pd.concat([current_df, new_row], ignore_index=True)
            
            current_date += timedelta(days=1)
            
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
        
        return all_predictions
        
    except Exception as e:
        logger.error(f"Error in predict_multiple_days: {str(e)}")
        logger.error(traceback.format_exc())
        return {}
    

def evaluate_predictions(y_true, y_pred):
    """
    Calculate evaluation metrics for predictions
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle zero values in y_true to avoid division by zero
        y_true_nonzero = np.maximum(np.abs(y_true), 1.0)  # Use epsilon=1.0 like in training
        mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
    
    except Exception as e:
        logger.error(f"Error evaluating predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'MAE': float('nan'),
            'RMSE': float('nan'),
            'R²': float('nan'),
            'MAPE': float('nan')
        }
    

def _get_required_features(model):
    """Helper function to get feature names required by the model - config-driven"""
    try:
        # If it's a pipeline, try to get feature names from the model
        if hasattr(model, 'steps'):
            if hasattr(model.named_steps['model'], 'feature_names_in_'):
                return list(model.named_steps['model'].feature_names_in_)
            
            # Extract from preprocessor
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor and hasattr(preprocessor, 'transformers_'):
                feature_names = []
                for name, transformer, cols in preprocessor.transformers_:
                    feature_names.extend(cols)
                
                if feature_names:
                    return feature_names
        
        # Fallback: Use same config-driven features as training
        numeric_features, categorical_features = get_required_features()
        all_features = categorical_features + numeric_features
        
        active_groups = [group for group, enabled in FEATURE_GROUPS.items() if enabled]
        logger.info(f"Using config features from {active_groups}: {len(all_features)} total features")
        
        return all_features

    except Exception as e:
        logger.error(f"Error getting required features: {str(e)}")
        # Simple fallback
        return ['DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat', 'NoOfMan_lag_1', 'NoOfMan_lag_7', 'NoOfMan_rolling_mean_7']    