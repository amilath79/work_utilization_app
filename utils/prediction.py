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
from utils.holiday_utils import is_non_working_day

# Import the torch_utils module for neural network support
try:
    from utils.torch_utils import load_torch_models, predict_with_torch_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import from configuration
from config import MODELS_DIR

# Import holiday utils
from utils.holiday_utils import is_swedish_holiday

# Configure logger
logger = logging.getLogger(__name__)

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
    Make prediction using the PyTorch neural network model
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
        
        # Extract features we want to use (must match what was used in training)
        features = ['NoOfMan', 'NoOfMan_lag_1', 'NoOfMan_lag_7', 
                    'NoOfMan_rolling_mean_7', 'DayOfWeek_feat', 'Month_feat', 
                    'IsWeekend_feat']
        
        sequence = recent_data[features].values
        
        # Ensure sequence is in chronological order (oldest to newest)
        sequence = sequence[::-1]
        
        # Use the utility function to make prediction
        prediction = predict_with_torch_model(model, scaler, sequence, sequence_length)
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error predicting with neural network: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def predict_next_day(df, models, date=None, use_neural_network=False):
    """
    Predict NoOfMan for the next day for each WorkType
    """
    try:
        # Find the latest date in the dataset or use specified date
        latest_date = df['Date'].max() if date is None else date
        next_date = latest_date + timedelta(days=1)
        
        logger.info(f"Predicting NoOfMan for {next_date.strftime('%Y-%m-%d')}")
        
       # Check if the prediction date is a non-working day (holiday or Saturday)
        is_nonworking, reason = is_non_working_day(next_date)

        if is_nonworking:
            logger.info(f"Date {next_date.strftime('%Y-%m-%d')} is a non-working day: {reason}")
            logger.info("No work is carried out on this day. Setting all predictions to 0.")
    
            # Return zero predictions for all work types
            zero_predictions = {work_type: 0 for work_type in models.keys()}
            return next_date, zero_predictions

        # Load neural network models if requested
        nn_models, nn_scalers, nn_metrics = {}, {}, {}
        if use_neural_network:
            nn_models, nn_scalers, nn_metrics = load_neural_models()
        
        # Get the day of week, month, etc. for the next day
        next_day_features = {
            'Date': next_date,
            'DayOfWeek_feat': next_date.dayofweek,
            'Month_feat': next_date.month,
            'IsWeekend_feat': 1 if next_date.dayofweek >= 5 else 0,
            'Year_feat': next_date.year,
            'Quarter': (next_date.month - 1) // 3 + 1,
            'DayOfMonth': next_date.day,
            'WeekOfYear': next_date.isocalendar()[1]
        }
        
        predictions = {}
        
        for work_type in models.keys():
            # Try neural network prediction first if requested
            if use_neural_network and nn_models and work_type in nn_models:
                nn_prediction = predict_with_neural_network(df, nn_models, nn_scalers, work_type, date)
                
                if nn_prediction is not None:
                    predictions[work_type] = nn_prediction
                    logger.info(f"Neural network predicted {nn_prediction:.2f} workers for WorkType {work_type}")
                    continue
            
            # Fall back to traditional model if neural prediction fails or wasn't requested
            # Filter data for this WorkType
            work_type_data = df[df['WorkType'] == work_type]
            
            if len(work_type_data) < 8:  # Need at least 7 days of history for lag features
                logger.warning(f"Not enough data for WorkType {work_type}. Skipping.")
                continue
                
            # Get the most recent values for lag features
            lag_features = {}
            for lag in [1, 2, 3, 7, 14, 30, 365]:
                try:
                    lag_date = latest_date - timedelta(days=lag)
                    lag_records = work_type_data[work_type_data['Date'] == lag_date]
                    lag_value = lag_records['NoOfMan'].sum() if not lag_records.empty else 0
                    lag_features[f'NoOfMan_lag_{lag}'] = lag_value
                except:
                    # If a lag day isn't available
                    lag_features[f'NoOfMan_lag_{lag}'] = 0
            
            # Calculate rolling statistics
            for window in [7, 14, 30]:
                try:
                    recent_data = work_type_data[work_type_data['Date'] > latest_date - timedelta(days=window)]
                    values = recent_data['NoOfMan'].values
                    
                    # Calculate rolling mean
                    rolling_mean = values.mean() if len(values) > 0 else 0
                    lag_features[f'NoOfMan_rolling_mean_{window}'] = rolling_mean
                    
                    # Calculate other rolling statistics if needed by the model
                    lag_features[f'NoOfMan_rolling_max_{window}'] = values.max() if len(values) > 0 else 0
                    lag_features[f'NoOfMan_rolling_min_{window}'] = values.min() if len(values) > 0 else 0
                    lag_features[f'NoOfMan_rolling_std_{window}'] = values.std() if len(values) > 1 else 0
                except:
                    lag_features[f'NoOfMan_rolling_mean_{window}'] = 0
                    lag_features[f'NoOfMan_rolling_max_{window}'] = 0
                    lag_features[f'NoOfMan_rolling_min_{window}'] = 0
                    lag_features[f'NoOfMan_rolling_std_{window}'] = 0
            
            # Same day of week lag
            try:
                same_dow_records = work_type_data[
                    (work_type_data['DayOfWeek_feat'] == next_day_features['DayOfWeek_feat']) &
                    (work_type_data['Date'] < latest_date)
                ].sort_values('Date', ascending=False)
                
                same_dow_value = same_dow_records.iloc[0]['NoOfMan'] if not same_dow_records.empty else 0
                lag_features['NoOfMan_same_dow_lag'] = same_dow_value
            except:
                lag_features['NoOfMan_same_dow_lag'] = 0
            
            # Same day of month lag
            try:
                same_dom_records = work_type_data[
                    (work_type_data['DayOfMonth'] == next_day_features['DayOfMonth']) &
                    (work_type_data['Date'] < latest_date)
                ].sort_values('Date', ascending=False)
                
                same_dom_value = same_dom_records.iloc[0]['NoOfMan'] if not same_dom_records.empty else 0
                lag_features['NoOfMan_same_dom_lag'] = same_dom_value
            except:
                lag_features['NoOfMan_same_dom_lag'] = 0
            
            # Add trend indicators
            try:
                lag_features['NoOfMan_7day_trend'] = lag_features['NoOfMan_lag_1'] - lag_features['NoOfMan_lag_7']
                lag_features['NoOfMan_1day_trend'] = lag_features['NoOfMan_lag_1'] - lag_features['NoOfMan_lag_2']
            except:
                lag_features['NoOfMan_7day_trend'] = 0
                lag_features['NoOfMan_1day_trend'] = 0
            
            # Combine features
            all_features = {**next_day_features, **lag_features}
            
            # Prepare input for prediction based on model's expected features
            model = models[work_type]
            required_features = _get_required_features(model)
            
            # Create prediction dataframe with only the required features
            X_pred = pd.DataFrame([{
                feature: all_features.get(feature, 0) 
                for feature in required_features
            }])
            
            # Make prediction
            try:
                prediction = model.predict(X_pred)[0]
                
                # Ensure prediction is not negative
                prediction = max(0, prediction)
                
                predictions[work_type] = prediction
                logger.info(f"RandomForest predicted {prediction:.2f} workers for WorkType {work_type}")
            except Exception as e:
                logger.error(f"Error predicting for WorkType {work_type}: {str(e)}")
                continue
        
        return next_date, predictions
    
    except Exception as e:
        logger.error(f"Error predicting next day: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to predict next day: {str(e)}")

# Modify the predict_multiple_days function to check for non-working days
def predict_multiple_days(df, models, num_days=7, use_neural_network=False):
    """
    Predict NoOfMan for multiple days for each WorkType
    """
    try:
        logger.info(f"Predicting for the next {num_days} days")
        
        # Initialize results dictionary and holiday info dictionary
        multi_day_predictions = {}
        nonworking_info = {}
        
        # Create a working copy of the dataframe that we'll extend with predictions
        current_df = df.copy()
        
        # Find the latest date in the dataset
        latest_date = current_df['Date'].max()
        
        # Predict for each day
        for i in range(num_days):
            prediction_date = latest_date + timedelta(days=i+1)  # +1 to start with the next day
            
            # Check if the date is a non-working day
            is_nonworking, reason = is_non_working_day(prediction_date)
            
            if is_nonworking:
                logger.info(f"Date {prediction_date.strftime('%Y-%m-%d')} is a non-working day: {reason}")
                logger.info("No work is carried out on this day. Setting all predictions to 0.")
                
                # Create zero predictions for all work types
                zero_predictions = {work_type: 0 for work_type in models.keys()}
                
                # Store predictions and non-working day info
                multi_day_predictions[prediction_date] = zero_predictions
                nonworking_info[prediction_date] = reason
                
                # Add the zero predictions to the dataframe for the next iteration
                new_rows = []
                for work_type in models.keys():
                    new_row = {
                        'Date': prediction_date,
                        'WorkType': work_type,
                        'NoOfMan': 0,
                        
                        # Add the date features
                        'DayOfWeek_feat': prediction_date.dayofweek,
                        'Month_feat': prediction_date.month,
                        'IsWeekend_feat': 1 if prediction_date.dayofweek == 5 else 0,  # Only Saturday is a weekend
                        'Year_feat': prediction_date.year,
                        'Quarter': (prediction_date.month - 1) // 3 + 1,
                        'DayOfMonth': prediction_date.day,
                        'WeekOfYear': prediction_date.isocalendar()[1]
                    }
                    
                    # Add necessary lag features for the next iteration
                    new_rows.append(new_row)
                
                # Append new rows to the dataframe
                if new_rows:
                    current_df = pd.concat([current_df, pd.DataFrame(new_rows)], ignore_index=True)
                continue
            
            # Not a non-working day, proceed with normal prediction
            next_date, predictions = predict_next_day(
                current_df, 
                models, 
                date=latest_date + timedelta(days=i), 
                use_neural_network=use_neural_network
            )
            
            multi_day_predictions[next_date] = predictions
            
            # Add the predictions back to the dataframe for the next iteration
            new_rows = []
            for work_type, pred_value in predictions.items():
                new_row = {
                    'Date': next_date,
                    'WorkType': work_type,
                    'NoOfMan': pred_value,
                    
                    # Add the date features - Update the IsWeekend_feat to only count Saturday
                    'DayOfWeek_feat': next_date.dayofweek,
                    'Month_feat': next_date.month,
                    'IsWeekend_feat': 1 if next_date.dayofweek == 5 else 0,  # Only Saturday is a weekend
                    'Year_feat': next_date.year,
                    'Quarter': (next_date.month - 1) // 3 + 1,
                    'DayOfMonth': next_date.day,
                    'WeekOfYear': next_date.isocalendar()[1]
                }
                
                # Add necessary lag features for the next iteration
                new_rows.append(new_row)
            
            # Append new rows to the dataframe
            if new_rows:
                current_df = pd.concat([current_df, pd.DataFrame(new_rows)], ignore_index=True)
        
        logger.info(f"Predictions completed for {num_days} days")
        return multi_day_predictions, nonworking_info
    
    except Exception as e:
        logger.error(f"Error predicting multiple days: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to predict multiple days: {str(e)}")

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
        y_true_nonzero = np.array([max(0.0001, y) for y in y_true])
        mape = np.mean(np.abs((y_true_nonzero - y_pred) / y_true_nonzero)) * 100
        
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
    """Helper function to get feature names required by the model"""
    try:
        # If it's a pipeline, get the feature names from the preprocessor
        if hasattr(model, 'steps'):
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                # Get the categorical features from the preprocessor
                cat_cols = []
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'cat' and hasattr(transformer, 'categories_'):
                            for i, category_list in enumerate(transformer.categories_):
                                original_col = cols[i]
                                cat_cols.append(original_col)
                
                # The numeric features are passed through
                remainder_cols = []
                if hasattr(preprocessor, 'remainder') and preprocessor.remainder != 'drop':
                    remainder_cols = [
                        col for col in [
                            'NoOfMan_lag_1', 'NoOfMan_lag_2', 'NoOfMan_lag_3', 'NoOfMan_lag_7',
                            'NoOfMan_lag_14', 'NoOfMan_lag_30', 'NoOfMan_lag_365',
                            'NoOfMan_rolling_mean_7', 'NoOfMan_rolling_mean_14', 'NoOfMan_rolling_mean_30',
                            'NoOfMan_rolling_max_7', 'NoOfMan_rolling_min_7', 'NoOfMan_rolling_std_7',
                            'NoOfMan_same_dow_lag', 'NoOfMan_same_dom_lag',
                            'NoOfMan_7day_trend', 'NoOfMan_1day_trend',
                            'IsWeekend_feat'
                        ]
                    ]
                
                return cat_cols + remainder_cols
        
        # Default set of features if we can't determine them from the model
        return [
            'DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat',
            'NoOfMan_lag_1', 'NoOfMan_lag_2', 'NoOfMan_lag_3', 'NoOfMan_lag_7',
            'NoOfMan_rolling_mean_7'
        ]
    
    except Exception as e:
        logger.error(f"Error getting required features: {str(e)}")
        
        # Return minimal set of features
        return [
            'DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat',
            'NoOfMan_lag_1', 'NoOfMan_lag_2', 'NoOfMan_lag_3', 'NoOfMan_lag_7',
            'NoOfMan_rolling_mean_7'
        ]