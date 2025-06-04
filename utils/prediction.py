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

# Import the torch_utils module for neural network support
try:
    from utils.torch_utils import load_torch_models, predict_with_torch_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import from configuration
from config import (
    MODELS_DIR, DATA_DIR, LAG_DAYS, ROLLING_WINDOWS,  # ✅ Uses config
    CHUNK_SIZE, DEFAULT_MODEL_PARAMS,
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION,
    SQL_USERNAME, SQL_PASSWORD
)
# Configure logger
logger = logging.getLogger(__name__)

# Import holiday utils
from utils.holiday_utils import is_swedish_holiday

def _precompute_feature_lists():
    """Pre-compute feature lists from tier configuration at module load time"""
    from utils.feature_builder import get_feature_lists_for_data
    
    feature_cache = {}
    
    # Get comprehensive feature lists based on tier configuration
    numeric_features, categorical_features = get_feature_lists_for_data()
    
    # Cache different combinations
    feature_cache['remainder'] = numeric_features
    feature_cache['default'] = categorical_features + numeric_features
    feature_cache['categorical'] = categorical_features
    feature_cache['numeric'] = numeric_features
    
    # Fallback features (basic tier only)
    fallback_features = ['DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat']
    for lag in [1, 2, 3, 7]:  # Basic lags
        fallback_features.append(f'NoOfMan_lag_{lag}')
    fallback_features.append('NoOfMan_rolling_mean_7')
    
    feature_cache['fallback'] = fallback_features
    
    # Log feature summary
    logger.info(f"Feature cache built: {len(numeric_features)} numeric, {len(categorical_features)} categorical features")
    
    return feature_cache

# Pre-compute at module load
_FEATURE_CACHE = _precompute_feature_lists()




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
        
        # ✅ FULLY DYNAMIC FEATURE GENERATION
        features = ['NoOfMan', 'DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat']
        
        # Add ALL available lag features from config
        for lag in LAG_DAYS:
            lag_feature = f'NoOfMan_lag_{lag}'
            features.append(lag_feature)
        
        # Add ALL available rolling features from config  
        for window in ROLLING_WINDOWS:
            rolling_feature = f'NoOfMan_rolling_mean_{window}'
            features.append(rolling_feature)
        
        # Filter to only include features that actually exist in the data
        available_features = [f for f in features if f in recent_data.columns]
        
        # ✅ VALIDATION: Check if we have minimum required features
        if len(available_features) < 4:  # At minimum need NoOfMan + 3 date features
            logger.warning(f"Not enough features available for neural prediction for WorkType: {work_type}. "
                         f"Available: {available_features}")
            return None
        
        # ✅ LOG WHAT FEATURES ARE BEING USED
        logger.info(f"Neural network using {len(available_features)} features for {work_type}: {available_features}")
        
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
        
        # ✅ DYNAMIC INPUT SIZE CHECK
        expected_input_size = sequence.shape[1]
        
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

def predict_next_day(df, models, date=None, use_neural_network=False):
    """
    Predict NoOfMan and Hours for the next day for each WorkType
    """
    try:
        # Find the latest date in the dataset or use specified date
        latest_date = df['Date'].max() if date is None else date
        next_date = latest_date + timedelta(days=1)
        
        logger.info(f"Predicting NoOfMan for {next_date.strftime('%Y-%m-%d')}")
        
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
        hours_predictions = {}
        
        for work_type in models.keys():
            # Check if this punch code should work on this date
            is_working, reason = is_working_day_for_punch_code(next_date, work_type)
            
            if not is_working:
                logger.info(f"Date {next_date.strftime('%Y-%m-%d')} is non-working for punch code {work_type}: {reason}")
                predictions[work_type] = 0
                hours_predictions[work_type] = 0
                continue


            # Try neural network prediction first if requested
            if use_neural_network and nn_models and work_type in nn_models:
                nn_prediction = predict_with_neural_network(df, nn_models, nn_scalers, work_type, date)
                
                if nn_prediction is not None:
                    predictions[work_type] = nn_prediction
                    hours_predictions[work_type] = calculate_hours_prediction(df, work_type, nn_prediction, next_date)
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
            for lag in LAG_DAYS:
                try:
                    lag_date = latest_date - timedelta(days=lag)
                    lag_records = work_type_data[work_type_data['Date'] == lag_date]
                    lag_value = lag_records['NoOfMan'].sum() if not lag_records.empty else 0
                    lag_features[f'NoOfMan_lag_{lag}'] = lag_value
                except:
                    # If a lag day isn't available
                    lag_features[f'NoOfMan_lag_{lag}'] = 0
            
            # Calculate rolling statistics
            for window in ROLLING_WINDOWS:
                try:
                    recent_data = work_type_data[work_type_data['Date'] > latest_date - timedelta(days=window)]
                    values = recent_data['NoOfMan'].values
                    
                    # Calculate rolling mean
                    rolling_mean = values.mean() if len(values) > 0 else 0
                    lag_features[f'NoOfMan_rolling_mean_{window}'] = rolling_mean
                    
                    # Calculate other rolling statistics if needed by the model
                    if ROLLING_WINDOWS and window == ROLLING_WINDOWS[0]:  # Only for first window to avoid feature explosion
                        lag_features[f'NoOfMan_rolling_max_{window}'] = values.max() if len(values) > 0 else 0
                        lag_features[f'NoOfMan_rolling_min_{window}'] = values.min() if len(values) > 0 else 0
                        lag_features[f'NoOfMan_rolling_std_{window}'] = values.std() if len(values) > 1 else 0
                except Exception as roll_err:
                    logger.warning(f"Error calculating rolling stats for window {window}: {str(roll_err)}")
                    lag_features[f'NoOfMan_rolling_mean_{window}'] = 0
                    if ROLLING_WINDOWS and window == ROLLING_WINDOWS[0]:
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
                if 1 in LAG_DAYS and 7 in LAG_DAYS:
                    lag_features['NoOfMan_7day_trend'] = lag_features['NoOfMan_lag_1'] - lag_features['NoOfMan_lag_7']
                if 1 in LAG_DAYS and 2 in LAG_DAYS:
                    lag_features['NoOfMan_1day_trend'] = lag_features['NoOfMan_lag_1'] - lag_features['NoOfMan_lag_2']
            except:
                if 1 in LAG_DAYS and 7 in LAG_DAYS:
                    lag_features['NoOfMan_7day_trend'] = 0
                if 1 in LAG_DAYS and 2 in LAG_DAYS:
                    lag_features['NoOfMan_1day_trend'] = 0
            
            # Combine features
            all_features = {**next_day_features, **lag_features}
            
            # Get required features for the model
            model = models[work_type]
            required_features = _get_required_features(model)
            
            # Create the prediction dataframe with required features
            X_pred = pd.DataFrame([{
                feature: all_features.get(feature, 0) 
                for feature in required_features
            }])
            
            # Make prediction
            try:
                # X_pred already contains all required features from _get_required_features()
                # No need to add additional features - just predict directly
                prediction = model.predict(X_pred)[0]
                
                # Ensure prediction is not negative
                prediction = max(0, prediction)
                
                predictions[work_type] = prediction
                hours_predictions[work_type] = calculate_hours_prediction(df, work_type, prediction, next_date)
                
                logger.info(f"RandomForest predicted {prediction:.2f} workers for WorkType {work_type}")
                
            except Exception as e:
                logger.error(f"Error predicting for WorkType {work_type}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Return the date, predictions dictionary, and hours dictionary
        return next_date, predictions, hours_predictions
    
    except Exception as e:
        logger.error(f"Error predicting next day: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to predict next day: {str(e)}")
        
def predict_multiple_days(df, models, num_days=7, use_neural_network=False):
    """
    Predict NoOfMan and Hours for multiple days for each WorkType
    """
    try:
        logger.info(f"Predicting for the next {num_days} days")
        
        # Initialize results dictionaries and holiday info dictionary
        multi_day_predictions = {}
        multi_day_hours_predictions = {}
        nonworking_info = {}
        
        # Create a working copy of the dataframe that we'll extend with predictions
        current_df = df.copy()
        
        # Find the latest date in the dataset
        latest_date = current_df['Date'].max()
        
        # Predict for each day
        for i in range(num_days):
            prediction_date = latest_date + timedelta(days=i+1)  # +1 to start with the next day
            
            try:
                # Use the same prediction logic but now it handles punch-code specific rules
                next_date, predictions, hours_predictions = predict_next_day(
                    current_df, 
                    models, 
                    date=latest_date + timedelta(days=i), 
                    use_neural_network=use_neural_network
                )
                
                # Debug log to see what predictions are being generated
                logger.info(f"Generated predictions for {next_date}: {predictions}")
                
                # Store predictions
                multi_day_predictions[next_date] = predictions
                multi_day_hours_predictions[next_date] = hours_predictions
                
                # Check which punch codes are not working on this day for info
                non_working_codes = []
                for work_type in models.keys():
                    is_working, reason = is_working_day_for_punch_code(next_date, work_type)
                    if not is_working:
                        non_working_codes.append(f"{work_type}: {reason}")
                
                if non_working_codes:
                    nonworking_info[next_date] = "; ".join(non_working_codes)
                
                # Add the predictions back to the dataframe for the next iteration
                new_rows = []
                for work_type, pred_value in predictions.items():
                    hours_value = hours_predictions.get(work_type, pred_value * 8.0)  # Default to 8 hours if missing
                    
                    # Use actual prediction values (including zeros for non-working punch codes)
                    new_row = {
                        'Date': next_date,
                        'WorkType': work_type,
                        'NoOfMan': pred_value,
                        'Hours': hours_value,
                        
                        # Add the date features
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
            except Exception as day_error:
                # Handle errors for individual days without failing the entire prediction
                logger.error(f"Error processing day {prediction_date}: {str(day_error)}")
                # Continue with the next date
                continue
        
        logger.info(f"Predictions completed for {num_days} days")
        return multi_day_predictions, multi_day_hours_predictions, nonworking_info
    
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
    """Helper function to get feature names required by the model - now uses pre-computed lists"""
    try:
        # If it's a pipeline, get the feature names from the model
        if hasattr(model, 'steps'):
            # Try to get feature names directly from the model (e.g., RandomForest, etc.)
            if hasattr(model.named_steps['model'], 'feature_names_in_'):
                return list(model.named_steps['model'].feature_names_in_)
            
            # Otherwise, extract from preprocessor
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                cat_cols = []

                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'cat' and hasattr(transformer, 'categories_'):
                            for i, _ in enumerate(transformer.categories_):
                                if i < len(cols):
                                    cat_cols.append(cols[i])
                
                # Use pre-computed remainder columns
                return cat_cols + _FEATURE_CACHE.get('remainder', [])

        # Use pre-computed default features
        return _FEATURE_CACHE.get('default', [])

    except Exception as e:
        logger.error(f"Error getting required features: {str(e)}")
        logger.error(traceback.format_exc())
        # Use pre-computed fallback features
        return _FEATURE_CACHE.get('fallback', [])
