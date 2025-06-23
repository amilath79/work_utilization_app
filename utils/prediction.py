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
    MODELS_DIR, DATA_DIR, CHUNK_SIZE, DEFAULT_MODEL_PARAMS,
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, SQL_USERNAME, SQL_PASSWORD,
    FEATURE_GROUPS, PRODUCTIVITY_FEATURES, DATE_FEATURES, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS, ENHANCED_WORK_TYPES
)

# Configure logger
logger = logging.getLogger(__name__)

# Import holiday utils
from utils.holiday_utils import is_swedish_holiday

def get_required_features():
    """Get required features based on config - simple and direct"""
    numeric_features = []
    categorical_features = []
    
    # Essential lag features - NOW USING HOURS
    if FEATURE_GROUPS['LAG_FEATURES']:
        for lag in ESSENTIAL_LAGS:
            numeric_features.append(f'Hours_lag_{lag}')  # CHANGED
        if FEATURE_GROUPS['PRODUCTIVITY_FEATURES']:
            numeric_features.append('Quantity_lag_1')
    
    # Essential rolling features - NOW USING HOURS  
    if FEATURE_GROUPS['ROLLING_FEATURES']:
        for window in ESSENTIAL_WINDOWS:
            numeric_features.append(f'Hours_rolling_mean_{window}')  # CHANGED
    
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
    
# def create_prediction_features(df, work_type, next_date, latest_date):
#     """
#     Create prediction features using EnhancedFeatureTransformer
#     """
#     try:
#         # Process historical data ONLY
#         work_type_data = df[df['WorkType'] == work_type].copy()
        
#         # Initialize and use the transformer
#         feature_transformer = EnhancedFeatureTransformer()
        
#         # Fit and transform the data
#         feature_transformer.fit(work_type_data)
#         features_df = feature_transformer.transform(work_type_data)
        
#         # Get latest row and update date features only
#         latest_features = features_df.iloc[-1:].copy()
#         latest_features['Date'] = next_date
#         latest_features['DayOfWeek_feat'] = next_date.weekday()
#         latest_features['Month_feat'] = next_date.month
#         latest_features['IsWeekend_feat'] = 1 if next_date.weekday() == 5 else 0
#         latest_features['DayOfMonth'] = next_date.day
#         latest_features['Quarter'] = (next_date.month - 1) // 3 + 1
        
#         return latest_features
        
#     except Exception as e:
#         logger.error(f"Error creating prediction features: {str(e)}")
#         return None
    
        
def predict_next_day(df, models, date=None, use_neural_network=False):
    """
    Predict next day using COMPLETE PIPELINES with PROPER feature estimation
    """
    try:
        # Determine prediction date
        if date is None:
            latest_date = df['Date'].max()
            next_date = latest_date + timedelta(days=1)
        else:
            next_date = pd.to_datetime(date)

        logger.info(f"🎯 Predicting for {next_date.strftime('%Y-%m-%d')}")

        predictions = {}
        hours_predictions = {}

        for work_type, pipeline in models.items():
            try:
                # Check if working day
                if not is_working_day_for_punch_code(next_date, work_type):
                    predictions[work_type] = 0
                    hours_predictions[work_type] = 0
                    logger.info(f"📅 {work_type}: Non-working day")
                    continue

                # Get historical data for this work type
                work_data = df[df['WorkType'] == work_type].copy()
                work_data = work_data.sort_values('Date')

                if len(work_data) < 30:
                    logger.warning(f"Insufficient data for {work_type}")
                    predictions[work_type] = 0
                    hours_predictions[work_type] = 0
                    continue

                # CRITICAL FIX: Estimate features based on historical patterns
                # Get day of week for prediction
                pred_dow = next_date.dayofweek
                
                # Find similar days in history (same day of week)
                similar_days = work_data[
                    pd.to_datetime(work_data['Date']).dt.dayofweek == pred_dow
                ].tail(8)  # Last 8 similar weekdays
                
                if len(similar_days) < 3:
                    # Fallback to last 30 days if not enough similar days
                    similar_days = work_data.tail(30)
                
                # Calculate averages from similar days
                avg_quantity = similar_days['Quantity'].mean()
                avg_system_hours = similar_days['SystemHours'].mean()
                avg_hours = similar_days['Hours'].mean()
                
                # Calculate typical ratios
                if avg_hours > 0:
                    quantity_per_hour = avg_quantity / avg_hours
                    system_hours_ratio = avg_system_hours / avg_hours
                else:
                    quantity_per_hour = 10  # Default
                    system_hours_ratio = 1   # Default
                
                # Account for recent trends (optional)
                recent_7_days = work_data.tail(7)
                recent_avg_hours = recent_7_days['Hours'].mean()
                
                # Blend historical average with recent trend
                trend_weight = 0.3  # 30% recent trend, 70% historical pattern
                expected_hours = (avg_hours * 0.7) + (recent_avg_hours * 0.3)
                
                # Create prediction row WITHOUT actual Hours!
                pred_row = pd.DataFrame([{
                    'Date': next_date,
                    'WorkType': work_type,
                    # DON'T include Hours - that's what we're predicting!
                    'Quantity': avg_quantity,
                    'SystemHours': avg_system_hours,
                    'SystemKPI': avg_quantity / avg_system_hours if avg_system_hours > 0 else 1.0
                }])

                # Combine with historical data for lag feature calculation
                combined_data = pd.concat([work_data, pred_row], ignore_index=True)
                combined_data = combined_data.sort_values('Date')

                # The pipeline will transform this data
                # It will create lag features from historical data
                # But won't have Hours lag features for the prediction row (correct!)
                
                # Make prediction
                hours_pred = pipeline.predict(pred_row)[0]  # Predict only the new row
                
                # Apply sanity checks
                # Ensure prediction is within reasonable range
                historical_min = similar_days['Hours'].quantile(0.1)
                historical_max = similar_days['Hours'].quantile(0.9)
                
                hours_pred = np.clip(hours_pred, 
                                   historical_min * 0.5,  # Allow some flexibility
                                   historical_max * 1.5)
                
                hours_pred = max(0, hours_pred)  # Never negative
                
                predictions[work_type] = hours_pred
                hours_predictions[work_type] = hours_pred
                
                logger.info(f"✅ {work_type}: {hours_pred:.1f} hours (Historical avg: {avg_hours:.1f})")
                
            except Exception as e:
                logger.error(f"Error predicting {work_type}: {str(e)}")
                logger.error(f"Details: {traceback.format_exc()}")
                predictions[work_type] = 0
                hours_predictions[work_type] = 0
        
        return next_date, predictions, hours_predictions
        
    except Exception as e:
        logger.error(f"Error in predict_next_day: {str(e)}")
        return None, {}, {}



    
def predict_multiple_days(df, models, start_date, num_days, use_neural_network=False):
    """
    Multi-day prediction using complete pipelines
    """
    try:
        all_predictions = {}
        all_hours = {}
        current_df = df.copy()
        
        current_date = pd.to_datetime(start_date) if start_date else df['Date'].max()
        
        for i in range(num_days):
            pred_date = current_date + timedelta(days=i+1)
            
            # Use updated predict_next_day
            _, day_preds, day_hours = predict_next_day(
                current_df, models, date=pred_date
            )
            
            all_predictions[pred_date] = day_preds
            all_hours[pred_date] = day_hours
            
            # Add predictions back to dataframe for next iteration
            for work_type, hours_value in day_hours.items():
                if hours_value > 0:  # Only add working days
                    new_row = pd.DataFrame([{
                        'Date': pred_date,
                        'WorkType': work_type,
                        'Hours': hours_value,
                        'Quantity': 100,  # Default
                        'SystemHours': hours_value
                    }])
                    current_df = pd.concat([current_df, new_row], ignore_index=True)
            
            current_df = current_df.sort_values(['WorkType', 'Date'])
        
        return all_predictions, all_hours, {}
        
    except Exception as e:
        logger.error(f"Error in multi-day prediction: {str(e)}")
        return {}, {}, {}


# def _get_required_features(model):
#     """Helper function to get feature names required by the model - config-driven"""
#     try:
#         # If it's a pipeline, try to get feature names from the model
#         if hasattr(model, 'steps'):
#             if hasattr(model.named_steps['model'], 'feature_names_in_'):
#                 return list(model.named_steps['model'].feature_names_in_)
            
#             # Extract from preprocessor
#             preprocessor = model.named_steps.get('preprocessor')
#             if preprocessor and hasattr(preprocessor, 'transformers_'):
#                 feature_names = []
#                 for name, transformer, cols in preprocessor.transformers_:
#                     feature_names.extend(cols)
                
#                 if feature_names:
#                     return feature_names
        
#         # Fallback: Use same config-driven features as training
#         numeric_features, categorical_features = get_required_features()
#         all_features = categorical_features + numeric_features
        
#         active_groups = [group for group, enabled in FEATURE_GROUPS.items() if enabled]
#         logger.info(f"Using config features from {active_groups}: {len(all_features)} total features")
        
#         return all_features

#     except Exception as e:
#         logger.error(f"Error getting required features: {str(e)}")
#         # Simple fallback
#         return ['DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat', 'NoOfMan_lag_1', 'NoOfMan_lag_7', 'NoOfMan_rolling_mean_7']    

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
    
  
    

def predict_hours_and_calculate_noof_man(df, models, work_type, date=None):
    """
    Predict Hours and calculate NoOfMan from it
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with features
    models : dict
        Trained models
    work_type : str
        Work type to predict
    date : datetime, optional
        Date for prediction
        
    Returns:
    --------
    dict
        Dictionary with Hours prediction and calculated NoOfMan
    """
    try:
        # Get Hours prediction using existing model
        hours_prediction = predict_single(df, models, work_type, date)
        
        # Convert to NoOfMan using business rule
        noof_man_prediction = calculate_noof_man_from_hours(hours_prediction)
        
        return {
            'Hours': hours_prediction,
            'NoOfMan': noof_man_prediction,
            'work_type': work_type,
            'date': date
        }
        
    except Exception as e:
        logger.error(f"Error predicting hours and NoOfMan: {str(e)}")
        return {
            'Hours': 0,
            'NoOfMan': 0,
            'work_type': work_type,
            'date': date
        }
    

def calculate_noof_man_from_hours(hours_prediction, punch_code=None):
    """
    Simple function to convert Hours to NoOfMan using configurable business rule
    
    Parameters:
    -----------
    hours_prediction : float
        Predicted total hours
    punch_code : str, optional
        Punch code for specific rules
        
    Returns:
    --------
    int
        Number of workers needed
    """
    try:
        from config import DEFAULT_HOURS_PER_WORKER, PUNCH_CODE_HOURS_PER_WORKER
        
        if hours_prediction <= 0:
            return 0
        
        # Get hours per worker (punch code specific or default)
        if punch_code and str(punch_code) in PUNCH_CODE_HOURS_PER_WORKER:
            hours_per_worker = PUNCH_CODE_HOURS_PER_WORKER[str(punch_code)]
        else:
            hours_per_worker = DEFAULT_HOURS_PER_WORKER
        
        # Calculate NoOfMan
        noof_man = hours_prediction / hours_per_worker
        
        # Round to whole number (minimum 1 if hours > 0)f
        if noof_man > 0 and noof_man < 1:
            return 1
        else:
            return max(0, round(noof_man))
            
    except Exception as e:
        logger.error(f"Error calculating NoOfMan: {str(e)}")
        # Safe fallback: simple division by 8
        return max(0, round(hours_prediction / 8.0)) if hours_prediction > 0 else 0
    

def predict_single(df, models, work_type, date=None):
    """
    Predict Hours for a single work type on a single date
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    models : dict
        Trained models
    work_type : str
        Work type to predict
    date : datetime, optional
        Date for prediction
        
    Returns:
    --------
    float
        Predicted Hours for the work type
    """
    try:
        # Use existing predict_next_day function
        pred_date, predictions, hours_predictions = predict_next_day(df, models, date)
        
        # Extract the specific work type prediction
        if work_type in predictions:
            # Since models predict Hours directly, return that
            hours_pred = predictions.get(work_type, 0)
            return hours_pred
        else:
            logger.warning(f"No prediction available for work type {work_type}")
            return 0
            
    except Exception as e:
        logger.error(f"Error in predict_single for {work_type}: {str(e)}")
        return 0