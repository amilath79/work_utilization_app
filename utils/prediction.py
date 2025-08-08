"""
Prediction utilities for work utilization forecasting with multiple model types.
ENHANCED VERSION with critical accuracy fixes based on multy (2).py analysis.
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
from utils.feature_selection import FeatureSelector
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


# def create_prediction_row_enhanced(work_data, next_date, work_type):
#     """
#     Create prediction row with better feature estimation like multy (2).py
#     """
#     try:
#         # Get same day of week data for better estimation
#         same_dow_data = work_data[work_data['Date'].dt.dayofweek == next_date.weekday()]
        
#         if len(same_dow_data) >= 3:
#             # Use same day of week patterns (better estimation)
#             quantity = same_dow_data['Quantity'].median()
#             system_hours = same_dow_data['SystemHours'].median()
#         else:
#             # Fallback to recent data
#             recent_data = work_data.tail(7)
#             quantity = recent_data['Quantity'].median()
#             system_hours = recent_data['SystemHours'].median()
        
#         # Handle SystemKPI if present
#         if 'SystemKPI' in work_data.columns:
#             if len(same_dow_data) >= 3:
#                 system_kpi = same_dow_data['SystemKPI'].median()
#             else:
#                 system_kpi = work_data['SystemKPI'].median()
#         else:
#             system_kpi = 1.0
        
#         # Create prediction row
#         pred_row = pd.DataFrame([{
#             'Date': next_date,
#             'WorkType': work_type,
#             'Quantity': quantity,
#             'SystemHours': system_hours,
#             'SystemKPI': system_kpi,
#             'Hours': work_data['Hours'].iloc[-1]  # Placeholder for lag calculation
#         }])
        
#         return pred_row
        
#     except Exception as e:
#         logger.error(f"Error creating enhanced prediction row: {str(e)}")
#         # Fallback to simple approach
#         return pd.DataFrame([{
#             'Date': next_date,
#             'WorkType': work_type,
#             'Quantity': work_data['Quantity'].mean(),
#             'SystemHours': work_data['SystemHours'].mean(),
#             'SystemKPI': work_data['SystemKPI'].mean() if 'SystemKPI' in work_data.columns else 1.0,
#             'Hours': work_data['Hours'].iloc[-1]
#         }])

def create_prediction_row_enhanced(work_data, next_date, work_type):
    """Enhanced prediction using manual forecaster hierarchy"""
    
    # STRATEGY 1: Exact same day last year (HIGHEST PRIORITY)
    last_year_date = next_date.replace(year=next_date.year - 1)
    exact_same_day = work_data[work_data['Date'] == last_year_date]
    
    if not exact_same_day.empty:
        # Use exact same day last year
        base_quantity = exact_same_day['Quantity'].iloc[0]
        base_hours = exact_same_day['Hours'].iloc[0] if 'Hours' in exact_same_day.columns else 0
        system_hours = exact_same_day['SystemHours'].iloc[0] if 'SystemHours' in exact_same_day.columns else 8.0
        
        logger.info(f"Using exact same day last year: {last_year_date} for {work_type}")
        
    else:
        # STRATEGY 2: Same week same day last year
        week_start = last_year_date - timedelta(days=3)
        week_end = last_year_date + timedelta(days=3)
        
        same_week_last_year = work_data[
            (work_data['Date'] >= week_start) & 
            (work_data['Date'] <= week_end) &
            (work_data['Date'].dt.dayofweek == next_date.weekday())
        ]
        
        if not same_week_last_year.empty:
            base_quantity = same_week_last_year['Quantity'].median()
            base_hours = same_week_last_year['Hours'].median()
            system_hours = same_week_last_year['SystemHours'].median()
            
            logger.info(f"Using same week last year for {work_type}")
        else:
            # STRATEGY 3: Fallback to current approach
            same_dow_data = work_data[work_data['Date'].dt.dayofweek == next_date.weekday()]
            
            if len(same_dow_data) >= 3:
                base_quantity = same_dow_data['Quantity'].median()
                base_hours = same_dow_data['Hours'].median()
                system_hours = same_dow_data['SystemHours'].median()
            else:
                recent_data = work_data.tail(7)
                base_quantity = recent_data['Quantity'].median()
                base_hours = recent_data['Hours'].median()
                system_hours = recent_data['SystemHours'].median()
    
    # Apply year-over-year growth adjustment
    growth_rate = calculate_yoy_growth_rate(work_data, work_type, next_date)
    adjusted_quantity = base_quantity * (1 + growth_rate)
    
    return pd.DataFrame([{
        'Date': next_date,
        'WorkType': work_type,
        'Quantity': adjusted_quantity,
        'SystemHours': system_hours,
        'Hours': base_hours * (1 + growth_rate),  # Apply growth to hours too
        'BaseYear': last_year_date.year,
        'GrowthRate': growth_rate
    }])

def calculate_yoy_growth_rate(work_data, work_type, target_date):
    """Calculate year-over-year growth rate"""
    try:
        # Get same period last year
        last_year_start = target_date.replace(year=target_date.year - 1) - timedelta(days=30)
        last_year_end = target_date.replace(year=target_date.year - 1) + timedelta(days=30)
        
        # Get same period this year
        this_year_start = target_date - timedelta(days=30)
        this_year_end = target_date + timedelta(days=30)
        
        last_year_avg = work_data[
            (work_data['Date'] >= last_year_start) & 
            (work_data['Date'] <= last_year_end)
        ]['Hours'].mean()
        
        this_year_avg = work_data[
            (work_data['Date'] >= this_year_start) & 
            (work_data['Date'] <= this_year_end)
        ]['Hours'].mean()
        
        if last_year_avg > 0 and not pd.isna(this_year_avg) and not pd.isna(last_year_avg):
            growth_rate = (this_year_avg - last_year_avg) / last_year_avg
            # Cap growth rate between -50% and +50%
            return max(-0.5, min(0.5, growth_rate))
        else:
            return 0.0
            
    except Exception as e:
        logger.warning(f"Could not calculate growth rate: {e}")
        return 0.0
    
    
def apply_prediction_bounds(hours_pred, work_data, next_date):
    """
    Apply intelligent bounds similar to multy (2).py approach
    """
    try:
        # Get same day of week historical data for better bounds
        same_dow_data = work_data[work_data['Date'].dt.dayofweek == next_date.weekday()]
        
        if len(same_dow_data) >= 8:
            bounds_data = same_dow_data.tail(8)
        else:
            bounds_data = work_data.tail(30)
        
        # Calculate bounds using tighter quantiles
        historical_min = bounds_data['Hours'].quantile(0.05)
        historical_max = bounds_data['Hours'].quantile(0.95)
        
        # Apply bounds with some flexibility (similar to multy approach)
        hours_pred = np.clip(hours_pred, historical_min * 0.6, historical_max * 1.4)
        
        # Ensure positive
        hours_pred = max(0, hours_pred)
        
        # Additional sanity check - prevent extreme values
        historical_mean = bounds_data['Hours'].mean()
        if hours_pred > historical_mean * 3:
            hours_pred = historical_mean * 1.5
        
        return hours_pred
        
    except Exception as e:
        logger.error(f"Error applying prediction bounds: {str(e)}")
        # Fallback to simple bounds
        return max(0, hours_pred)
        
        
def predict_next_day(df, models, date=None, use_neural_network=False):
    """
    Predict next day using COMPLETE PIPELINES with PROPER temporal consistency
    ENHANCED VERSION with critical fixes for accuracy
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
                if not is_working_day_for_punch_code(next_date, work_type):
                    predictions[work_type] = 0
                    hours_predictions[work_type] = 0
                    logger.info(f"📅 {work_type}: Non-working day")
                    continue

                work_data = df[df['WorkType'] == work_type].copy()
                work_data = work_data.sort_values('Date')

                if len(work_data) < 30:
                    logger.warning(f"Insufficient data for {work_type}")
                    predictions[work_type] = 0
                    hours_predictions[work_type] = 0
                    continue

                # ENHANCED: Better prediction row creation
                pred_row = create_prediction_row_enhanced(work_data, next_date, work_type)

                # Combine with historical data for lag/rolling calculation
                combined_data = pd.concat([work_data, pred_row], ignore_index=True)
                combined_data = combined_data.sort_values('Date')

                # ✅ CRITICAL FIX: Use complete pipeline properly
                try:
                    # Apply complete pipeline in one go (preferred method)
                    prediction_input = combined_data.tail(1).drop(columns=['Hours'], errors='ignore')
                    hours_pred_log = pipeline.predict(prediction_input)[0]
                    
                except Exception as pipeline_error:
                    logger.debug(f"Direct pipeline prediction failed for {work_type}, using step-by-step: {str(pipeline_error)}")
                    # Fallback to step-by-step approach
                    # ✅ CRITICAL FIX: Use transform (not fit_transform) on already fitted pipeline
                    X_all = pipeline.named_steps['feature_engineering'].transform(combined_data)
                    X_selected = pipeline.named_steps['feature_selection'].transform(X_all)
                    hours_pred_log = pipeline.named_steps['model'].predict(X_selected[-1:])[0]

                # ✅ CRITICAL FIX: Apply inverse log transformation
                hours_pred = np.expm1(hours_pred_log)

                # Apply intelligent bounds (enhanced version)
                hours_pred = apply_prediction_bounds(hours_pred, work_data, next_date)

                workers_pred = max(0, round(hours_pred / 8.0)) if hours_pred > 0 else 0
                predictions[work_type] = workers_pred  # Number of workers
                hours_predictions[work_type] = hours_pred  # Hours

                logger.info(f"✅ {work_type}: {hours_pred:.1f} hours (Historical avg: {work_data['Hours'].mean():.1f})")

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
    Multi-day prediction using complete pipelines with RECURSIVE updates
    ENHANCED VERSION maintaining temporal consistency like multy (2).py
    """
    try:
        all_predictions = {}
        all_hours = {}
        current_df = df.copy()
        
        current_date = pd.to_datetime(start_date) if start_date else df['Date'].max()
        
        logger.info(f"🚀 Starting multi-day prediction from {current_date} for {num_days} days")
        
        for i in range(num_days):
            pred_date = current_date + timedelta(days=i+1)
            
            # Use updated predict_next_day with current dataframe (includes previous predictions)
            _, day_preds, day_hours = predict_next_day(
                current_df, models, date=pred_date
            )
            
            all_predictions[pred_date] = day_preds
            all_hours[pred_date] = day_hours
            
            # ✅ CRITICAL: Add predictions back to dataframe for next iteration (like multy (2).py)
            for work_type, hours_value in day_hours.items():
                if hours_value > 0:  # Only add working days
                    # Get features for the new row - use same approach as create_prediction_row_enhanced
                    work_data = current_df[current_df['WorkType'] == work_type]
                    
                    # Get last week's same day for Quantity and SystemHours
                    last_week_date = pred_date - timedelta(days=7)
                    last_week_row = current_df[
                        (current_df['WorkType'] == work_type) &
                        (current_df['Date'] == last_week_date)
                    ]
                    
                    if not last_week_row.empty:
                        quantity = last_week_row['Quantity'].values[0]
                        system_hours = last_week_row['SystemHours'].values[0]
                        system_kpi = last_week_row['SystemKPI'].values[0] if 'SystemKPI' in last_week_row.columns else 1.0
                    else:
                        # Fallback to same day of week pattern
                        same_dow_data = work_data[work_data['Date'].dt.dayofweek == pred_date.weekday()]
                        if len(same_dow_data) >= 3:
                            quantity = same_dow_data['Quantity'].median()
                            system_hours = same_dow_data['SystemHours'].median()
                            system_kpi = same_dow_data['SystemKPI'].median() if 'SystemKPI' in same_dow_data.columns else 1.0
                        else:
                            # Final fallback to recent mean
                            recent = work_data.tail(7)
                            quantity = recent['Quantity'].mean()
                            system_hours = recent['SystemHours'].mean()
                            system_kpi = recent['SystemKPI'].mean() if 'SystemKPI' in recent.columns else 1.0

                    # Create new row with predicted Hours
                    new_row = pd.DataFrame([{
                        'Date': pred_date,
                        'WorkType': work_type,
                        'Hours': hours_value,  # ✅ Use predicted Hours value
                        'NoOfMan': max(0, round(hours_value / 8.0)) if hours_value > 0 else 0,  # Add NoOfMan calculation
                        'Quantity': quantity,
                        'SystemHours': system_hours,
                        'SystemKPI': system_kpi
                    }])
                    
                    # Add to current dataframe for next iteration
                    current_df = pd.concat([current_df, new_row], ignore_index=True)
                    
            logger.info(f"📅 Completed predictions for {pred_date.strftime('%Y-%m-%d')}")
        
        logger.info(f"✅ Multi-day prediction completed for {num_days} days")
        return all_predictions, all_hours, {}
        
    except Exception as e:
        logger.error(f"Error in multi-day prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, {}


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
        
        # Round to whole number (minimum 1 if hours > 0)
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