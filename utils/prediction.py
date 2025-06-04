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
    import torch
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
    from utils.feature_builder import FeatureBuilder
    from config import FEATURE_TIERS
    
    feature_cache = {}
    
    # Create a feature builder with reasonable defaults for precomputation
    # We'll use conservative estimates since we don't have actual data at module load
    feature_builder = FeatureBuilder(
        data_columns=['Date', 'WorkType', 'NoOfMan', 'Hours', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI'],
        data_length=1000  # Assume sufficient data for advanced features
    )
    
    # Get comprehensive feature lists based on tier configuration
    numeric_features, categorical_features = feature_builder.get_feature_lists()
    
    # Cache different combinations
    feature_cache['remainder'] = numeric_features
    feature_cache['default'] = categorical_features + numeric_features
    feature_cache['categorical'] = categorical_features
    feature_cache['numeric'] = numeric_features
    
    # Fallback features (basic tier only) - use config-based basic features
    fallback_features = ['DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat']
    
    # Add basic lag features from config
    if FEATURE_TIERS['BASIC']:
        for lag in [1, 7]:  # Basic lags from config BASIC_FEATURES
            fallback_features.append(f'NoOfMan_lag_{lag}')
        fallback_features.append('NoOfMan_rolling_mean_7')
    
    feature_cache['fallback'] = fallback_features
    
    # Log feature summary with tier information
    active_tiers = [tier for tier, enabled in FEATURE_TIERS.items() if enabled]
    logger.info(f"Feature cache built with tiers {active_tiers}: {len(numeric_features)} numeric, {len(categorical_features)} categorical features")
    
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
    Make prediction using the PyTorch neural network model with tiered features
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
        
        # Use tiered feature system to get appropriate features
        from utils.feature_builder import FeatureBuilder
        from config import FEATURE_TIERS, LAG_DAYS, ROLLING_WINDOWS
        
        # Create feature builder for this data
        feature_builder = FeatureBuilder(
            data_columns=list(recent_data.columns),
            data_length=len(recent_data)
        )
        
        # Get features based on tier configuration
        numeric_features, categorical_features = feature_builder.get_feature_lists()
        
        # Base features for neural network
        base_features = ['NoOfMan', 'DayOfWeek_feat', 'Month_feat', 'IsWeekend_feat']
        
        # Add lag features based on active tiers
        lag_features = []
        if FEATURE_TIERS['ADVANCED']:
            for lag in LAG_DAYS:
                lag_features.append(f'NoOfMan_lag_{lag}')
        elif FEATURE_TIERS['INTERMEDIATE']:
            for lag in [1, 2, 3, 7, 14, 30]:
                lag_features.append(f'NoOfMan_lag_{lag}')
        else:  # Basic
            for lag in [1, 7]:
                lag_features.append(f'NoOfMan_lag_{lag}')
        
        # Add rolling features based on active tiers
        rolling_features = []
        if FEATURE_TIERS['ADVANCED']:
            for window in ROLLING_WINDOWS:
                rolling_features.append(f'NoOfMan_rolling_mean_{window}')
        elif FEATURE_TIERS['INTERMEDIATE']:
            for window in [7, 14, 30]:
                rolling_features.append(f'NoOfMan_rolling_mean_{window}')
        else:  # Basic
            rolling_features.append('NoOfMan_rolling_mean_7')
        
        # Combine all features
        all_neural_features = base_features + lag_features + rolling_features
        
        # Filter to only include features that actually exist in the data
        available_features = [f for f in all_neural_features if f in recent_data.columns]
        
        # ✅ VALIDATION: Check if we have minimum required features
        if len(available_features) < 4:  # At minimum need NoOfMan + 3 date features
            logger.warning(f"Not enough features available for neural prediction for WorkType: {work_type}. "
                         f"Available: {available_features}")
            return None
        
        # ✅ LOG WHAT FEATURES ARE BEING USED
        active_tiers = [tier for tier, enabled in FEATURE_TIERS.items() if enabled]
        logger.info(f"Neural network using {len(available_features)} features from tiers {active_tiers} for {work_type}: {available_features}")
        
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
    Create a properly engineered feature set for prediction using the tiered feature system
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data
    work_type : str
        Work type to predict for
    next_date : datetime
        Date to predict for
    latest_date : datetime
        Latest date in historical data
        
    Returns:
    --------
    pd.DataFrame
        Single row dataframe with all required features
    """
    try:
        from utils.feature_engineering import engineer_features, create_lag_features
        from utils.feature_builder import FeatureBuilder
        
        # Filter data for this work type
        work_type_data = df[df['WorkType'] == work_type].copy()
        
        # Create a mock prediction row with the next date
        prediction_row = {
            'Date': next_date,
            'WorkType': work_type,
            'NoOfMan': 0,  # Placeholder - we're predicting this
        }
        
        # Add any other columns that might be in the original data for productivity features
        productivity_columns = ['Hours', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI']
        for col in productivity_columns:
            if col in df.columns:
                # Use the mean value from recent data as placeholder
                recent_data = work_type_data.tail(7)  # Last 7 days
                mean_value = recent_data[col].mean() if not recent_data.empty else 0
                prediction_row[col] = mean_value if not pd.isna(mean_value) else 0
        
        # Create a temporary dataframe with historical data + prediction row
        temp_df = pd.concat([work_type_data, pd.DataFrame([prediction_row])], ignore_index=True)
        
        # Run through the same feature engineering pipeline as training
        engineered_df = engineer_features(temp_df)
        
        # Use tiered feature system to determine appropriate lag days and rolling windows
        feature_builder = FeatureBuilder(
            data_columns=list(engineered_df.columns),
            data_length=len(engineered_df)
        )
        
        # Get the appropriate feature configuration based on available data
        numeric_features, categorical_features = feature_builder.get_feature_lists()
        
        # Determine lag days and rolling windows based on feature tiers enabled
        from config import FEATURE_TIERS, LAG_DAYS, ROLLING_WINDOWS
        
        if FEATURE_TIERS['ADVANCED']:
            lag_days_to_use = LAG_DAYS
            rolling_windows_to_use = ROLLING_WINDOWS
        elif FEATURE_TIERS['INTERMEDIATE']:
            # Use intermediate configuration - defined in config
            lag_days_to_use = [1, 2, 3, 7, 14, 30]
            rolling_windows_to_use = [7, 14, 30]
        else:
            # Use basic configuration
            lag_days_to_use = [1, 7]
            rolling_windows_to_use = [7]
        
        # Create lag features with appropriate configuration
        features_df = create_lag_features(
            engineered_df, 
            'WorkType', 
            'NoOfMan',
            lag_days=lag_days_to_use,
            rolling_windows=rolling_windows_to_use
        )
        
        # Get the last row (our prediction row) with all engineered features
        prediction_features = features_df.iloc[-1:].copy()
        
        # Log which feature tier is being used
        active_tiers = [tier for tier, enabled in FEATURE_TIERS.items() if enabled]
        logger.info(f"Using feature tiers: {active_tiers} for prediction of {work_type}")
        
        return prediction_features
        
    except Exception as e:
        logger.error(f"Error creating prediction features: {str(e)}")
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
            
            # Fall back to traditional model - USE PROPER FEATURE ENGINEERING
            try:
                # Create properly engineered features using the same pipeline as training
                prediction_features = create_prediction_features(df, work_type, next_date, latest_date)
                
                if prediction_features is None:
                    logger.warning(f"Could not create features for WorkType {work_type}. Skipping.")
                    continue
                
                # Get the model
                model = models[work_type]
                
                # Get required features for the model
                required_features = _get_required_features(model)
                
                # Filter to only include features that exist in our engineered features
                available_features = [f for f in required_features if f in prediction_features.columns]
                
                if len(available_features) == 0:
                    logger.warning(f"No required features available for WorkType {work_type}. Skipping.")
                    continue
                
                # Create prediction dataframe with only the available required features
                X_pred = prediction_features[available_features].copy()
                
                # Fill any missing values with 0
                X_pred = X_pred.fillna(0)
                
                logger.info(f"Using {len(available_features)} features for WorkType {work_type}")
                
                # Make prediction
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
                    
                    # Create new row with all the necessary columns for feature engineering
                    new_row = {
                        'Date': next_date,
                        'WorkType': work_type,
                        'NoOfMan': pred_value,
                        'Hours': hours_value
                    }
                    
                    # Add productivity columns if they exist in original data
                    productivity_columns = ['SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI']
                    for col in productivity_columns:
                        if col in current_df.columns:
                            # Use recent average for productivity metrics
                            recent_data = current_df[current_df['WorkType'] == work_type].tail(7)
                            mean_value = recent_data[col].mean() if not recent_data.empty else 0
                            new_row[col] = mean_value if not pd.isna(mean_value) else 0
                    
                    new_rows.append(new_row)
                
                # Append new rows to the dataframe
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    current_df = pd.concat([current_df, new_df], ignore_index=True)
                    
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
    """Helper function to get feature names required by the model - uses tiered feature system"""
    try:
        # If it's a pipeline, get the feature names from the model
        if hasattr(model, 'steps'):
            # Try to get feature names directly from the model (e.g., RandomForest, etc.)
            if hasattr(model.named_steps['model'], 'feature_names_in_'):
                return list(model.named_steps['model'].feature_names_in_)
            
            # Otherwise, extract from preprocessor
            preprocessor = model.named_steps.get('preprocessor')
            if preprocessor:
                feature_names = []
                
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'num':
                            # Numeric features - add as-is
                            feature_names.extend(cols)
                        elif name == 'cat' and hasattr(transformer, 'categories_'):
                            # Categorical features - add original column names
                            feature_names.extend(cols)
                
                if feature_names:
                    return feature_names
        
        # Fallback: Use tiered feature system to get appropriate features
        from utils.feature_builder import FeatureBuilder
        from config import FEATURE_TIERS
        
        # Create a feature builder with conservative assumptions
        feature_builder = FeatureBuilder(
            data_columns=['Date', 'WorkType', 'NoOfMan', 'Hours', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI'],
            data_length=1000
        )
        
        # Get features based on current tier configuration
        numeric_features, categorical_features = feature_builder.get_feature_lists()
        
        # Combine and return
        all_features = categorical_features + numeric_features
        
        active_tiers = [tier for tier, enabled in FEATURE_TIERS.items() if enabled]
        logger.info(f"Using tiered features from {active_tiers}: {len(all_features)} total features")
        
        return all_features

    except Exception as e:
        logger.error(f"Error getting required features: {str(e)}")
        logger.error(traceback.format_exc())
        # Use pre-computed fallback features
        return _FEATURE_CACHE.get('fallback', [])
    