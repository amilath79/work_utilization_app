"""
Utility functions for ensemble modeling approaches.
"""
import numpy as np
import logging
import os
import pickle
from datetime import datetime, timedelta

from utils.prediction import predict_next_day, predict_multiple_days

# Configure logger
logger = logging.getLogger(__name__)

def check_pytorch_availability():
    """
    Check if PyTorch is available and can be imported without errors
    """
    try:
        import torch
        torch_version = torch.__version__
        logger.info(f"PyTorch is available (version: {torch_version})")
        return True
    except ImportError:
        logger.warning("PyTorch is not available. Neural network features will be disabled.")
        return False
    except Exception as e:
        logger.error(f"Error checking PyTorch availability: {str(e)}")
        return False

def ensemble_predict(df, rf_models, nn_available=None, weights=None, date=None):
    """
    Make predictions using an ensemble of models with optional weighting
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    rf_models : dict
        Dictionary of trained RandomForest models
    nn_available : bool, optional
        Whether neural networks are available
    weights : dict, optional
        Dictionary of weights for each model type {'rf': 0.5, 'nn': 0.5}
    date : datetime, optional
        Date for prediction, or None for latest date in data
    
    Returns:
    --------
    tuple
        (next_date, predictions)
    """
    # Check neural network availability if not provided
    if nn_available is None:
        nn_available = check_pytorch_availability()
    
    # Set default weights if not provided (equal weighting)
    if weights is None:
        if nn_available:
            weights = {'rf': 0.5, 'nn': 0.5}
        else:
            weights = {'rf': 1.0, 'nn': 0.0}
    
    # Get predictions from RF model
    next_date, rf_predictions = predict_next_day(df, rf_models, date=date, use_neural_network=False)
    
    # Get predictions from neural network model if available
    if nn_available and weights.get('nn', 0) > 0:
        try:
            _, nn_predictions = predict_next_day(df, rf_models, date=date, use_neural_network=True)
            
            # Combine predictions with weighting
            ensemble_preds = {}
            for work_type, rf_pred in rf_predictions.items():
                # Use both predictions if available, or just the RF prediction
                nn_pred = nn_predictions.get(work_type)
                if nn_pred is not None:
                    # Weighted average of predictions
                    weighted_pred = (weights['rf'] * rf_pred) + (weights['nn'] * nn_pred)
                    ensemble_preds[work_type] = weighted_pred
                else:
                    # Fall back to just RF prediction if NN not available for this work type
                    ensemble_preds[work_type] = rf_pred
            
            return next_date, ensemble_preds
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            # Fall back to RF only if there's an error
            return next_date, rf_predictions
    else:
        # Just return RF predictions if NN not available or not requested
        return next_date, rf_predictions

def ensemble_predict_multiple(df, rf_models, num_days=7, nn_available=None, weights=None):
    """
    Make ensemble predictions for multiple days
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    rf_models : dict
        Dictionary of trained RandomForest models
    num_days : int
        Number of days to predict
    nn_available : bool, optional
        Whether neural networks are available
    weights : dict, optional
        Dictionary of weights for each model type {'rf': 0.5, 'nn': 0.5}
    
    Returns:
    --------
    dict
        Dictionary of predictions for each date
    """
    # Check neural network availability if not provided
    if nn_available is None:
        nn_available = check_pytorch_availability()
    
    # Set default weights if not provided (equal weighting)
    if weights is None:
        if nn_available:
            weights = {'rf': 0.5, 'nn': 0.5}
        else:
            weights = {'rf': 1.0, 'nn': 0.0}
    
    # Initialize results dictionary
    multi_day_predictions = {}
    
    # Create a working copy of the dataframe that we'll extend with predictions
    current_df = df.copy()
    
    # Find the latest date in the dataset
    latest_date = current_df['Date'].max()
    
    # Predict for each day
    for i in range(num_days):
        try:
            # Get the next date to predict
            pred_date = latest_date + timedelta(days=i+1)
            
            # Use ensemble prediction for this day
            _, predictions = ensemble_predict(
                current_df, 
                rf_models, 
                nn_available=nn_available,
                weights=weights,
                date=latest_date + timedelta(days=i)
            )
            
            # Store predictions
            multi_day_predictions[pred_date] = predictions
            
            # Add the predictions back to the dataframe for the next iteration
            new_rows = []
            for work_type, pred_value in predictions.items():
                new_row = {
                    'Date': pred_date,
                    'WorkType': work_type,
                    'NoOfMan': pred_value,
                    
                    # Add the date features
                    'DayOfWeek_feat': pred_date.dayofweek,
                    'Month_feat': pred_date.month,
                    'IsWeekend_feat': 1 if pred_date.dayofweek >= 5 else 0,
                    'Year_feat': pred_date.year,
                    'Quarter': (pred_date.month - 1) // 3 + 1,
                    'DayOfMonth': pred_date.day,
                    'WeekOfYear': pred_date.isocalendar()[1]
                }
                
                # Add necessary lag features for the next iteration
                new_rows.append(new_row)
            
            # Append new rows to the dataframe
            if new_rows:
                current_df = pd.concat([current_df, pd.DataFrame(new_rows)], ignore_index=True)
        
        except Exception as e:
            logger.error(f"Error predicting day {i+1}: {str(e)}")
            continue
    
    return multi_day_predictions

def auto_adjust_weights(df, rf_models, work_type, lookback_days=30):
    """
    Automatically determine optimal weights for ensemble models based on recent performance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    rf_models : dict
        Dictionary of trained RandomForest models
    work_type : str
        Work type to adjust weights for
    lookback_days : int
        Number of days to look back for performance evaluation
    
    Returns:
    --------
    dict
        Dictionary of weights {'rf': weight, 'nn': weight}
    """
    try:
        # Check if both model types are available
        nn_available = check_pytorch_availability()
        
        if not nn_available:
            # If neural network is not available, return weights for RF only
            return {'rf': 1.0, 'nn': 0.0}
        
        # Filter data for this work type
        work_type_data = df[df['WorkType'] == work_type]
        
        if len(work_type_data) < lookback_days + 7:  # Need some history
            # Not enough data, use equal weights
            return {'rf': 0.5, 'nn': 0.5}
        
        # Get the date range for evaluation
        max_date = work_type_data['Date'].max()
        min_date = max_date - timedelta(days=lookback_days)
        
        # Filter data to the evaluation period
        eval_data = work_type_data[(work_type_data['Date'] >= min_date) & 
                                   (work_type_data['Date'] <= max_date)]
        
        # Make predictions with both models
        rf_errors = []
        nn_errors = []
        
        for i in range(len(eval_data) - 7):  # Use a week of data to predict the next day
            # Get the date range for this prediction
            pred_date = eval_data.iloc[i+7]['Date']
            
            # Create a subset of data up to the prediction date
            temp_df = df[df['Date'] < pred_date]
            
            # Make predictions with both models
            try:
                # RF prediction
                _, rf_preds = predict_next_day(temp_df, rf_models, date=pred_date-timedelta(days=1), use_neural_network=False)
                rf_pred = rf_preds.get(work_type, None)
                
                # NN prediction
                _, nn_preds = predict_next_day(temp_df, rf_models, date=pred_date-timedelta(days=1), use_neural_network=True)
                nn_pred = nn_preds.get(work_type, None)
                
                # Get actual value
                actual = eval_data[eval_data['Date'] == pred_date]['NoOfMan'].values[0]
                
                # Calculate errors
                if rf_pred is not None:
                    rf_errors.append(abs(actual - rf_pred))
                
                if nn_pred is not None:
                    nn_errors.append(abs(actual - nn_pred))
            except:
                continue
        
        # Calculate average errors
        if not rf_errors or not nn_errors:
            # If no errors calculated, use equal weights
            return {'rf': 0.5, 'nn': 0.5}
        
        avg_rf_error = sum(rf_errors) / len(rf_errors)
        avg_nn_error = sum(nn_errors) / len(nn_errors)
        
        # Calculate weights inversely proportional to errors
        # The lower the error, the higher the weight
        total_error_inverse = (1/avg_rf_error) + (1/avg_nn_error)
        rf_weight = (1/avg_rf_error) / total_error_inverse
        nn_weight = (1/avg_nn_error) / total_error_inverse
        
        return {'rf': rf_weight, 'nn': nn_weight}
    
    except Exception as e:
        logger.error(f"Error adjusting weights: {str(e)}")
        # Fall back to equal weights
        return {'rf': 0.5, 'nn': 0.5}

def get_ensemble_prediction_with_confidence(df, rf_models, work_type, date=None, num_samples=10):
    """
    Get ensemble prediction with confidence interval for a specific work type
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data
    rf_models : dict
        Dictionary of trained RandomForest models
    work_type : str
        Work type to predict
    date : datetime, optional
        Date for prediction, or None for latest date in data
    num_samples : int
        Number of samples to use for confidence interval
    
    Returns:
    --------
    dict
        Dictionary with prediction, lower_bound, upper_bound
    """
    try:
        # Check if both model types are available
        nn_available = check_pytorch_availability()
        
        # Initialize list to store predictions
        predictions = []
        
        # Get base prediction
        next_date, pred_dict = ensemble_predict(
            df, 
            rf_models, 
            nn_available=nn_available,
            date=date
        )
        
        base_pred = pred_dict.get(work_type, None)
        
        if base_pred is None:
            return None
        
        # Add base prediction
        predictions.append(base_pred)
        
        # If neural networks are available, make predictions with different weights
        if nn_available:
            for i in range(num_samples - 1):
                # Use different random weights for each sample
                rf_weight = np.random.uniform(0.3, 0.7)
                weights = {'rf': rf_weight, 'nn': 1.0 - rf_weight}
                
                _, pred_dict = ensemble_predict(
                    df, 
                    rf_models, 
                    nn_available=True,
                    weights=weights,
                    date=date
                )
                
                pred = pred_dict.get(work_type, None)
                if pred is not None:
                    predictions.append(pred)
        
        # Calculate confidence interval
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_dev = np.std(predictions)
        
        # 95% confidence interval (approximately 2 standard deviations)
        lower_bound = max(0, mean_pred - 1.96 * std_dev)
        upper_bound = mean_pred + 1.96 * std_dev
        
        return {
            'prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std_dev': std_dev
        }
    
    except Exception as e:
        logger.error(f"Error generating prediction with confidence: {str(e)}")
        return None