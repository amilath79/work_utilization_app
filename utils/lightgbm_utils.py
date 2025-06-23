"""
LightGBM utilities for workforce prediction optimization
Enhanced callbacks and validation for time series workforce data
"""

import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def get_lightgbm_callbacks(verbose=False):
    """
    Get optimized LightGBM callbacks for workforce prediction
    
    Returns:
    --------
    list
        List of LightGBM callbacks
    """
    callbacks = [
        early_stopping(stopping_rounds=50, verbose=verbose),
    ]
    
    if verbose:
        callbacks.append(log_evaluation(period=100))
    
    return callbacks

def validate_lightgbm_params(params):
    """
    Validate LightGBM parameters for workforce prediction
    
    Parameters:
    -----------
    params : dict
        LightGBM parameters
        
    Returns:
    --------
    dict
        Validated parameters
    """
    validated_params = params.copy()
    
    # Ensure regression objective
    if 'objective' not in validated_params:
        validated_params['objective'] = 'regression'
    
    # Set default metrics for workforce prediction
    if 'metric' not in validated_params:
        validated_params['metric'] = ['mae', 'rmse']
    
    # Optimize for workforce time series
    if 'num_leaves' not in validated_params:
        validated_params['num_leaves'] = 50
    
    if 'learning_rate' not in validated_params:
        validated_params['learning_rate'] = 0.05
    
    # Feature sampling for stability
    if 'feature_fraction' not in validated_params:
        validated_params['feature_fraction'] = 0.8
    
    # Bagging for variance reduction
    if 'bagging_fraction' not in validated_params:
        validated_params['bagging_fraction'] = 0.8
        validated_params['bagging_freq'] = 5
    
    # Regularization to prevent overfitting
    if 'lambda_l1' not in validated_params:
        validated_params['lambda_l1'] = 0.1
    if 'lambda_l2' not in validated_params:
        validated_params['lambda_l2'] = 0.1
    
    # Suppress verbose output by default
    if 'verbosity' not in validated_params:
        validated_params['verbosity'] = -1
    
    logger.info(f"Validated LightGBM parameters for workforce prediction")
    return validated_params

def get_feature_importance_lightgbm(model, feature_names, importance_type='gain'):
    """
    Extract feature importance from LightGBM model
    
    Parameters:
    -----------
    model : LGBMRegressor
        Trained LightGBM model
    feature_names : list
        List of feature names
    importance_type : str
        Type of importance ('gain', 'split', 'split')
        
    Returns:
    --------
    dict
        Dictionary of feature importances
    """
    try:
        # Get importance values
        importance_values = model.feature_importances_
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance_values))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        logger.info(f"Extracted {len(importance_dict)} feature importances from LightGBM")
        return importance_dict
        
    except Exception as e:
        logger.error(f"Error extracting LightGBM feature importance: {str(e)}")
        return {}

def lightgbm_cross_validate_fold(model, X_train, X_val, y_train, y_val):
    """
    Perform single fold validation for LightGBM with workforce-specific metrics
    
    Parameters:
    -----------
    model : LGBMRegressor
        LightGBM model to train
    X_train, X_val : array-like
        Training and validation features
    y_train, y_val : array-like
        Training and validation targets
        
    Returns:
    --------
    dict
        Fold metrics
    """
    try:
        # Fit model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=get_lightgbm_callbacks(verbose=False)
        )
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate workforce-specific metrics
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Custom MAPE for workforce data
        mape = np.mean(np.abs((y_val - y_pred) / np.where(y_val == 0, 1, y_val))) * 100
        
        return {
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'n_estimators_used': model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        }
        
    except Exception as e:
        logger.error(f"Error in LightGBM fold validation: {str(e)}")
        return {'MAE': np.inf, 'R2': -np.inf, 'MAPE': np.inf}

def optimize_lightgbm_for_worktype(X, y, work_type):
    """
    Optimize LightGBM parameters for specific work type
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    work_type : str
        Work type identifier
        
    Returns:
    --------
    dict
        Optimized parameters for the work type
    """
    try:
        # Base parameters
        base_params = {
            'objective': 'regression',
            'metric': ['mae', 'rmse'],
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        # Adjust parameters based on data size
        n_samples, n_features = X.shape
        
        if n_samples < 1000:
            # Small dataset - prevent overfitting
            base_params.update({
                'num_leaves': 20,
                'learning_rate': 0.1,
                'min_child_samples': 10,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7
            })
        elif n_samples < 5000:
            # Medium dataset - balanced approach
            base_params.update({
                'num_leaves': 35,
                'learning_rate': 0.08,
                'min_child_samples': 15,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8
            })
        else:
            # Large dataset - can handle more complexity
            base_params.update({
                'num_leaves': 50,
                'learning_rate': 0.05,
                'min_child_samples': 20,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.9
            })
        
        # Add regularization based on feature count
        if n_features > 50:
            base_params['lambda_l1'] = 0.1
            base_params['lambda_l2'] = 0.1
        else:
            base_params['lambda_l1'] = 0.05
            base_params['lambda_l2'] = 0.05
        
        logger.info(f"Optimized LightGBM parameters for {work_type} "
                   f"(samples: {n_samples}, features: {n_features})")
        
        return base_params
        
    except Exception as e:
        logger.error(f"Error optimizing LightGBM for {work_type}: {str(e)}")
        # Return safe defaults
        return {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': -1
        }