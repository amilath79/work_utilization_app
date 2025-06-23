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

    validated_params['max_depth'] = min(validated_params.get('max_depth', 8), 8)
    validated_params['num_leaves'] = min(validated_params.get('num_leaves', 31), 31)
    validated_params['min_child_samples'] = max(validated_params.get('min_child_samples', 20), 30)
    
    # Force stronger regularization
    validated_params['lambda_l1'] = max(validated_params.get('lambda_l1', 0.1), 0.5)
    validated_params['lambda_l2'] = max(validated_params.get('lambda_l2', 0.1), 0.5)
    
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
    Optimize LightGBM parameters for specific work type with anti-overfitting focus
    """
    try:
        # Base parameters with strong regularization
        base_params = {
            'objective': 'regression',
            'metric': ['mae', 'rmse'],
            'boosting_type': 'gbdt',
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1,
            'force_col_wise': True  # Better performance for many features
        }
        
        # Adjust parameters based on data size
        n_samples, n_features = X.shape
        
        if n_samples < 1000:
            # Small dataset - aggressive anti-overfitting
            base_params.update({
                'num_leaves': 15,           # Very simple trees
                'learning_rate': 0.05,      # Conservative learning
                'min_child_samples': 20,    # Large leaves
                'feature_fraction': 0.5,    # High randomness
                'bagging_fraction': 0.6,    
                'bagging_freq': 1,
                'lambda_l1': 1.0,           # Strong L1
                'lambda_l2': 1.0,           # Strong L2
                'min_gain_to_split': 0.02,  # High threshold
                'max_depth': 5              # Limit depth
            })
        elif n_samples < 5000:
            # Medium dataset - balanced approach
            base_params.update({
                'num_leaves': 25,           # Moderate complexity
                'learning_rate': 0.03,      
                'min_child_samples': 25,    
                'feature_fraction': 0.6,    
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'lambda_l1': 0.5,           
                'lambda_l2': 0.5,
                'min_gain_to_split': 0.01,
                'max_depth': 8
            })
        else:
            # Large dataset - still conservative
            base_params.update({
                'num_leaves': 31,           # Standard complexity
                'learning_rate': 0.02,      # Very slow learning
                'min_child_samples': 30,    
                'feature_fraction': 0.7,    
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'lambda_l1': 0.3,
                'lambda_l2': 0.3,
                'min_gain_to_split': 0.005,
                'max_depth': 10
            })
        
        # Additional regularization based on feature count
        if n_features > 50:
            base_params['feature_fraction'] = max(0.4, base_params['feature_fraction'] - 0.1)
            base_params['lambda_l1'] = base_params['lambda_l1'] * 1.5
            base_params['lambda_l2'] = base_params['lambda_l2'] * 1.5
            base_params['path_smooth'] = 1.0  # Extra smoothing for many features
        
        # Add data-dependent parameters
        base_params['min_data_per_group'] = max(100, n_samples // 100)
        base_params['cat_smooth'] = max(10, n_samples // 1000)
        
        logger.info(f"Optimized anti-overfitting LightGBM parameters for {work_type} "
                   f"(samples: {n_samples}, features: {n_features})")
        
        return base_params
        
    except Exception as e:
        logger.error(f"Error optimizing LightGBM for {work_type}: {str(e)}")
        # Return safe anti-overfitting defaults
        return {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 20,
            'learning_rate': 0.05,
            'min_child_samples': 30,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.7,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'random_state': 42,
            'verbosity': -1
        }